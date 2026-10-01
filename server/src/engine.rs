// SPDX-License-Identifier: Apache-2.0
//! The engine process and its JSON-lines protocol (docs/design.md). A reader task routes the
//! engine's events to the request they belong to; the engine itself queues generate requests in
//! arrival order and serves one at a time.
//!
//! When the engine's output ends (it exited, or crashed), the engine is down for good: requests
//! in flight get an error event, and new ones fail with `EngineDown`.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

/// The engine's first line.
#[derive(Debug, Deserialize)]
pub struct Ready {
    pub version: String,
    pub arch: String,
    pub max_context: u64,
    #[serde(default)]
    pub features: Vec<String>,
}

type Routes = Arc<StdMutex<HashMap<String, mpsc::UnboundedSender<Value>>>>;

/// The engine process is gone; a request cannot be served (HTTP 503).
#[derive(Debug)]
pub struct EngineDown;

impl std::fmt::Display for EngineDown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the engine is not running")
    }
}

impl std::error::Error for EngineDown {}

pub struct Engine {
    pub ready: Ready,
    stdin: Mutex<ChildStdin>,
    routes: Routes,
    alive: Arc<AtomicBool>, // false once the engine's output ended; changed under the routes lock
    quitting: Arc<AtomicBool>,
    down: Arc<tokio::sync::Notify>,
    child: Mutex<Child>,
    next_id: AtomicU64,
}

pub struct GenerateParams {
    pub prompt: Vec<u32>,
    pub max_new: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub seed: u64,
    pub stop_ids: Vec<u32>,
}

impl Engine {
    pub async fn spawn(exe: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(exe)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting engine {exe}"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("engine has no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("engine has no stdout"))?;
        let mut lines = BufReader::new(stdout).lines();
        let ready = loop {
            let line = lines.next_line().await?.ok_or_else(|| anyhow!("engine exited before it was ready"))?;
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            match v.get("ev").and_then(Value::as_str) {
                Some("ready") => {
                    let ready = serde_json::from_value::<Ready>(v)?;
                    // the features say what this engine's protocol supports (docs/design.md)
                    tracing::info!(version = %ready.version, arch = %ready.arch, max_context = ready.max_context,
                                   features = ?ready.features, "engine ready");
                    break ready;
                }
                Some("error") => return Err(anyhow!("engine: {}", v.get("msg").and_then(Value::as_str).unwrap_or("error"))),
                _ => {}
            }
        };
        let routes: Routes = Arc::new(StdMutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let down = Arc::new(tokio::sync::Notify::new());
        let quitting = Arc::new(AtomicBool::new(false));
        let (r2, alive2, down2, quitting2) = (routes.clone(), alive.clone(), down.clone(), quitting.clone());
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(v) = serde_json::from_str::<Value>(&line) else {
                    tracing::warn!("engine: unparsable line: {line}");
                    continue;
                };
                let id = v.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                let ev = v.get("ev").and_then(Value::as_str).unwrap_or("").to_string();
                let mut routes = r2.lock().unwrap();
                if let Some(tx) = routes.get(&id) {
                    let _ = tx.send(v);
                    if ev == "done" || ev == "error" {
                        routes.remove(&id);
                    }
                } else if ev == "error" {
                    tracing::error!("engine: {}", v.get("msg").and_then(|x| x.as_str()).unwrap_or(""));
                }
            }
            if quitting2.load(Ordering::SeqCst) {
                tracing::info!("engine output ended");
            } else {
                tracing::error!("engine exited");
            }
            {
                let mut routes = r2.lock().unwrap();
                alive2.store(false, Ordering::SeqCst);
                for (_, tx) in routes.drain() {
                    let _ = tx.send(json!({"ev": "error", "msg": "engine exited"}));
                }
            }
            down2.notify_waiters();
        });
        Ok(Self { ready, stdin: Mutex::new(stdin), routes, alive, quitting, down, child: Mutex::new(child), next_id: AtomicU64::new(0) })
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Resolves once the engine is down.
    pub async fn wait_down(&self) {
        let n = self.down.notified();
        if self.alive() {
            n.await;
        }
    }

    async fn send(&self, op: &Value) -> Result<()> {
        let mut stdin = self.stdin.lock().await;
        let r = async {
            stdin.write_all(format!("{op}\n").as_bytes()).await?;
            stdin.flush().await
        }
        .await;
        r.map_err(|e| {
            tracing::error!("engine stdin: {e}");
            anyhow::Error::new(EngineDown)
        })
    }

    /// Asks the engine to finish the running request and exit; kills it after `grace`.
    pub async fn shutdown(&self, grace: std::time::Duration) {
        self.quitting.store(true, Ordering::SeqCst);
        if self.alive() {
            let _ = self.send(&json!({"op": "quit"})).await;
        }
        let mut child = self.child.lock().await;
        match tokio::time::timeout(grace, child.wait()).await {
            Ok(Ok(st)) => tracing::info!("engine exited: {st}"),
            _ => {
                tracing::warn!("engine did not exit in {grace:?}; killing it");
                let _ = child.kill().await;
            }
        }
    }

    /// Queues a generation; its events (token, progress, done, error) arrive on the receiver.
    pub async fn generate(&self, p: &GenerateParams) -> Result<(String, mpsc::UnboundedReceiver<Value>)> {
        let id = format!("r{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut routes = self.routes.lock().unwrap();
            if !self.alive() {
                return Err(EngineDown.into());
            }
            routes.insert(id.clone(), tx);
        }
        let op = json!({
            "op": "generate", "id": id, "prompt": p.prompt, "max_new": p.max_new,
            "sampling": {"temperature": p.temperature, "top_p": p.top_p, "top_k": p.top_k, "min_p": p.min_p, "seed": p.seed},
            "stop_ids": p.stop_ids,
        });
        if let Err(e) = self.send(&op).await {
            self.routes.lock().unwrap().remove(&id);
            return Err(e);
        }
        Ok((id, rx))
    }

    #[cfg(test)]
    pub fn has_route(&self, id: &str) -> bool {
        self.routes.lock().unwrap().contains_key(id)
    }

    /// Cancels a queued or running generation (its done event still arrives).
    pub async fn stop(&self, id: &str) {
        if let Err(e) = self.send(&json!({"op": "stop", "id": id})).await {
            tracing::warn!("engine stop failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const READY: &str = r#"{"ev":"ready","version":"0","arch":"test","max_context":64,"features":[]}"#;

    fn params() -> GenerateParams {
        GenerateParams { prompt: vec![1, 2], max_new: 4, temperature: 0.0, top_p: 1.0, top_k: 20, min_p: 0.0, seed: 1, stop_ids: vec![] }
    }

    #[tokio::test]
    async fn dead_engine_fails_fast() {
        // an engine that exits right after it is ready
        let e = Engine::spawn("sh", &["-c".into(), format!("echo '{READY}'")]).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), e.wait_down()).await.expect("down within 1 s");
        assert!(!e.alive());
        let err = e.generate(&params()).await.expect_err("an error");
        assert!(err.is::<EngineDown>());
    }

    #[tokio::test]
    async fn request_in_flight_gets_an_error_when_the_engine_dies() {
        // an engine that takes one request and dies without answering it
        let e = Engine::spawn("sh", &["-c".into(), format!("echo '{READY}'; read line; exit 1")]).await.unwrap();
        let (_, mut rx) = e.generate(&params()).await.unwrap();
        let ev = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await.expect("an event within 1 s").unwrap();
        assert_eq!(ev["ev"], "error");
        assert!(e.generate(&params()).await.expect_err("an error").is::<EngineDown>());
    }

    #[tokio::test]
    async fn shutdown_sends_quit() {
        // an engine that exits cleanly on quit
        let script = format!(r#"echo '{READY}'; while read line; do case "$line" in *quit*) exit 0;; esac; done; exit 7"#);
        let e = Engine::spawn("sh", &["-c".into(), script]).await.unwrap();
        e.shutdown(std::time::Duration::from_secs(2)).await;
        let st = e.child.lock().await.wait().await.unwrap();
        assert_eq!(st.code(), Some(0));
    }
}
