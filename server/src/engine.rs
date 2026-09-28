// SPDX-License-Identifier: Apache-2.0
//! The engine process and its JSON-lines protocol (docs/design.md). A reader task routes the
//! engine's events to the request they belong to; the engine itself queues generate requests in
//! arrival order and serves one at a time.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

/// The engine's first line.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Ready {
    pub version: String,
    pub arch: String,
    pub max_context: u64,
    #[serde(default)]
    pub features: Vec<String>,
}

type Routes = Arc<StdMutex<HashMap<String, mpsc::UnboundedSender<Value>>>>;

pub struct Engine {
    pub ready: Ready,
    stdin: Mutex<ChildStdin>,
    routes: Routes,
    _child: Child,
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
                Some("ready") => break serde_json::from_value::<Ready>(v)?,
                Some("error") => return Err(anyhow!("engine: {}", v.get("msg").and_then(Value::as_str).unwrap_or("error"))),
                _ => {}
            }
        };
        let routes: Routes = Arc::new(StdMutex::new(HashMap::new()));
        let r2 = routes.clone();
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
                    tracing::error!("engine: {}", v.get("msg").and_then(Value::as_str).unwrap_or(""));
                }
            }
            tracing::error!("engine exited");
            let mut routes = r2.lock().unwrap();
            for (_, tx) in routes.drain() {
                let _ = tx.send(json!({"ev": "error", "msg": "engine exited"}));
            }
        });
        Ok(Self { ready, stdin: Mutex::new(stdin), routes, _child: child, next_id: AtomicU64::new(0) })
    }

    async fn send(&self, op: &Value) -> Result<()> {
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(format!("{op}\n").as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    /// Queues a generation; its events (token, progress, done, error) arrive on the receiver.
    pub async fn generate(&self, p: &GenerateParams) -> Result<(String, mpsc::UnboundedReceiver<Value>)> {
        let id = format!("r{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::unbounded_channel();
        self.routes.lock().unwrap().insert(id.clone(), tx);
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

    /// Cancels a queued or running generation (its done event still arrives).
    pub async fn stop(&self, id: &str) {
        if let Err(e) = self.send(&json!({"op": "stop", "id": id})).await {
            tracing::warn!("engine stop failed: {e}");
        }
    }
}
