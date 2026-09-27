// SPDX-License-Identifier: Apache-2.0
//! The engine process and its JSON-lines protocol (docs/design.md).

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

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

struct Pipe {
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

pub struct Engine {
    pub ready: Ready,
    // One sequence at a time: requests queue on this lock, in arrival order.
    pipe: Mutex<Pipe>,
    _child: Child,
    next_id: AtomicU64,
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
            let line = lines
                .next_line()
                .await?
                .ok_or_else(|| anyhow!("engine exited before it was ready"))?;
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if v.get("ev").and_then(Value::as_str) == Some("ready") {
                break serde_json::from_value::<Ready>(v)?;
            }
        };
        Ok(Self { ready, pipe: Mutex::new(Pipe { stdin, lines }), _child: child, next_id: AtomicU64::new(0) })
    }

    /// Sends a generate op and returns the engine's first reply message. Until phase 1 the
    /// engine answers with an error event; this keeps the queueing and the pipe exercised.
    pub async fn generate_probe(&self, _request: &Value) -> Result<String> {
        let id = format!("r{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let mut pipe = self.pipe.lock().await;
        let op = json!({"op": "generate", "id": id, "prompt": [], "max_new": 1});
        pipe.stdin.write_all(format!("{op}\n").as_bytes()).await?;
        pipe.stdin.flush().await?;
        let line = pipe.lines.next_line().await?.ok_or_else(|| anyhow!("engine exited"))?;
        let v: Value = serde_json::from_str(&line)?;
        Ok(v.get("msg").and_then(Value::as_str).unwrap_or("unexpected engine reply").to_string())
    }
}
