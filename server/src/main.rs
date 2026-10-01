// SPDX-License-Identifier: Apache-2.0
//! flashrt-server: OpenAI- and Anthropic-compatible HTTP front end for flashrt-engine.
//!
//! The server owns the text side: the tokenizer and chat template (read from the model's GGUF),
//! tool-call and reasoning parsing, stop strings and the HTTP APIs. The engine process owns the
//! model and speaks token ids (docs/design.md). One sequence runs at a time; requests queue.
//!
//! If the engine exits, requests get 503 and /health reports it; the server exits with status 1
//! a few seconds later, for a supervisor to restart both. On SIGINT or SIGTERM it stops taking
//! requests, asks the engine to quit, and exits.
//!
//!   flashrt-server --model MODEL.gguf --engine build/flashrt-engine --engine-arg MODEL.gguf [--engine-arg ...]
//!   flashrt-server --model MODEL.gguf --check-tokenizer TEXT IDS    (compare with a llama.cpp tokenization)
//!   flashrt-server --model MODEL.gguf --render REQUEST.json         (print a chat request's prompt)
//!   flashrt-server --model MODEL.gguf --tokenize TEXT               (print a text's token ids)

mod anthropic;
mod chat;
mod engine;
mod gguf;
mod metrics;
mod openai;
mod template;
mod tokenizer;

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde_json::{json, Value};

/// The largest request body accepted.
const BODY_LIMIT: usize = 64 << 20;

#[derive(Parser, Debug)]
#[command(name = "flashrt-server", about = "OpenAI/Anthropic-compatible front end for flashrt-engine")]
struct Args {
    /// The model's GGUF (first shard): tokenizer, chat template and sampling defaults.
    #[arg(long)]
    model: String,
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Path to the flashrt-engine binary.
    #[arg(long)]
    engine: Option<String>,
    /// Arguments passed to the engine (repeatable), starting with the model path.
    #[arg(long = "engine-arg", allow_hyphen_values = true)]
    engine_args: Vec<String>,
    /// Model name reported by the APIs.
    #[arg(long, default_value = "flashrt")]
    model_name: String,
    /// Require this key (Authorization: Bearer KEY, or x-api-key: KEY).
    #[arg(long)]
    api_key: Option<String>,
    /// Default output limit when a request sets none.
    #[arg(long, default_value_t = 32768)]
    max_tokens: u32,
    /// Compare the tokenizer with a reference tokenization: a text file and its token ids.
    #[arg(long, num_args = 2, value_names = ["TEXT", "IDS"])]
    check_tokenizer: Option<Vec<String>>,
    /// Print the prompt (and its token count) of an OpenAI chat request in a JSON file.
    #[arg(long)]
    render: Option<String>,
    /// Print the token ids of a text file (special tokens' strings become their ids), for benchmark
    /// inputs.
    #[arg(long)]
    tokenize: Option<String>,
    /// Tokenize special-token strings typed in the text of user, system and tool messages and of
    /// tool definitions (<|im_start|>, <think>, <tool_call>, ...) as the special tokens themselves,
    /// as llama.cpp's server does. By default they are plain text, so a message cannot forge the
    /// conversation's structure; the template's own special tokens and those in assistant messages
    /// are special either way.
    #[arg(long)]
    special_in_text: bool,
}

pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
}

pub struct SpecialIds {
    pub think: u32,
    pub think_end: u32,
    pub tool_call: u32,
    pub tool_call_end: u32,
}

pub struct AppState {
    pub engine: engine::Engine,
    pub tokenizer: tokenizer::Tokenizer,
    pub template: template::ChatTemplate,
    pub model_name: String,
    pub max_context: u64,
    pub default_max_tokens: u32,
    pub sampling: Sampling,
    pub stop_ids: Vec<u32>,
    pub ids: SpecialIds,
    pub api_key: Option<String>,
    pub metrics: metrics::Metrics,
    /// --special-in-text: special-token strings in message text are special tokens (chat::prompt_of).
    pub special_in_text: bool,
}

/// A request that failed for the server's reasons, not the request's (a chat template that cannot
/// render a well-formed request, a panic while preparing the prompt): HTTP 500.
#[derive(Debug)]
pub struct ServerFault(pub String);

impl std::fmt::Display for ServerFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ServerFault {}

/// The HTTP status of a request that could not start: 503 when the engine is down, 500 for a
/// `ServerFault`, else 400.
pub fn start_error_status(e: &anyhow::Error) -> u16 {
    if e.is::<engine::EngineDown>() {
        503
    } else if e.is::<ServerFault>() {
        500
    } else {
        400
    }
}

/// Which API's error shape a response takes.
#[derive(Clone, Copy)]
enum Api {
    OpenAi,
    Anthropic,
}

/// A request body: a JSON object, or the status and message of its error. A body that is not
/// JSON, not an object, too large or sent without a JSON content type gets the API's error shape
/// (`body_error`) with axum's status (400, 413, 415) instead of axum's plain-text rejection.
fn json_body(body: Result<Json<Value>, JsonRejection>) -> Result<Value, (u16, String)> {
    match body {
        Ok(Json(v)) if v.is_object() => Ok(v),
        Ok(_) => Err((400, "the request body must be a JSON object".to_string())),
        Err(r) => Err((r.status().as_u16(), r.body_text())),
    }
}

fn body_error(api: Api, (status, msg): (u16, String)) -> Response {
    match api {
        Api::OpenAi => api_error(status, "invalid_request_error", &msg),
        Api::Anthropic => anthropic::error(status, if status == 413 { "request_too_large" } else { "invalid_request_error" }, &msg),
    }
}

pub fn api_error(status: u16, kind: &str, msg: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (code, Json(json!({"error": {"message": msg, "type": kind}}))).into_response()
}

fn check_tokenizer(tok: &tokenizer::Tokenizer, text: &str, ids: &str) -> Result<()> {
    let text = std::fs::read_to_string(text)?;
    let reference: Vec<u32> = std::fs::read_to_string(ids)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let cut: String = text.chars().take(200_000).collect();
    let t0 = std::time::Instant::now();
    let ours = tok.encode(&cut, false)?;
    let dt = t0.elapsed().as_secs_f64();
    let n = ours.len().min(reference.len()).saturating_sub(64);   // the text was cut mid-token
    let same = ours[..n].iter().zip(&reference[..n]).filter(|(a, b)| a == b).count();
    let first_diff = ours[..n].iter().zip(&reference[..n]).position(|(a, b)| a != b);
    println!("tokenizer: {same} of {n} ids match the reference ({} tokens in {:.3} s); first difference at {first_diff:?}", ours.len(), dt);
    // round trip
    let mut dec = tokenizer::Decoder::default();
    let mut back = String::new();
    for &t in &ours {
        back.push_str(&dec.push(tok.token_bytes(t)));
    }
    back.push_str(&dec.finish());
    println!("decode round trip: {}", if back == cut { "identical" } else { "DIFFERENT" });
    if same != n || back != cut {
        bail!("tokenizer check failed");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let kv = gguf::read_metadata(&args.model)?;
    let tok = tokenizer::Tokenizer::from_gguf(&kv)?;
    if let Some(c) = &args.check_tokenizer {
        return check_tokenizer(&tok, &c[0], &c[1]);
    }
    if let Some(path) = &args.tokenize {
        let ids = tok.encode(&std::fs::read_to_string(path)?, true)?;
        println!("{}", ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        return Ok(());
    }
    let tmpl_src = kv.get("tokenizer.chat_template").and_then(gguf::Value::as_str).ok_or_else(|| anyhow!("the GGUF has no chat template"))?;
    let template = template::ChatTemplate::new(tmpl_src)?;
    let id_of = |s: &str| tok.token_id(s).ok_or_else(|| anyhow!("the vocabulary has no {s}"));
    let ids = SpecialIds { think: id_of("<think>")?, think_end: id_of("</think>")?, tool_call: id_of("<tool_call>")?, tool_call_end: id_of("</tool_call>")? };
    let mut stop_ids = vec![tok.eos];
    for s in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tok.token_id(s) {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
    }
    let f = |k: &str| kv.get(k).and_then(|v| match v {
        gguf::Value::Float(x) => Some(*x),
        gguf::Value::Int(x) => Some(*x as f64),
        _ => None,
    });
    let sampling = Sampling {
        temperature: f("general.sampling.temp").unwrap_or(1.0) as f32,
        top_p: f("general.sampling.top_p").unwrap_or(0.95) as f32,
        top_k: f("general.sampling.top_k").unwrap_or(20.0) as u32,
        min_p: f("general.sampling.min_p").unwrap_or(0.0) as f32,
    };
    if let Some(path) = &args.render {
        let req: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let mut r = openai::empty_request();
        r.messages = req.get("messages").cloned().unwrap_or(json!([]));
        r.tools = req.get("tools").cloned();
        if let Some(Value::Object(kw)) = req.get("chat_template_kwargs") {
            r.template_vars = kw.clone();
        }
        let (text, toks) = chat::render_prompt(&template, &tok, &r, args.special_in_text)?;
        println!("{text}");
        println!("--- {} tokens; first ids {:?}", toks.len(), &toks[..toks.len().min(12)]);
        return Ok(());
    }

    let exe = args.engine.as_deref().context("--engine is required")?;
    let engine = engine::Engine::spawn(exe, &args.engine_args).await?;
    let state = Arc::new(AppState {
        max_context: engine.ready.max_context,
        engine,
        tokenizer: tok,
        template,
        model_name: args.model_name.clone(),
        default_max_tokens: args.max_tokens,
        sampling,
        stop_ids,
        ids,
        api_key: args.api_key.clone(),
        metrics: metrics::Metrics::default(),
        special_in_text: args.special_in_text,
    });
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(|State(s): State<Arc<AppState>>, b: Result<Json<Value>, JsonRejection>| async move {
            match json_body(b) {
                Ok(v) => openai::chat_completions(s, v).await,
                Err(e) => body_error(Api::OpenAi, e),
            }
        }))
        .route("/v1/completions", post(|State(s): State<Arc<AppState>>, b: Result<Json<Value>, JsonRejection>| async move {
            match json_body(b) {
                Ok(v) => openai::completions(s, v).await,
                Err(e) => body_error(Api::OpenAi, e),
            }
        }))
        .route("/v1/messages", post(|State(s): State<Arc<AppState>>, b: Result<Json<Value>, JsonRejection>| async move {
            match json_body(b) {
                Ok(v) => anthropic::messages(s, v).await,
                Err(e) => body_error(Api::Anthropic, e),
            }
        }))
        .route("/v1/messages/count_tokens", post(|State(s): State<Arc<AppState>>, b: Result<Json<Value>, JsonRejection>| async move {
            match json_body(b) {
                Ok(v) => anthropic::count_tokens(s, v).await,
                Err(e) => body_error(Api::Anthropic, e),
            }
        }))
        .route("/metrics", get(metrics_text))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // axum's default of 2 MiB is too little for a full context: 262K tokens of text with
        // tool definitions and JSON escaping come to several MiB
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .route("/health", get(health))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    let s2 = state.clone();
    tokio::spawn(async move {
        s2.engine.wait_down().await;
        tracing::error!("the engine is down; exiting in 3 s");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;   // requests in flight get their errors
        std::process::exit(1);
    });
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    tracing::info!("shutting down");
    state.engine.shutdown(std::time::Duration::from_secs(30)).await;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}

async fn health(State(s): State<Arc<AppState>>) -> Response {
    if s.engine.alive() {
        Json(json!({"status": "ok"})).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "engine down"}))).into_response()
    }
}

/// The running totals over finished generations, in Prometheus's text format (metrics.rs).
async fn metrics_text(State(s): State<Arc<AppState>>) -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], s.metrics.render(s.engine.alive())).into_response()
}

async fn auth(State(s): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(key) = &s.api_key {
        let h = req.headers();
        let bearer = h.get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
        let x = h.get("x-api-key").and_then(|v| v.to_str().ok());
        if bearer != Some(key.as_str()) && x != Some(key.as_str()) {
            // in the shape of the API that was called
            return if req.uri().path().starts_with("/v1/messages") {
                anthropic::error(401, "authentication_error", "invalid API key")
            } else {
                api_error(401, "authentication_error", "invalid API key")
            };
        }
    }
    next.run(req).await
}

async fn models(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{"id": s.model_name, "object": "model", "owned_by": "flashrt", "context_length": s.max_context}]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::FromRequest;

    async fn reject(body: &'static str, content_type: Option<&str>, api: Api) -> (u16, Value) {
        let mut req = axum::http::Request::builder().method("POST").uri("/x");
        if let Some(ct) = content_type {
            req = req.header("content-type", ct);
        }
        let b = Json::<Value>::from_request(req.body(Body::from(body)).unwrap(), &()).await;
        let r = body_error(api, json_body(b).expect_err("rejected"));
        let status = r.status().as_u16();
        let bytes = axum::body::to_bytes(r.into_body(), 1 << 16).await.unwrap();
        (status, serde_json::from_slice(&bytes).expect("a JSON error body"))
    }

    #[tokio::test]
    async fn bad_bodies_get_the_apis_json_error() {
        let json = Some("application/json");
        // not JSON: a 400 in the OpenAI shape for the OpenAI routes, Anthropic's for the others
        let (st, v) = reject("{not json", json, Api::OpenAi).await;
        assert_eq!(st, 400);
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{v}");
        let (st, v) = reject("{not json", json, Api::Anthropic).await;
        assert_eq!((st, v["type"].as_str(), v["error"]["type"].as_str()), (400, Some("error"), Some("invalid_request_error")));
        // JSON, but not an object
        for body in ["[1, 2]", "\"text\"", "null"] {
            let (st, v) = reject(body, json, Api::OpenAi).await;
            assert_eq!(st, 400);
            assert_eq!(v["error"]["message"], "the request body must be a JSON object");
        }
        // no JSON content type: axum's 415, in the API's shape
        let (st, v) = reject("{}", None, Api::Anthropic).await;
        assert_eq!((st, v["error"]["type"].as_str()), (415, Some("invalid_request_error")));
        // an object passes
        let b = Json::<Value>::from_request(
            axum::http::Request::builder().method("POST").header("content-type", "application/json").body(Body::from("{\"a\": 1}")).unwrap(),
            &(),
        )
        .await;
        assert_eq!(json_body(b).ok(), Some(json!({"a": 1})));
    }
}
