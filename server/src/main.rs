// SPDX-License-Identifier: Apache-2.0
//! flashrt-server: OpenAI/Anthropic-compatible HTTP front end for flashrt-engine.
//!
//! Phase 0: starts the engine, performs the protocol handshake, and serves /health and
//! /v1/models. Chat completions reach the engine but return 501 until phase 1 adds the
//! chat template, tokenizer and token streaming.

mod engine;

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
#[command(name = "flashrt-server", about = "OpenAI/Anthropic-compatible front end for flashrt-engine")]
struct Args {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Path to the flashrt-engine binary.
    #[arg(long)]
    engine: String,
    /// Arguments passed to the engine (repeatable).
    #[arg(long = "engine-arg", allow_hyphen_values = true)]
    engine_args: Vec<String>,
    /// Model name reported on /v1/models.
    #[arg(long, default_value = "flashrt")]
    model_name: String,
}

struct AppState {
    engine: engine::Engine,
    model_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let engine = engine::Engine::spawn(&args.engine, &args.engine_args).await?;
    tracing::info!(version = %engine.ready.version, arch = %engine.ready.arch, "engine ready");

    let state = Arc::new(AppState { engine, model_name: args.model_name });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{"id": s.model_name, "object": "model", "owned_by": "flashrt"}]
    }))
}

async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<Value>) -> impl IntoResponse {
    match s.engine.generate_probe(&req).await {
        Ok(msg) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": {"message": msg, "type": "not_implemented"}})),
        ),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": e.to_string(), "type": "engine_error"}})),
        ),
    }
}
