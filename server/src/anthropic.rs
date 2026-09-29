// SPDX-License-Identifier: Apache-2.0
//! Anthropic-compatible endpoints: /v1/messages (text, thinking and tool_use blocks, streaming)
//! and /v1/messages/count_tokens.
//!
//! Thinking: `thinking: {type: "enabled"}` returns the model's reasoning as thinking blocks;
//! `{type: "disabled"}` turns reasoning off in the template; without the field the model still
//! reasons (its default) but the reasoning is not returned.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::chat::{self, ChatEvent, ChatRequest, Finish};
use crate::{api_error, AppState};

fn anth_error(status: u16, kind: &str, msg: &str) -> Response {
    let mut r = Json(json!({"type": "error", "error": {"type": kind, "message": msg}})).into_response();
    *r.status_mut() = axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::BAD_REQUEST);
    r
}

fn stop_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::Cancelled => "end_turn",
        Finish::StopSequence => "stop_sequence",
        Finish::Length => "max_tokens",
        Finish::ToolCalls => "tool_use",
    }
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Anthropic request -> the template's OpenAI-style messages and tools, plus whether reasoning is
/// returned.
fn to_chat(req: &Value) -> Result<(ChatRequest, bool), String> {
    let mut msgs = Vec::new();
    if let Some(sys) = req.get("system") {
        let t = text_of(sys);
        if !t.is_empty() {
            msgs.push(json!({"role": "system", "content": t}));
        }
    }
    for m in req.get("messages").and_then(Value::as_array).ok_or("messages must be an array")? {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let blocks: Vec<Value> = match &content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(a) => a.clone(),
            _ => vec![],
        };
        match role {
            "user" => {
                let mut texts = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text" => texts.push(b.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                        "tool_result" => {
                            let mut c = text_of(b.get("content").unwrap_or(&Value::Null));
                            if b.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
                                c = format!("Error: {c}");
                            }
                            msgs.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                                             "content": c}));
                        }
                        "image" | "document" => return Err("image and document blocks are not supported".into()),
                        _ => {}
                    }
                }
                if !texts.is_empty() {
                    msgs.push(json!({"role": "user", "content": texts.join("\n")}));
                }
            }
            "assistant" => {
                let (mut text, mut thinking, mut calls) = (Vec::new(), Vec::new(), Vec::new());
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text" => text.push(b.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                        "thinking" => thinking.push(b.get("thinking").and_then(Value::as_str).unwrap_or("").to_string()),
                        "tool_use" => calls.push(json!({"id": b.get("id"), "type": "function",
                            "function": {"name": b.get("name"), "arguments": b.get("input").cloned().unwrap_or(json!({}))}})),
                        _ => {}
                    }
                }
                let mut am = json!({"role": "assistant", "content": text.join("\n")});
                if !thinking.is_empty() {
                    am["reasoning_content"] = json!(thinking.join("\n"));
                }
                if !calls.is_empty() {
                    am["tool_calls"] = json!(calls);
                }
                msgs.push(am);
            }
            _ => return Err(format!("unknown role '{role}'")),
        }
    }
    let mut r = crate::openai::empty_request();
    r.messages = Value::Array(msgs);
    let choice_none = req.pointer("/tool_choice/type").and_then(Value::as_str) == Some("none");
    if let Some(tools) = req.get("tools").and_then(Value::as_array) {
        let fns: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("input_schema").is_some())
            .map(|t| {
                json!({"type": "function", "function": {"name": t.get("name"), "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("input_schema")}})
            })
            .collect();
        if !fns.is_empty() && !choice_none {
            r.tools = Some(Value::Array(fns));
        }
    }
    r.max_tokens = req.get("max_tokens").and_then(Value::as_u64).map(|v| v as u32);
    r.temperature = req.get("temperature").and_then(Value::as_f64).map(|v| v as f32);
    r.top_p = req.get("top_p").and_then(Value::as_f64).map(|v| v as f32);
    r.top_k = req.get("top_k").and_then(Value::as_u64).map(|v| v as u32);
    r.stop = req
        .get("stop_sequences")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let show = match req.pointer("/thinking/type").and_then(Value::as_str) {
        Some("enabled") => true,
        Some("disabled") => {
            r.template_vars.insert("enable_thinking".into(), Value::Bool(false));
            false
        }
        _ => false,
    };
    Ok((r, show))
}

pub async fn count_tokens(st: Arc<AppState>, req: Value) -> Response {
    let (r, _) = match to_chat(&req) {
        Ok(x) => x,
        Err(e) => return anth_error(400, "invalid_request_error", &e),
    };
    match tokio::task::spawn_blocking(move || chat::prompt_of(&st, &r)).await {
        Ok(Ok((_, toks))) => Json(json!({"input_tokens": toks.len()})).into_response(),
        Ok(Err(e)) => anth_error(400, "invalid_request_error", &e.to_string()),
        Err(e) => anth_error(500, "api_error", &e.to_string()),
    }
}

pub async fn messages(st: Arc<AppState>, req: Value) -> Response {
    let (r, show_thinking) = match to_chat(&req) {
        Ok(x) => x,
        Err(e) => return anth_error(400, "invalid_request_error", &e),
    };
    let model = req.get("model").and_then(Value::as_str).unwrap_or(&st.model_name).to_string();
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let (mut rx, n_prompt) = match chat::start(st, r).await {
        Ok(x) => x,
        Err(e) if crate::start_error_status(&e) == 503 => return anth_error(503, "api_error", &e.to_string()),
        Err(e) => return anth_error(400, "invalid_request_error", &e.to_string()),
    };
    let id = chat::new_id("msg_");
    if stream {
        return stream_messages(rx, id, model, n_prompt, show_thinking).into_response();
    }
    let (mut thinking, mut text, mut tools) = (String::new(), String::new(), Vec::new());
    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Reasoning(s) => thinking.push_str(&s),
            ChatEvent::Content(s) => text.push_str(&s),
            ChatEvent::ToolCall { id, name, arguments } => tools.push(json!({"type": "tool_use", "id": id, "name": name, "input": arguments})),
            ChatEvent::Error(e) => return anth_error(500, "api_error", &e),
            ChatEvent::Done { finish, stop_sequence, prompt_tokens, completion_tokens, reused, timings } => {
                let mut content = Vec::new();
                if show_thinking && !thinking.is_empty() {
                    content.push(json!({"type": "thinking", "thinking": thinking, "signature": ""}));
                }
                if !text.is_empty() || tools.is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                }
                content.extend(tools);
                return Json(json!({
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
                    "stop_reason": stop_reason(finish), "stop_sequence": stop_sequence,
                    "usage": {"input_tokens": prompt_tokens - reused.min(prompt_tokens), "cache_read_input_tokens": reused,
                              "output_tokens": completion_tokens},
                    "timings": timings,
                }))
                .into_response();
            }
        }
    }
    api_error(500, "api_error", "generation ended without a result")
}

fn stream_messages(
    mut rx: mpsc::Receiver<ChatEvent>,
    id: String,
    model: String,
    n_prompt: u32,
    show_thinking: bool,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let ev = |name: &str, v: Value| Event::default().event(name).data(v.to_string());
        let mut index: i64 = -1;
        let mut open: Option<&'static str> = None;   // the type of the open block
        macro_rules! send {
            ($e:expr) => {
                if tx.send($e).await.is_err() {
                    return;
                }
            };
        }
        macro_rules! close {
            () => {
                if open.take().is_some() {
                    send!(ev("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                }
            };
        }
        macro_rules! open_block {
            ($kind:expr, $block:expr) => {
                if open != Some($kind) {
                    close!();
                    index += 1;
                    open = Some($kind);
                    send!(ev("content_block_start", json!({"type": "content_block_start", "index": index, "content_block": $block})));
                }
            };
        }
        send!(ev("message_start", json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": n_prompt, "output_tokens": 0}}})));
        send!(ev("ping", json!({"type": "ping"})));
        while let Some(e) = rx.recv().await {
            match e {
                ChatEvent::Reasoning(s) => {
                    if !show_thinking {
                        continue;
                    }
                    open_block!("thinking", json!({"type": "thinking", "thinking": ""}));
                    send!(ev("content_block_delta", json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "thinking_delta", "thinking": s}})));
                }
                ChatEvent::Content(s) => {
                    if open == Some("thinking") {
                        send!(ev("content_block_delta", json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "signature_delta", "signature": ""}})));
                    }
                    open_block!("text", json!({"type": "text", "text": ""}));
                    send!(ev("content_block_delta", json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": s}})));
                }
                ChatEvent::ToolCall { id: tid, name, arguments } => {
                    close!();
                    index += 1;
                    send!(ev("content_block_start", json!({"type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}})));
                    send!(ev("content_block_delta", json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments.to_string()}})));
                    send!(ev("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                }
                ChatEvent::Error(e) => {
                    send!(ev("error", json!({"type": "error", "error": {"type": "api_error", "message": e}})));
                    return;
                }
                ChatEvent::Done { finish, stop_sequence, completion_tokens, .. } => {
                    if index < 0 {
                        open_block!("text", json!({"type": "text", "text": ""}));
                    }
                    close!();
                    send!(ev("message_delta", json!({"type": "message_delta",
                        "delta": {"stop_reason": stop_reason(finish), "stop_sequence": stop_sequence},
                        "usage": {"output_tokens": completion_tokens}})));
                    send!(ev("message_stop", json!({"type": "message_stop"})));
                    return;
                }
            }
        }
    });
    Sse::new(ReceiverStream::new(out).map(Ok)).keep_alive(KeepAlive::default())
}
