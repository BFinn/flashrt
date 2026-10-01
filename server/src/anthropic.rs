// SPDX-License-Identifier: Apache-2.0
//! Anthropic-compatible endpoints: /v1/messages (text, thinking and tool_use blocks, streaming)
//! and /v1/messages/count_tokens.
//!
//! Thinking: `thinking: {type: "enabled"}` returns the model's reasoning as thinking blocks;
//! `{type: "disabled"}` turns reasoning off in the template; without the field the model still
//! reasons (its default) but the reasoning is not returned.
//!
//! Usage: `input_tokens` counts the prompt tokens not reused from the engine's cache and
//! `cache_read_input_tokens` the reused ones, as Anthropic's API counts them. A stream's
//! message_start cannot know the reuse yet (the engine reports it when done): it carries the whole
//! prompt as input_tokens, and the final message_delta carries the request's usage, the same as a
//! non-streaming response's.
//!
//! Not implemented, and a 400: image, document and other non-text content blocks, and tools
//! without an input_schema (server tools). Accepted and ignored: metadata, service_tier,
//! thinking.budget_tokens, tool_choice "any" or a named tool and disable_parallel_tool_use (the
//! model decides), cache_control, and redacted_thinking blocks in the history.

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
use crate::AppState;

pub(crate) fn error(status: u16, kind: &str, msg: &str) -> Response {
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

/// A finished request's usage: the prompt tokens prefilled, those reused, and those generated.
fn usage(prompt_tokens: u32, reused: u32, completion_tokens: u32) -> Value {
    json!({"input_tokens": prompt_tokens - reused.min(prompt_tokens), "cache_read_input_tokens": reused, "output_tokens": completion_tokens})
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

/// The error response of a request that could not start (chat::start).
fn start_error(e: &anyhow::Error) -> Response {
    match crate::start_error_status(e) {
        400 => error(400, "invalid_request_error", &e.to_string()),
        st => error(st, "api_error", &e.to_string()),
    }
}

/// Anthropic request -> the template's OpenAI-style messages and tools, plus whether reasoning is
/// returned.
pub(crate) fn to_chat(req: &Value) -> Result<(ChatRequest, bool), String> {
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
                            let inner = b.get("content").unwrap_or(&Value::Null);
                            for ib in inner.as_array().into_iter().flatten() {
                                match ib.get("type").and_then(Value::as_str).unwrap_or("") {
                                    "text" => {}
                                    "image" | "document" => return Err("image and document blocks are not supported".into()),
                                    t => return Err(format!("tool_result content block type '{t}' is not supported")),
                                }
                            }
                            let mut c = text_of(inner);
                            if b.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
                                c = format!("Error: {c}");
                            }
                            msgs.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                                             "content": c}));
                        }
                        "image" | "document" => return Err("image and document blocks are not supported".into()),
                        t => return Err(format!("content block type '{t}' is not supported in a user message")),
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
                        "tool_use" => {
                            let name = b.get("name").and_then(Value::as_str).ok_or("a tool_use block without a name")?;
                            calls.push(json!({"id": b.get("id"), "type": "function",
                                "function": {"name": name, "arguments": b.get("input").cloned().unwrap_or(json!({}))}}))
                        }
                        // Anthropic's encrypted reasoning: nothing this model can read
                        "redacted_thinking" => {}
                        t => return Err(format!("content block type '{t}' is not supported in an assistant message")),
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
        let mut fns = Vec::new();
        for t in tools {
            // a server tool (web search, code execution, ...) has a type and no schema
            let (Some(name), Some(schema)) = (t.get("name").and_then(Value::as_str), t.get("input_schema")) else {
                let kind = t.get("type").and_then(Value::as_str).unwrap_or("?");
                return Err(format!("tools: only client tools (a name and an input_schema) are supported, not type '{kind}'"));
            };
            fns.push(json!({"type": "function", "function": {"name": name, "description": t.get("description").cloned().unwrap_or(json!("")),
                            "parameters": schema}}));
        }
        if !fns.is_empty() && !choice_none {
            r.tools = Some(Value::Array(fns));
        }
    }
    // a field of the wrong type or out of range is an error; absent or null takes the default
    r.max_tokens = chat::field_u32(req, "max_tokens")?;
    r.temperature = chat::field_f32(req, "temperature")?;
    r.top_p = chat::field_f32(req, "top_p")?;
    r.top_k = chat::field_u32(req, "top_k")?;
    r.stop = chat::field_stops(req, "stop_sequences")?;
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
        Err(e) => return error(400, "invalid_request_error", &e),
    };
    match tokio::task::spawn_blocking(move || chat::prompt_of(&st, &r)).await {
        Ok(Ok((_, toks))) => Json(json!({"input_tokens": toks.len()})).into_response(),
        Ok(Err(e)) => start_error(&e),
        Err(e) => error(500, "api_error", &format!("preparing the prompt failed: {e}")),
    }
}

pub async fn messages(st: Arc<AppState>, req: Value) -> Response {
    let (r, show_thinking) = match to_chat(&req) {
        Ok(x) => x,
        Err(e) => return error(400, "invalid_request_error", &e),
    };
    let model = req.get("model").and_then(Value::as_str).unwrap_or(&st.model_name).to_string();
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let (mut rx, n_prompt) = match chat::start(st, r).await {
        Ok(x) => x,
        Err(e) => return start_error(&e),
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
            ChatEvent::Error(e) => return error(500, "api_error", &e),
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
                    "usage": usage(prompt_tokens, reused, completion_tokens),
                    "timings": timings,
                }))
                .into_response();
            }
        }
    }
    error(500, "api_error", "generation ended without a result")
}

pub(crate) fn stream_messages(
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
        // the reuse is not known yet: the whole prompt, corrected by message_delta
        send!(ev("message_start", json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": n_prompt, "output_tokens": 0}}})));
        send!(ev("ping", json!({"type": "ping"})));
        // next_event also ends the task when the client leaves while nothing is sent (a prefill,
        // or reasoning that is not shown)
        while let Some(e) = chat::next_event(&mut rx, &tx).await {
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
                ChatEvent::Done { finish, stop_sequence, prompt_tokens, completion_tokens, reused, .. } => {
                    if index < 0 {
                        open_block!("text", json!({"type": "text", "text": ""}));
                    }
                    close!();
                    // the request's usage, as a non-streaming response reports it (Anthropic's
                    // message_delta usage is cumulative and carries the input counts too)
                    send!(ev("message_delta", json!({"type": "message_delta",
                        "delta": {"stop_reason": stop_reason(finish), "stop_sequence": stop_sequence},
                        "usage": usage(prompt_tokens, reused, completion_tokens)})));
                    send!(ev("message_stop", json!({"type": "message_stop"})));
                    return;
                }
            }
        }
    });
    Sse::new(ReceiverStream::new(out).map(Ok)).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_and_mistyped_numbers_are_rejected() {
        let base = |k: &str, v: Value| json!({"messages": [{"role": "user", "content": "x"}], k: v});
        // 2^32 would have become 0 and 2^32 + 1 a top_k of 1 (under the 64 limit)
        for (k, v) in [("max_tokens", json!(4294967296u64)), ("top_k", json!(4294967297u64)), ("temperature", json!("1")),
                       ("top_p", json!([])), ("stop_sequences", json!([1]))] {
            let e = to_chat(&base(k, v)).err().unwrap_or_else(|| panic!("{k} accepted"));
            assert!(e.starts_with(k), "{k}: {e}");
        }
        let (r, _) = to_chat(&base("max_tokens", json!(100))).unwrap();
        assert_eq!((r.max_tokens, r.top_k), (Some(100), None));
        let (r, _) = to_chat(&base("stop_sequences", json!(["a", "b"]))).unwrap();
        assert_eq!(r.stop, ["a", "b"]);
    }

    async fn body_of(resp: Response) -> String {
        let b = tokio::time::timeout(std::time::Duration::from_secs(5), axum::body::to_bytes(resp.into_body(), 1 << 20))
            .await
            .expect("the response ends")
            .unwrap();
        String::from_utf8(b.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_reports_the_same_usage_as_a_response() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(ChatEvent::Content("hi".into())).await.unwrap();
        tx.send(ChatEvent::Done { finish: Finish::Stop, stop_sequence: None, prompt_tokens: 100, completion_tokens: 3, reused: 60,
                                  timings: json!({}) }).await.unwrap();
        drop(tx);
        let text = body_of(stream_messages(rx, "m1".into(), "x".into(), 100, false).into_response()).await;
        let data: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).map(|d| serde_json::from_str(d).unwrap()).collect();
        let start = data.iter().find(|d| d["type"] == "message_start").unwrap();
        assert_eq!(start["message"]["usage"]["input_tokens"], 100);   // the reuse is not known yet
        let delta = data.iter().find(|d| d["type"] == "message_delta").unwrap();
        // what the non-streaming response reports
        assert_eq!(delta["usage"], usage(100, 60, 3));
        assert_eq!(usage(100, 60, 3), json!({"input_tokens": 40, "cache_read_input_tokens": 60, "output_tokens": 3}));
    }

    #[test]
    fn unsupported_blocks_and_server_tools_are_rejected() {
        let user = |content: Value| json!({"messages": [{"role": "user", "content": content}]});
        for (req, msg) in [
            (user(json!([{"type": "image", "source": {}}])), "image and document"),
            (user(json!([{"type": "search_result", "content": []}])), "'search_result'"),
            (user(json!([{"type": "tool_result", "tool_use_id": "1", "content": [{"type": "image", "source": {}}]}])), "image and document"),
            (json!({"messages": [{"role": "assistant", "content": [{"type": "server_tool_use", "id": "1"}]}]}), "'server_tool_use'"),
            (json!({"messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "1", "input": {}}]}]}), "without a name"),
            (json!({"messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}), "'web_search_20250305'"),
        ] {
            let e = to_chat(&req).err().unwrap_or_else(|| panic!("accepted: {req}"));
            assert!(e.contains(msg), "{e}");
        }
        // what is accepted: text, tool results with text, redacted reasoning, client tools
        let ok = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                                                                   {"type": "tool_result", "tool_use_id": "1", "content": [{"type": "text", "text": "r"}]}]},
                                     {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}, {"type": "text", "text": "b"}]}],
                        "tools": [{"name": "f", "input_schema": {"type": "object"}}], "metadata": {"user_id": "u"}});
        let (r, _) = to_chat(&ok).unwrap();
        assert_eq!(r.tools.unwrap()[0]["function"]["name"], "f");
    }
}
