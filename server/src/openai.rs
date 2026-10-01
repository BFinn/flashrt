// SPDX-License-Identifier: Apache-2.0
//! OpenAI-compatible endpoints: /v1/chat/completions (with reasoning_content, tool calls and
//! streaming) and /v1/completions (raw text).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::chat::{self, ChatEvent, ChatRequest, Finish};
use crate::{api_error, AppState};

fn start_error(e: &anyhow::Error) -> Response {
    match crate::start_error_status(e) {
        503 => api_error(503, "server_error", &e.to_string()),
        st => api_error(st, "invalid_request_error", &e.to_string()),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn finish_str(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::StopSequence | Finish::Cancelled => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

/// OpenAI messages to the template's form: text parts joined into a string (as this model's
/// template joins them, with nothing between; a template that expects strings works too),
/// tool-call arguments parsed into objects, reasoning under reasoning_content.
fn normalize_messages(msgs: &Value) -> Result<Value, String> {
    let arr = msgs.as_array().ok_or("messages must be an array")?;
    let mut out = Vec::new();
    for (i, m) in arr.iter().enumerate() {
        let mut m = m.as_object().ok_or("each message must be an object")?.clone();
        if let Some(Value::Array(parts)) = m.get("content") {
            let mut text = String::new();
            for p in parts {
                let ty = p.get("type").and_then(Value::as_str).unwrap_or("text");
                if ty != "text" {
                    return Err(format!("content part type '{ty}' is not supported"));
                }
                text.push_str(p.get("text").and_then(Value::as_str).ok_or("a text part without text")?);
            }
            m.insert("content".into(), Value::String(text));
        }
        if m.get("reasoning_content").is_none() {
            if let Some(r) = m.remove("reasoning") {
                m.insert("reasoning_content".into(), r);
            }
        }
        if let Some(Value::Array(calls)) = m.get_mut("tool_calls") {
            for c in calls.iter_mut() {
                if let Some(f) = c.get_mut("function") {
                    if let Some(Value::String(a)) = f.get("arguments") {
                        // the template renders the arguments' keys: a string that is not a JSON
                        // object would render as something else than what the model produced
                        let parsed = if a.trim().is_empty() { Some(json!({})) } else { serde_json::from_str::<Value>(a).ok() };
                        match parsed {
                            Some(v @ Value::Object(_)) => f["arguments"] = v,
                            _ => return Err(format!("messages[{i}]: tool call arguments are not a JSON object: {a}")),
                        }
                    }
                }
            }
        }
        out.push(Value::Object(m));
    }
    Ok(Value::Array(out))
}

/// The settings both endpoints share; a field of the wrong type or out of range is an error.
fn common(req: &Value, r: &mut ChatRequest) -> Result<(), String> {
    r.max_tokens = match chat::field_u32(req, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => chat::field_u32(req, "max_tokens")?,
    };
    r.temperature = chat::field_f32(req, "temperature")?;
    r.top_p = chat::field_f32(req, "top_p")?;
    r.top_k = chat::field_u32(req, "top_k")?;
    r.min_p = chat::field_f32(req, "min_p")?;
    r.seed = chat::field_seed(req, "seed")?;
    r.stop = chat::field_stops(req, "stop")?;
    Ok(())
}

pub fn empty_request() -> ChatRequest {
    ChatRequest {
        messages: Value::Array(vec![]),
        tools: None,
        template_vars: Map::new(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        seed: None,
        stop: vec![],
        raw_prompt: None,
    }
}

pub async fn chat_completions(st: Arc<AppState>, req: Value) -> Response {
    let messages = match normalize_messages(req.get("messages").unwrap_or(&Value::Null)) {
        Ok(m) => m,
        Err(e) => return api_error(400, "invalid_request_error", &e),
    };
    let mut r = empty_request();
    r.messages = messages;
    if let Err(e) = common(&req, &mut r) {
        return api_error(400, "invalid_request_error", &e);
    }
    let tool_choice_none = req.get("tool_choice").and_then(Value::as_str) == Some("none");
    if let Some(t) = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        if !tool_choice_none {
            r.tools = Some(t.clone());
        }
    }
    if let Some(Value::Object(kw)) = req.get("chat_template_kwargs") {
        for (k, v) in kw {
            r.template_vars.insert(k.clone(), v.clone());
        }
    }
    match req.get("reasoning_effort").and_then(Value::as_str) {
        Some("none") => {
            r.template_vars.insert("enable_thinking".into(), Value::Bool(false));
        }
        Some("minimal") => {
            r.template_vars.insert("reasoning_effort".into(), json!("low"));
        }
        Some(e) => {
            r.template_vars.insert("reasoning_effort".into(), json!(e));
        }
        None => {}
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = req.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let model = st.model_name.clone();
    let rx = match chat::start(st, r).await {
        Ok((rx, _)) => rx,
        Err(e) => return start_error(&e),
    };
    let id = chat::new_id("chatcmpl-");
    let created = now();
    if stream {
        return stream_chat(rx, id, created, model, include_usage).into_response();
    }
    let mut rx = rx;
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Reasoning(s) => reasoning.push_str(&s),
            ChatEvent::Content(s) => content.push_str(&s),
            ChatEvent::ToolCall { id, name, arguments } => calls.push(json!({
                "id": id, "type": "function", "function": {"name": name, "arguments": arguments.to_string()}
            })),
            ChatEvent::Error(e) => return api_error(500, "server_error", &e),
            ChatEvent::Done { finish, prompt_tokens, completion_tokens, reused, timings, .. } => {
                let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    msg["tool_calls"] = json!(calls);
                }
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish_str(finish)}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": reused}},
                    "timings": timings,
                }))
                .into_response();
            }
        }
    }
    api_error(500, "server_error", "generation ended without a result")
}

pub(crate) fn stream_chat(
    mut rx: mpsc::Receiver<ChatEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let chunk = |delta: Value, finish: Value| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        };
        let mut n_calls = 0;
        let send = |v: Value| Event::default().data(v.to_string());
        if tx.send(send(chunk(json!({"role": "assistant", "content": ""}), Value::Null))).await.is_err() {
            return;
        }
        while let Some(ev) = chat::next_event(&mut rx, &tx).await {
            let e = match ev {
                ChatEvent::Reasoning(s) => send(chunk(json!({"reasoning_content": s}), Value::Null)),
                ChatEvent::Content(s) => send(chunk(json!({"content": s}), Value::Null)),
                ChatEvent::ToolCall { id: cid, name, arguments } => {
                    n_calls += 1;
                    send(chunk(
                        json!({"tool_calls": [{"index": n_calls - 1, "id": cid, "type": "function",
                                               "function": {"name": name, "arguments": arguments.to_string()}}]}),
                        Value::Null,
                    ))
                }
                ChatEvent::Error(e) => send(json!({"error": {"message": e, "type": "server_error"}})),
                ChatEvent::Done { finish, prompt_tokens, completion_tokens, reused, timings, .. } => {
                    let mut last = chunk(json!({}), json!(finish_str(finish)));
                    last["timings"] = timings;   // as llama.cpp's server, on the final chunk
                    let _ = tx.send(send(last)).await;
                    if include_usage {
                        let _ = tx
                            .send(send(json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens, "prompt_tokens_details": {"cached_tokens": reused}}})))
                            .await;
                    }
                    let _ = tx.send(Event::default().data("[DONE]")).await;
                    return;
                }
            };
            if tx.send(e).await.is_err() {
                return;   // the client left; dropping rx makes the generation task stop the engine
            }
        }
    });
    Sse::new(ReceiverStream::new(out).map(Ok)).keep_alive(KeepAlive::default())
}

pub async fn completions(st: Arc<AppState>, req: Value) -> Response {
    let prompt = match req.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap().to_string(),
        _ => return api_error(400, "invalid_request_error", "prompt must be a string"),
    };
    let mut r = empty_request();
    if let Err(e) = common(&req, &mut r) {
        return api_error(400, "invalid_request_error", &e);
    }
    r.raw_prompt = Some(prompt);
    let model = st.model_name.clone();
    let mut rx = match chat::start(st, r).await {
        Ok((rx, _)) => rx,
        Err(e) => return start_error(&e),
    };
    let id = chat::new_id("cmpl-");
    let created = now();
    if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return stream_completion(rx, id, created, model).into_response();
    }
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Content(s) => text.push_str(&s),
            ChatEvent::Error(e) => return api_error(500, "server_error", &e),
            ChatEvent::Done { finish, prompt_tokens, completion_tokens, timings, .. } => {
                return Json(json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish_str(finish)}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                    "timings": timings,
                }))
                .into_response();
            }
            _ => {}
        }
    }
    api_error(500, "server_error", "generation ended without a result")
}

pub(crate) fn stream_completion(
    mut rx: mpsc::Receiver<ChatEvent>,
    id: String,
    created: u64,
    model: String,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        while let Some(ev) = chat::next_event(&mut rx, &tx).await {
            let (text, finish, last) = match ev {
                ChatEvent::Content(s) => (s, Value::Null, false),
                ChatEvent::Done { finish, .. } => (String::new(), json!(finish_str(finish)), true),
                ChatEvent::Error(e) => (String::new(), json!(format!("error: {e}")), true),
                _ => continue,
            };
            let v = json!({"id": id, "object": "text_completion", "created": created, "model": model,
                           "choices": [{"index": 0, "text": text, "finish_reason": finish}]});
            if tx.send(Event::default().data(v.to_string())).await.is_err() {
                return;   // the client left; dropping rx makes the generation task stop the engine
            }
            if last {
                let _ = tx.send(Event::default().data("[DONE]")).await;
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(out).map(Ok)).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_parts_join_and_tool_arguments_parse() {
        let m = normalize_messages(&json!([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": null, "reasoning": "r",
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
                            {"id": "2", "type": "function", "function": {"name": "g", "arguments": " "}}]},
        ]))
        .unwrap();
        assert_eq!(m[0]["content"], "ab");
        assert_eq!(m[1]["reasoning_content"], "r");
        assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], json!({"x": 1}));
        assert_eq!(m[1]["tool_calls"][1]["function"]["arguments"], json!({}));
    }

    /// The SSE frames of a streamed chat: the role first, one delta per event, the finish chunk
    /// with the engine's timings, the usage chunk when asked for, then [DONE].
    #[tokio::test]
    async fn stream_frames() {
        let (tx, rx) = mpsc::channel(8);
        for ev in [
            ChatEvent::Reasoning("think".into()),
            ChatEvent::Content("hi".into()),
            ChatEvent::ToolCall { id: "c1".into(), name: "f".into(), arguments: json!({"x": 1}) },
            ChatEvent::Done {
                finish: Finish::ToolCalls,
                stop_sequence: None,
                prompt_tokens: 10,
                completion_tokens: 4,
                reused: 6,
                timings: json!({"cache_n": 6, "prompt_n": 4}),
            },
        ] {
            tx.send(ev).await.unwrap();
        }
        drop(tx);
        let resp = stream_chat(rx, "id1".into(), 7, "m".into(), true).into_response();
        let body = tokio::time::timeout(std::time::Duration::from_secs(5), axum::body::to_bytes(resp.into_body(), 1 << 20))
            .await
            .expect("the stream ends after Done")
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let data: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
        assert_eq!(data.last(), Some(&"[DONE]"), "{text}");
        let v: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
        assert_eq!(v.len(), 6, "{text}");
        assert_eq!(v[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v[1]["choices"][0]["delta"]["reasoning_content"], "think");
        assert_eq!(v[2]["choices"][0]["delta"]["content"], "hi");
        let call = &v[3]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!((call["index"].as_u64(), call["function"]["arguments"].as_str()), (Some(0), Some("{\"x\":1}")));
        assert_eq!(v[4]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(v[4]["timings"]["cache_n"], 6);
        assert_eq!(v[5]["usage"]["prompt_tokens_details"]["cached_tokens"], 6);
        assert!(v.iter().all(|c| c["id"] == "id1" && c["object"] == "chat.completion.chunk"));
    }

    #[test]
    fn out_of_range_and_mistyped_numbers_are_rejected() {
        let mut r = empty_request();
        // 2^32 would have become 0 and 2^32 + 1 a top_k of 1 (under the 64 limit)
        for (k, v) in [("max_tokens", json!(4294967296u64)), ("top_k", json!(4294967297u64)), ("max_completion_tokens", json!(-1)),
                       ("temperature", json!("0.7")), ("top_p", json!(true)), ("seed", json!(1.5)), ("stop", json!(7))] {
            let e = common(&json!({k: v}), &mut r).unwrap_err();
            assert!(e.starts_with(k), "{k}: {e}");
        }
        // absent and null take the defaults; max_completion_tokens wins over max_tokens
        common(&json!({"max_tokens": null, "top_k": null}), &mut r).unwrap();
        assert_eq!((r.max_tokens, r.top_k), (None, None));
        common(&json!({"max_completion_tokens": null, "max_tokens": 9, "top_k": 64, "seed": 5, "stop": ["a", "b"]}), &mut r).unwrap();
        assert_eq!((r.max_tokens, r.top_k, r.seed, r.stop.len()), (Some(9), Some(64), Some(5), 2));
        common(&json!({"max_completion_tokens": 3, "max_tokens": 9}), &mut r).unwrap();
        assert_eq!(r.max_tokens, Some(3));
    }

    #[test]
    fn rejects_what_would_render_wrong() {
        let call = |a: &str| json!([{"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": a}}]}]);
        for bad in ["{not json", "[1, 2]", "3", "\"s\""] {
            let e = normalize_messages(&call(bad)).unwrap_err();
            assert!(e.contains("messages[0]") && e.contains("not a JSON object"), "{e}");
        }
        assert!(normalize_messages(&json!([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])).is_err());
        assert!(normalize_messages(&json!([{"role": "user", "content": [{"type": "text"}]}])).is_err());
    }
}
