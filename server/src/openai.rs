// SPDX-License-Identifier: Apache-2.0
//! OpenAI-compatible endpoints: /v1/chat/completions (with reasoning_content, tool calls and
//! streaming) and /v1/completions (raw text).
//!
//! Parameters this server does not implement are a 400 when set to a value that would change the
//! output (`unsupported`), never silently ignored. An error during a stream is an `error` object
//! (`{"error": {"message", "type"}}`) followed by [DONE], on both endpoints.

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
        400 => api_error(400, "invalid_request_error", &e.to_string()),
        st => api_error(st, "server_error", &e.to_string()),
    }
}

/// An error in a stream: OpenAI's error object, as its own event (the stream then ends with
/// [DONE]).
fn stream_error(msg: &str) -> Event {
    Event::default().data(json!({"error": {"message": msg, "type": "server_error"}}).to_string())
}

/// Whether a JSON value is set to something: not null, false, 0, "" or empty.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Parameters the server does not implement, rejected when set to a value that would change the
/// output: several choices, log probabilities, constrained decoding, penalties and biases, and
/// for /v1/completions echo, best_of and suffix. Absent, null and the neutral values (n 1,
/// logprobs false, penalties 0, repetition penalties 1, response_format text) are accepted.
/// Fields that do not change the output (`model`, `user`, `metadata`, `store`, `service_tier`,
/// ...) are ignored, and `parallel_tool_calls`, like `tool_choice`, is accepted but not enforced
/// (docs/engine.md, "Known issues").
fn unsupported(req: &Value, completions: bool) -> Result<(), String> {
    let get = |k: &str| req.get(k).filter(|v| !v.is_null());
    if let Some(n) = get("n") {
        match n.as_u64() {
            Some(1) => {}
            Some(k) if k > 1 => return Err("n > 1 is not supported (one choice per request)".into()),
            _ => return Err(format!("n must be 1, not {n}")),
        }
    }
    for k in ["logprobs", "top_logprobs"] {
        if get(k).is_some_and(truthy) {
            return Err(format!("{k} is not supported"));
        }
    }
    if let Some(f) = get("response_format") {
        let kind = f.get("type").and_then(Value::as_str);
        if kind != Some("text") {
            return Err(format!("response_format {} is not supported (no constrained decoding); only {{\"type\": \"text\"}}",
                               kind.map_or_else(|| f.to_string(), |k| format!("'{k}'"))));
        }
    }
    for k in ["grammar", "json_schema", "functions", "logit_bias"] {
        if get(k).is_some_and(truthy) {
            return Err(format!("{k} is not supported"));
        }
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        if let Some(v) = get(k) {
            if v.as_f64() != Some(0.0) {
                return Err(format!("{k} is not supported (only 0)"));
            }
        }
    }
    for k in ["repetition_penalty", "repeat_penalty"] {
        if let Some(v) = get(k) {
            if v.as_f64() != Some(1.0) {
                return Err(format!("{k} is not supported (only 1)"));
            }
        }
    }
    if completions {
        if get("echo").is_some_and(truthy) {
            return Err("echo is not supported".into());
        }
        if get("suffix").is_some_and(truthy) {
            return Err("suffix is not supported".into());
        }
        if let Some(b) = get("best_of") {
            if b.as_u64() != Some(1) {
                return Err("best_of is not supported (only 1)".into());
            }
        }
    }
    Ok(())
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
        if !m.get("role").is_some_and(Value::is_string) {
            return Err(format!("messages[{i}]: role must be a string"));
        }
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
                if !c.pointer("/function/name").is_some_and(Value::is_string) {
                    return Err(format!("messages[{i}]: a tool call without a function name"));
                }
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

/// The settings both endpoints share; a field of the wrong type or out of range is an error, as
/// is a parameter the server does not implement (`unsupported`).
fn common(req: &Value, r: &mut ChatRequest, completions: bool) -> Result<(), String> {
    unsupported(req, completions)?;
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
    if let Err(e) = common(&req, &mut r, false) {
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
                ChatEvent::Error(e) => {
                    if tx.send(stream_error(&e)).await.is_ok() {
                        let _ = tx.send(Event::default().data("[DONE]")).await;
                    }
                    return;
                }
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
    if let Err(e) = common(&req, &mut r, true) {
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
        let include_usage = req.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
        return stream_completion(rx, id, created, model, include_usage).into_response();
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
    include_usage: bool,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let chunk = |text: String, finish: Value| {
            Event::default().data(
                json!({"id": id, "object": "text_completion", "created": created, "model": model,
                       "choices": [{"index": 0, "text": text, "finish_reason": finish}]})
                .to_string(),
            )
        };
        while let Some(ev) = chat::next_event(&mut rx, &tx).await {
            let e = match ev {
                ChatEvent::Content(s) => chunk(s, Value::Null),
                ChatEvent::Done { finish, prompt_tokens, completion_tokens, timings, .. } => {
                    let mut last = json!({"id": id, "object": "text_completion", "created": created, "model": model,
                                          "choices": [{"index": 0, "text": "", "finish_reason": finish_str(finish)}]});
                    last["timings"] = timings;   // as the chat stream, on the final chunk
                    let _ = tx.send(Event::default().data(last.to_string())).await;
                    if include_usage {
                        let _ = tx
                            .send(Event::default().data(
                                json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": [],
                                       "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                                 "total_tokens": prompt_tokens + completion_tokens}})
                                .to_string(),
                            ))
                            .await;
                    }
                    let _ = tx.send(Event::default().data("[DONE]")).await;
                    return;
                }
                ChatEvent::Error(e) => {
                    if tx.send(stream_error(&e)).await.is_ok() {
                        let _ = tx.send(Event::default().data("[DONE]")).await;
                    }
                    return;
                }
                _ => continue,
            };
            if tx.send(e).await.is_err() {
                return;   // the client left; dropping rx makes the generation task stop the engine
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
            let e = common(&json!({k: v}), &mut r, false).unwrap_err();
            assert!(e.starts_with(k), "{k}: {e}");
        }
        // absent and null take the defaults; max_completion_tokens wins over max_tokens
        common(&json!({"max_tokens": null, "top_k": null}), &mut r, false).unwrap();
        assert_eq!((r.max_tokens, r.top_k), (None, None));
        common(&json!({"max_completion_tokens": null, "max_tokens": 9, "top_k": 64, "seed": 5, "stop": ["a", "b"]}), &mut r, false).unwrap();
        assert_eq!((r.max_tokens, r.top_k, r.seed, r.stop.len()), (Some(9), Some(64), Some(5), 2));
        common(&json!({"max_completion_tokens": 3, "max_tokens": 9}), &mut r, false).unwrap();
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

    #[test]
    fn unimplemented_parameters_are_rejected_not_ignored() {
        for (req, completions, msg) in [
            (json!({"n": 2}), false, "n > 1 is not supported"),
            (json!({"n": 0}), false, "n must be 1"),
            (json!({"n": "2"}), true, "n must be 1"),
            (json!({"logprobs": true}), false, "logprobs is not supported"),
            (json!({"logprobs": 3}), true, "logprobs is not supported"),
            (json!({"top_logprobs": 2}), false, "top_logprobs is not supported"),
            (json!({"response_format": {"type": "json_object"}}), false, "response_format 'json_object' is not supported"),
            (json!({"response_format": {"type": "json_schema", "json_schema": {}}}), false, "'json_schema'"),
            (json!({"response_format": "json"}), false, "response_format \"json\" is not supported"),
            (json!({"presence_penalty": 1.5}), false, "presence_penalty is not supported"),
            (json!({"frequency_penalty": -0.5}), true, "frequency_penalty is not supported"),
            (json!({"repetition_penalty": 1.1}), false, "repetition_penalty is not supported"),
            (json!({"repeat_penalty": 1.1}), true, "repeat_penalty"),
            (json!({"logit_bias": {"1": 5}}), false, "logit_bias is not supported"),
            (json!({"json_schema": {"type": "object"}}), false, "json_schema is not supported"),
            (json!({"grammar": "root ::= \"a\""}), true, "grammar is not supported"),
            (json!({"functions": [{"name": "f"}]}), false, "functions is not supported"),
            (json!({"echo": true}), true, "echo is not supported"),
            (json!({"suffix": "x"}), true, "suffix is not supported"),
            (json!({"best_of": 3}), true, "best_of is not supported"),
        ] {
            let e = common(&req, &mut empty_request(), completions).err().unwrap_or_else(|| panic!("accepted: {req}"));
            assert!(e.contains(msg), "{req}: {e}");
        }
        // the neutral values, and fields that do not change the output, are accepted
        let ok = json!({"n": 1, "logprobs": false, "top_logprobs": null, "response_format": {"type": "text"}, "presence_penalty": 0,
                        "frequency_penalty": 0.0, "repetition_penalty": 1, "logit_bias": {}, "echo": false, "suffix": "", "best_of": 1,
                        "user": "u", "metadata": {"a": "b"}, "store": false, "parallel_tool_calls": false, "model": "any",
                        "stream_options": {"include_usage": true}});
        common(&ok, &mut empty_request(), true).unwrap();
        // completions-only checks do not apply to chat
        common(&json!({"echo": true}), &mut empty_request(), false).unwrap();
        assert!(normalize_messages(&json!([{"content": "x"}])).unwrap_err().contains("role must be a string"));
        let e = normalize_messages(&json!([{"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]}])).unwrap_err();
        assert!(e.contains("without a function name"), "{e}");
    }

    async fn frames(resp: Response) -> Vec<String> {
        let body = tokio::time::timeout(std::time::Duration::from_secs(5), axum::body::to_bytes(resp.into_body(), 1 << 20))
            .await
            .expect("the stream ends")
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        text.lines().filter_map(|l| l.strip_prefix("data: ")).map(String::from).collect()
    }

    /// An engine error mid-stream: both endpoints send the error object and then [DONE], and no
    /// chunk carries a finish_reason that is not one of OpenAI's.
    #[tokio::test]
    async fn stream_errors_are_error_objects_on_both_endpoints() {
        for api in ["chat", "completions"] {
            let (tx, rx) = mpsc::channel(8);
            tx.send(ChatEvent::Content("partial".into())).await.unwrap();
            tx.send(ChatEvent::Error("the engine failed".into())).await.unwrap();
            drop(tx);
            let resp = match api {
                "chat" => stream_chat(rx, "id1".into(), 7, "m".into(), false).into_response(),
                _ => stream_completion(rx, "id1".into(), 7, "m".into(), false).into_response(),
            };
            let data = frames(resp).await;
            assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "{api}: {data:?}");
            let v: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
            let err = v.last().unwrap();
            assert_eq!(err["error"], json!({"message": "the engine failed", "type": "server_error"}), "{api}");
            for c in &v {
                for ch in c.get("choices").and_then(Value::as_array).into_iter().flatten() {
                    let f = &ch["finish_reason"];
                    assert!(f.is_null() || ["stop", "length", "tool_calls"].contains(&f.as_str().unwrap()), "{api}: {c}");
                }
            }
        }
    }

    #[tokio::test]
    async fn completion_stream_usage_and_timings() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(ChatEvent::Content("Paris".into())).await.unwrap();
        tx.send(ChatEvent::Done { finish: Finish::Length, stop_sequence: None, prompt_tokens: 5, completion_tokens: 1, reused: 0,
                                  timings: json!({"prompt_n": 5}) }).await.unwrap();
        drop(tx);
        let data = frames(stream_completion(rx, "c1".into(), 7, "m".into(), true).into_response()).await;
        assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
        let v: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
        assert_eq!(v.len(), 3, "{data:?}");
        assert_eq!(v[0]["choices"][0]["text"], "Paris");
        assert_eq!((v[1]["choices"][0]["finish_reason"].as_str(), v[1]["timings"]["prompt_n"].as_u64()), (Some("length"), Some(5)));
        assert_eq!(v[2]["usage"], json!({"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}));
    }
}
