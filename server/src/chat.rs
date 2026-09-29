// SPDX-License-Identifier: Apache-2.0
//! One chat generation, shared by the OpenAI and Anthropic endpoints: render the template,
//! tokenize, queue the request on the engine, and turn its token stream into events: reasoning
//! text (before </think>), answer text, and tool calls (the template's
//! <tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call> format).
//! The structure is read from the special tokens' ids, never from generated text.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use crate::engine::GenerateParams;
use crate::tokenizer::Decoder;
use crate::AppState;

pub struct ChatRequest {
    pub messages: Value,                 // OpenAI-style, tool-call arguments as objects
    pub tools: Option<Value>,            // [{type: function, function: {name, description, parameters}}]
    pub template_vars: Map<String, Value>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    /// A raw prompt (text completion): no template, and the output is plain text.
    pub raw_prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Finish {
    Stop,
    StopSequence,
    Length,
    ToolCalls,
    Cancelled,
}

#[derive(Debug)]
pub enum ChatEvent {
    Reasoning(String),
    Content(String),
    ToolCall { id: String, name: String, arguments: Value },
    /// `timings`: the engine's figures for the request, see `timings_of`.
    Done { finish: Finish, stop_sequence: Option<String>, prompt_tokens: u32, completion_tokens: u32, reused: u32, timings: Value },
    Error(String),
}

/// The engine's figures for one request, in the shape of llama.cpp's `timings` (prompt_n counts
/// the prefilled tokens, cache_n the reused ones), plus flashrt's decode expert-cache counts.
pub fn timings_of(done: &Value) -> Value {
    let n = |k: &str| done.get(k).and_then(Value::as_u64).unwrap_or(0);
    let f = |k: &str| done.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    let (prompt, reused, generated) = (n("prompt_tokens"), n("reused"), n("generated"));
    let (prompt_ms, decode_ms) = (f("prompt_ms"), f("decode_ms"));
    let per_s = |count: u64, ms: f64| if ms > 0.0 { count as f64 * 1000.0 / ms } else { 0.0 };
    let prefilled = prompt.saturating_sub(reused);
    let mut t = json!({
        "cache_n": reused,
        "prompt_n": prefilled, "prompt_ms": prompt_ms, "prompt_per_second": per_s(prefilled, prompt_ms),
        "predicted_n": generated, "predicted_ms": decode_ms, "predicted_per_second": per_s(generated, decode_ms),
    });
    if let Some(d) = done.get("drafts") {
        t["draft_n"] = d.get("proposed").cloned().unwrap_or(json!(0));
        t["draft_n_accepted"] = d.get("accepted").cloned().unwrap_or(json!(0));
    }
    if let Some(c) = done.get("cache") {
        t["expert_cache"] = c.clone();
    }
    t
}

pub fn random_u64() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
    h.finish()
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{:016x}{:08x}", random_u64(), random_u64() as u32)
}

/// The parameter types of each tool, from its JSON schema: name -> (parameter -> type).
fn tool_param_types(tools: Option<&Value>) -> std::collections::HashMap<String, Map<String, Value>> {
    let mut out = std::collections::HashMap::new();
    for t in tools.and_then(Value::as_array).into_iter().flatten() {
        let f = t.get("function").unwrap_or(t);
        let Some(name) = f.get("name").and_then(Value::as_str) else { continue };
        let props = f.pointer("/parameters/properties").and_then(Value::as_object).cloned().unwrap_or_default();
        let mut types = Map::new();
        for (p, schema) in props {
            types.insert(p, schema.get("type").cloned().unwrap_or(Value::Null));
        }
        out.insert(name.to_string(), types);
    }
    out
}

/// Parses one tool call's body ("<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>").
/// Values of parameters whose schema type is not "string" are parsed as JSON when they parse.
fn parse_tool_call(body: &str, types: &std::collections::HashMap<String, Map<String, Value>>) -> Option<(String, Value)> {
    let start = body.find("<function=")? + "<function=".len();
    let name_end = start + body[start..].find('>')?;
    let name = body[start..name_end].trim().to_string();
    let schema = types.get(&name);
    let mut args = Map::new();
    let mut rest = &body[name_end + 1..];
    while let Some(p) = rest.find("<parameter=") {
        let ps = p + "<parameter=".len();
        let Some(pe) = rest[ps..].find('>') else { break };
        let pname = rest[ps..ps + pe].trim().to_string();
        let vstart = ps + pe + 1;
        // the value ends at the first closing tag; the cursor moves past the tag actually found
        let (vend, tag_len) = ["</parameter>", "</function>"]
            .iter()
            .filter_map(|tag| rest[vstart..].find(tag).map(|i| (vstart + i, tag.len())))
            .min()
            .unwrap_or((rest.len(), 0));
        let mut v = &rest[vstart..vend];
        v = v.strip_prefix('\n').unwrap_or(v);
        v = v.strip_suffix('\n').unwrap_or(v);
        let ty = schema.and_then(|s| s.get(&pname)).and_then(Value::as_str);
        let value = match ty {
            Some("string") | None => Value::String(v.to_string()),
            _ => serde_json::from_str::<Value>(v.trim()).unwrap_or_else(|_| Value::String(v.to_string())),
        };
        args.insert(pname, value);
        rest = &rest[vend + tag_len..];
    }
    Some((name, Value::Object(args)))
}

#[derive(PartialEq)]
enum Mode {
    Reasoning,
    Content,
    Tool,
}

/// Streams text with trailing whitespace held back (and dropped at a section's end), leading
/// whitespace dropped, and, for the answer, stop strings cut off.
struct Section {
    text: String,
    sent: usize,
    started: bool,
    trim: bool,   // drop leading and hold trailing whitespace (not for raw completions)
}

impl Section {
    fn new() -> Self {
        Self { text: String::new(), sent: 0, started: false, trim: true }
    }
    fn raw() -> Self {
        Self { text: String::new(), sent: 0, started: true, trim: false }
    }
    fn push(&mut self, s: &str) {
        if self.started {
            self.text.push_str(s);
        } else {
            let t = s.trim_start();
            if !t.is_empty() {
                self.started = true;
                self.text.push_str(t);
            }
        }
    }
    /// The next piece to send: up to the trailing whitespace and any partial stop string.
    fn take(&mut self, stops: &[String]) -> String {
        let mut end = if self.trim { self.text.trim_end().len() } else { self.text.len() };
        for s in stops {
            for (k, _) in s.char_indices().skip(1) {
                if self.text.ends_with(&s[..k]) {
                    end = end.min(self.text.len() - k);
                }
            }
        }
        if end <= self.sent {
            return String::new();
        }
        let out = self.text[self.sent..end].to_string();
        self.sent = end;
        out
    }
    /// A stop string in the unsent text: cut the text there, return the stop and the piece before.
    fn stop_hit(&mut self, stops: &[String]) -> Option<(String, String)> {
        let from = self.sent;
        let mut best: Option<(usize, &String)> = None;
        for s in stops {
            if s.is_empty() {
                continue;
            }
            // search a little before `sent`, in case a stop straddles the boundary (held back, so it cannot)
            if let Some(i) = self.text[from..].find(s.as_str()) {
                if best.is_none_or(|(b, _)| from + i < b) {
                    best = Some((from + i, s));
                }
            }
        }
        let (at, s) = best?;
        let piece = self.text[self.sent..at].to_string();
        self.text.truncate(at);
        self.sent = at;
        Some((s.clone(), piece))
    }
}

/// The prompt text and tokens of a request.
pub fn prompt_of(st: &AppState, req: &ChatRequest) -> Result<(String, Vec<u32>)> {
    let text = match &req.raw_prompt {
        Some(p) => p.clone(),
        None => st.template.render(&req.messages, req.tools.as_ref(), &req.template_vars)?,
    };
    let toks = st.tokenizer.encode(&text, true);
    Ok((text, toks))
}

/// A request's sampling settings, checked against what the engine accepts. `top_k` 0 means no
/// limit, which is the engine's most (64); more than 64 is an error, not a silent cap. Seeds keep
/// their low 53 bits (the engine protocol carries them as JSON numbers).
fn sampling_of(st: &AppState, req: &ChatRequest) -> Result<(f32, f32, u32, f32, u64)> {
    let temperature = req.temperature.unwrap_or(st.sampling.temperature);
    let top_p = req.top_p.unwrap_or(st.sampling.top_p);
    let min_p = req.min_p.unwrap_or(st.sampling.min_p);
    let top_k = match req.top_k.unwrap_or(st.sampling.top_k) {
        0 => 64,
        k if k <= 64 => k,
        k => bail!("top_k {k} is above 64, the most this engine samples from (0 means 64)"),
    };
    if !(temperature.is_finite() && temperature >= 0.0) {
        bail!("temperature must be >= 0");
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        bail!("top_p must be in (0, 1]");
    }
    if !(0.0..1.0).contains(&min_p) {
        bail!("min_p must be in [0, 1)");
    }
    let seed = req.seed.unwrap_or_else(random_u64) & ((1u64 << 53) - 1);
    Ok((temperature, top_p, top_k, min_p, seed))
}

/// Renders, tokenizes and queues a chat request; events arrive on the receiver. Also returns
/// the prompt's length in tokens. Errors are the request's fault (HTTP 400), except
/// `engine::EngineDown` (503).
pub async fn start(st: Arc<AppState>, req: ChatRequest) -> Result<(mpsc::Receiver<ChatEvent>, u32)> {
    let raw = req.raw_prompt.is_some();
    let (temperature, top_p, top_k, min_p, seed) = sampling_of(&st, &req)?;
    let (prompt_text, prompt) = prompt_of(&st, &req)?;
    let ctx = st.max_context;
    if prompt.len() as u64 + 16 >= ctx {
        bail!("the prompt has {} tokens; the context holds {}", prompt.len(), ctx);
    }
    let room = (ctx - prompt.len() as u64 - 16) as u32;
    let max_new = req.max_tokens.unwrap_or(st.default_max_tokens).min(room).max(1);
    let params = GenerateParams {
        prompt: prompt.clone(),
        max_new,
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        stop_ids: st.stop_ids.clone(),
    };
    let (id, mut rx) = st.engine.generate(&params).await?;
    let thinking = !raw && prompt_text.ends_with("<think>\n");
    let (tx, out) = mpsc::channel(512);
    let n_prompt = prompt.len() as u32;
    let types = tool_param_types(req.tools.as_ref());
    let stops = req.stop;
    tokio::spawn(async move {
        let tok = &st.tokenizer;
        let (think_start, think_end, tool_start, tool_end) = if raw {
            (u32::MAX, u32::MAX, u32::MAX, u32::MAX)
        } else {
            (st.ids.think, st.ids.think_end, st.ids.tool_call, st.ids.tool_call_end)
        };
        let mut mode = if thinking { Mode::Reasoning } else { Mode::Content };
        let mut dec = Decoder::default();
        let mut reasoning = Section::new();
        let mut content = if raw { Section::raw() } else { Section::new() };
        let mut tool_buf = String::new();
        let mut n_tools = 0;
        let mut stopped: Option<String> = None;
        let mut gone = false;   // the client went away
        macro_rules! send {
            ($ev:expr) => {
                if !gone && tx.send($ev).await.is_err() {
                    gone = true;
                    st.engine.stop(&id).await;
                }
            };
        }
        loop {
            let ev = tokio::select! {
                ev = rx.recv() => ev,
                // the client left (also while the request waits in the engine's queue): cancel it
                // now, then drain its events to the done
                _ = tx.closed(), if !gone => {
                    gone = true;
                    st.engine.stop(&id).await;
                    continue;
                }
            };
            let Some(ev) = ev else { break };
            match ev.get("ev").and_then(Value::as_str) {
                Some("token") if stopped.is_none() => {
                    let t = ev.get("tok").and_then(Value::as_u64).unwrap_or(0) as u32;
                    if t == think_end && mode == Mode::Reasoning {
                        let s = dec.finish();
                        reasoning.push(&s);
                        let r = reasoning.take(&[]);
                        if !r.is_empty() {
                            send!(ChatEvent::Reasoning(r));
                        }
                        mode = Mode::Content;
                        continue;
                    }
                    if t == tool_start && mode != Mode::Tool {
                        let s = dec.finish();
                        if mode == Mode::Content {
                            content.push(&s);
                            let c = content.take(&stops);
                            if !c.is_empty() {
                                send!(ChatEvent::Content(c));
                            }
                        }
                        mode = Mode::Tool;
                        tool_buf.clear();
                        continue;
                    }
                    if t == tool_end && mode == Mode::Tool {
                        tool_buf.push_str(&dec.finish());
                        if let Some((name, arguments)) = parse_tool_call(&tool_buf, &types) {
                            n_tools += 1;
                            send!(ChatEvent::ToolCall { id: new_id("call_"), name, arguments });
                        }
                        mode = Mode::Content;
                        continue;
                    }
                    // control tokens (<|im_start|> and the like) and structure tokens the state
                    // machine did not consume (a stray </think>) are not text
                    if tok.is_control(t) || [think_start, think_end, tool_start, tool_end].contains(&t) {
                        continue;
                    }
                    let text = dec.push(tok.token_bytes(t));
                    match mode {
                        Mode::Reasoning => {
                            reasoning.push(&text);
                            let r = reasoning.take(&[]);
                            if !r.is_empty() {
                                send!(ChatEvent::Reasoning(r));
                            }
                        }
                        Mode::Tool => tool_buf.push_str(&text),
                        Mode::Content => {
                            content.push(&text);
                            if let Some((s, piece)) = content.stop_hit(&stops) {
                                if !piece.is_empty() {
                                    send!(ChatEvent::Content(piece));
                                }
                                stopped = Some(s);
                                st.engine.stop(&id).await;
                                continue;
                            }
                            let c = content.take(&stops);
                            if !c.is_empty() {
                                send!(ChatEvent::Content(c));
                            }
                        }
                    }
                }
                Some("token") => {}
                Some("done") => {
                    if stopped.is_none() {
                        let s = dec.finish();
                        match mode {
                            Mode::Reasoning => {
                                reasoning.push(&s);
                                let r = reasoning.take(&[]);
                                if !r.is_empty() {
                                    send!(ChatEvent::Reasoning(r));
                                }
                            }
                            Mode::Content => {
                                content.push(&s);
                                let c = content.take(&[]);
                                if !c.is_empty() {
                                    send!(ChatEvent::Content(c));
                                }
                            }
                            Mode::Tool => {}
                        }
                    }
                    let engine_finish = ev.get("finish").and_then(Value::as_str).unwrap_or("stop");
                    let finish = if stopped.is_some() {
                        Finish::StopSequence
                    } else if engine_finish == "length" {
                        Finish::Length
                    } else if engine_finish == "cancelled" {
                        Finish::Cancelled
                    } else if n_tools > 0 {
                        Finish::ToolCalls
                    } else {
                        Finish::Stop
                    };
                    let n = |k: &str| ev.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
                    tracing::info!(
                        "{id}: prompt {} (reused {}) in {:.0} ms, {} tokens in {:.0} ms, {:?}",
                        n("prompt_tokens"),
                        n("reused"),
                        ev.get("prompt_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                        n("generated"),
                        ev.get("decode_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                        finish
                    );
                    send!(ChatEvent::Done {
                        finish,
                        stop_sequence: stopped.clone(),
                        prompt_tokens: n("prompt_tokens"),
                        completion_tokens: n("generated"),
                        reused: n("reused"),
                        timings: timings_of(&ev),
                    });
                    break;
                }
                Some("error") => {
                    let msg = ev.get("msg").and_then(Value::as_str).unwrap_or("engine error").to_string();
                    tracing::error!("{id}: engine error: {msg}");
                    send!(ChatEvent::Error(msg));
                    break;
                }
                _ => {}
            }
        }
        let _ = gone;
    });
    Ok((out, n_prompt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn types() -> std::collections::HashMap<String, Map<String, Value>> {
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "opts": {"type": "object"}}}}}]);
        tool_param_types(Some(&tools))
    }

    #[test]
    fn tool_call_typed_arguments() {
        let body = "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n\
                    <parameter=opts>\n{\"unit\": \"C\"}\n</parameter>\n</function>";
        let (name, args) = parse_tool_call(body, &types()).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args, json!({"city": "Paris", "days": 3, "opts": {"unit": "C"}}));
    }

    #[test]
    fn tool_call_untyped_and_unparsable_stay_strings() {
        let t = types();
        let (_, a) = parse_tool_call("<function=other>\n<parameter=n>\n3\n</parameter>\n</function>", &t).unwrap();
        assert_eq!(a, json!({"n": "3"}));   // no schema: a string
        let (_, a) = parse_tool_call("<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>", &t).unwrap();
        assert_eq!(a, json!({"days": "three"}));   // not JSON: kept as text
        assert!(parse_tool_call("no call here", &t).is_none());
    }

    #[test]
    fn tool_call_multiline_value_and_missing_close() {
        let (_, a) = parse_tool_call("<function=f>\n<parameter=code>\nline 1\nline 2\n</function>", &Default::default()).unwrap();
        assert_eq!(a, json!({"code": "line 1\nline 2"}));
    }

    /// An AppState over a fake engine (a shell script speaking the protocol) and the test
    /// tokenizer; the template joins the messages' contents.
    async fn fake_state(script: &str) -> Arc<AppState> {
        let engine = crate::engine::Engine::spawn("sh", &["-c".into(), script.to_string()]).await.unwrap();
        Arc::new(AppState {
            engine,
            tokenizer: crate::tokenizer::test_tokenizer(),
            template: crate::template::ChatTemplate::new("{% for m in messages %}{{ m.content }}{% endfor %}").unwrap(),
            model_name: "test".into(),
            max_context: 4096,
            default_max_tokens: 16,
            sampling: crate::Sampling { temperature: 0.0, top_p: 1.0, top_k: 20, min_p: 0.0 },
            stop_ids: vec![261],
            ids: crate::SpecialIds { think: 256, think_end: 257, tool_call: 258, tool_call_end: 259 },
            api_key: None,
        })
    }

    const READY: &str = r#"echo '{"ev":"ready","version":"0","arch":"test","max_context":4096,"features":[]}'"#;

    fn request(text: &str) -> ChatRequest {
        let mut r = crate::openai::empty_request();
        r.messages = json!([{"role": "user", "content": text}]);
        r
    }

    #[tokio::test]
    async fn control_and_stray_structure_tokens_are_not_text() {
        // H i <|im_start|> </think> ! : the control token and the stray </think> (not in a
        // reasoning section) must not reach the answer
        let tok = |t: u32| format!(r#"echo '{{"ev":"token","id":"r0","tok":{t}}}'; "#);
        let script = format!(
            r#"{READY}; read line; {}{}{}{}{} echo '{{"ev":"done","id":"r0","generated":5,"finish":"stop"}}'; read line"#,
            tok(72), tok(105), tok(260), tok(257), tok(33)
        );
        let st = fake_state(&script).await;
        let (mut rx, _) = start(st, request("x")).await.unwrap();
        let mut text = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::Content(s) => text.push_str(&s),
                ChatEvent::Done { .. } => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(text, "Hi!");
    }

    #[tokio::test]
    async fn client_leaving_while_queued_cancels() {
        // the fake engine never starts the request; it only answers a stop for it
        let script = format!(
            r#"{READY}; read line; read line; case "$line" in *'"stop"'*) echo '{{"ev":"done","id":"r0","generated":0,"finish":"cancelled"}}';; esac; read line"#
        );
        let st = fake_state(&script).await;
        let (rx, _) = start(st.clone(), request("x")).await.unwrap();
        drop(rx);   // the client goes away before any token
        // the engine must get the stop: its done event ends the route
        let t0 = std::time::Instant::now();
        while st.engine.has_route("r0") {
            assert!(t0.elapsed() < std::time::Duration::from_secs(2), "no stop reached the engine");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn sampling_limits() {
        let st = fake_state(&format!("{READY}; read line")).await;
        let mut r = request("x");
        r.top_k = Some(0);
        assert_eq!(sampling_of(&st, &r).unwrap().2, 64);   // 0: no limit, the engine's most
        r.top_k = Some(64);
        assert_eq!(sampling_of(&st, &r).unwrap().2, 64);
        r.top_k = Some(65);
        assert!(sampling_of(&st, &r).is_err());   // not silently capped
        r.top_k = None;
        assert_eq!(sampling_of(&st, &r).unwrap().2, 20);   // the model's default
        r.temperature = Some(-0.5);
        assert!(sampling_of(&st, &r).is_err());
        r.temperature = None;
        r.top_p = Some(0.0);
        assert!(sampling_of(&st, &r).is_err());
        r.top_p = None;
        r.min_p = Some(1.0);
        assert!(sampling_of(&st, &r).is_err());
        r.min_p = None;
        r.seed = Some(u64::MAX);
        assert_eq!(sampling_of(&st, &r).unwrap().4, (1u64 << 53) - 1);   // exact in the protocol's JSON numbers
    }

    #[test]
    fn tool_call_multibyte_after_function_close() {
        // a value closed by </function> (11 bytes) followed by multibyte text: the cursor must move
        // by the tag found, not by "</parameter>" (12 bytes), which would split a character
        let (_, a) = parse_tool_call("<function=f>\n<parameter=x>\n1\n</function>é<parameter=y>\n2\n</parameter>", &Default::default())
            .unwrap();
        assert_eq!(a, json!({"x": "1", "y": "2"}));
    }

    #[test]
    fn tool_call_parser_never_panics() {
        // random UTF-8 around the tags (a small deterministic fuzz)
        let parts = ["<function=f>", "<parameter=", "p>", "</parameter>", "</function>", "é", "日本", "🦀", "\n", "x", ">", "<"];
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..20_000 {
            let mut body = String::new();
            for _ in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                body.push_str(parts[(state % parts.len() as u64) as usize]);
            }
            let _ = parse_tool_call(&body, &Default::default());
        }
    }

    #[test]
    fn section_trims_and_holds_back() {
        let mut s = Section::new();
        s.push("   ");
        assert_eq!(s.take(&[]), "");   // leading whitespace dropped
        s.push("hello ");
        assert_eq!(s.take(&[]), "hello");   // trailing whitespace held back
        s.push("world");
        assert_eq!(s.take(&[]), " world");
    }

    #[test]
    fn section_stop_strings() {
        let stops = vec!["STOP".to_string()];
        let mut s = Section::new();
        s.push("abc ST");
        assert_eq!(s.take(&stops), "abc ");   // a possible stop prefix is held back
        s.push("OP tail");
        let (stop, piece) = s.stop_hit(&stops).unwrap();
        assert_eq!(stop, "STOP");
        assert_eq!(piece, "");
        assert_eq!(s.take(&stops), "");   // nothing after the stop is sent
    }

    #[test]
    fn raw_section_keeps_whitespace() {
        let mut s = Section::raw();
        s.push("  x  ");
        assert_eq!(s.take(&[]), "  x  ");
    }

    #[test]
    fn timings_follow_the_done_event() {
        let t = timings_of(&json!({"prompt_tokens": 1000, "reused": 600, "generated": 50, "prompt_ms": 200.0,
            "decode_ms": 500.0, "drafts": {"proposed": 40, "accepted": 25}, "cache": {"hits": 900, "misses": 100}}));
        assert_eq!((t["prompt_n"].as_u64(), t["cache_n"].as_u64(), t["predicted_n"].as_u64()), (Some(400), Some(600), Some(50)));
        assert!((t["prompt_per_second"].as_f64().unwrap() - 2000.0).abs() < 1e-9);
        assert!((t["predicted_per_second"].as_f64().unwrap() - 100.0).abs() < 1e-9);
        assert_eq!((t["draft_n"].as_u64(), t["draft_n_accepted"].as_u64()), (Some(40), Some(25)));
        assert_eq!(t["expert_cache"], json!({"hits": 900, "misses": 100}));
        // a cancelled request's bare event: no rates from zero times, no draft or cache fields
        let bare = timings_of(&json!({"prompt_tokens": 5, "reused": 9, "generated": 0}));
        assert_eq!((bare["prompt_n"].as_u64(), bare["prompt_per_second"].as_f64()), (Some(0), Some(0.0)));
        assert!(bare.get("draft_n").is_none() && bare.get("expert_cache").is_none());
    }
}
