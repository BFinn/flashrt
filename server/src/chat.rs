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
use serde_json::{Map, Value};
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
    Done { finish: Finish, stop_sequence: Option<String>, prompt_tokens: u32, completion_tokens: u32, reused: u32 },
    Error(String),
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
        let vend = rest[vstart..]
            .find("</parameter>")
            .or_else(|| rest[vstart..].find("</function>"))
            .map(|i| vstart + i)
            .unwrap_or(rest.len());
        let mut v = &rest[vstart..vend];
        v = v.strip_prefix('\n').unwrap_or(v);
        v = v.strip_suffix('\n').unwrap_or(v);
        let ty = schema.and_then(|s| s.get(&pname)).and_then(Value::as_str);
        let value = match ty {
            Some("string") | None => Value::String(v.to_string()),
            _ => serde_json::from_str::<Value>(v.trim()).unwrap_or_else(|_| Value::String(v.to_string())),
        };
        args.insert(pname, value);
        rest = &rest[(vend + "</parameter>".len()).min(rest.len())..];
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
                if best.map_or(true, |(b, _)| from + i < b) {
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

/// Renders, tokenizes and queues a chat request; events arrive on the receiver. Also returns
/// the prompt's length in tokens.
pub async fn start(st: Arc<AppState>, req: ChatRequest) -> Result<(mpsc::Receiver<ChatEvent>, u32)> {
    let raw = req.raw_prompt.is_some();
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
        temperature: req.temperature.unwrap_or(st.sampling.temperature),
        top_p: req.top_p.unwrap_or(st.sampling.top_p),
        top_k: req.top_k.unwrap_or(st.sampling.top_k).clamp(1, 64),
        min_p: req.min_p.unwrap_or(st.sampling.min_p),
        seed: req.seed.unwrap_or_else(random_u64),
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
        let (think_end, tool_start, tool_end) =
            if raw { (u32::MAX, u32::MAX, u32::MAX) } else { (st.ids.think_end, st.ids.tool_call, st.ids.tool_call_end) };
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
        while let Some(ev) = rx.recv().await {
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
                    });
                    break;
                }
                Some("error") => {
                    send!(ChatEvent::Error(ev.get("msg").and_then(Value::as_str).unwrap_or("engine error").to_string()));
                    break;
                }
                _ => {}
            }
        }
        let _ = gone;
    });
    Ok((out, n_prompt))
}
