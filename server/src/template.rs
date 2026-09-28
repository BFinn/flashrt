// SPDX-License-Identifier: Apache-2.0
//! The model's chat template (the GGUF's tokenizer.chat_template, Jinja) rendered with minijinja.
//! Python string methods (startswith, split, ...) come from minijinja-contrib's pycompat;
//! `tojson` matches Python's json.dumps(ensure_ascii=False) as Hugging Face templates expect
//! (", " and ": " separators, keys in insertion order), since that is the text the model saw.

use anyhow::{anyhow, Result};
use minijinja::{Environment, Error, ErrorKind, Value as JValue};
use serde_json::Value;

pub struct ChatTemplate {
    env: Environment<'static>,
}

/// Python json.dumps formatting of a JSON value.
pub fn py_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                py_json(x, out);
            }
            out.push('}');
        }
    }
}

fn tojson(v: JValue) -> Result<JValue, Error> {
    let j: Value = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut s = String::new();
    py_json(&j, &mut s);
    Ok(JValue::from_safe_string(s))
}

fn raise_exception(msg: String) -> Result<JValue, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_template_owned("chat", source.to_string()).map_err(|e| anyhow!("chat template: {e}"))?;
        Ok(Self { env })
    }

    /// Renders the conversation. `messages` and `tools` are OpenAI-style JSON (tool-call
    /// arguments as objects); `extra` holds template variables such as enable_thinking.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, extra: &serde_json::Map<String, Value>) -> Result<String> {
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), messages.clone());
        if let Some(t) = tools {
            ctx.insert("tools".into(), t.clone());
        }
        ctx.insert("add_generation_prompt".into(), Value::Bool(true));
        for (k, v) in extra {
            ctx.insert(k.clone(), v.clone());
        }
        let t = self.env.get_template("chat")?;
        t.render(JValue::from_serialize(&Value::Object(ctx))).map_err(|e| {
            let mut msg = e.to_string();
            let mut src = std::error::Error::source(&e);
            while let Some(s) = src {
                msg.push_str(&format!(": {s}"));
                src = s.source();
            }
            anyhow!("chat template: {msg}")
        })
    }
}
