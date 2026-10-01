// SPDX-License-Identifier: Apache-2.0
//! The model's chat template (the GGUF's tokenizer.chat_template, Jinja) rendered with minijinja.
//! Python string methods (startswith, split, ...) come from minijinja-contrib's pycompat;
//! `tojson` matches Python's json.dumps(ensure_ascii=False) as Hugging Face templates expect
//! (", " and ": " separators, keys in insertion order), since that is the text the model saw.
//!
//! A request the template rejects with raise_exception (a system message after the first, an
//! unknown role) is the request's error (HTTP 400); any other failure to render is the server's
//! (`crate::ServerFault`, HTTP 500).

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

/// The source of raise_exception's errors: how `render` tells the template rejecting the request
/// from the template failing.
#[derive(Debug)]
struct Raised;

impl std::fmt::Display for Raised {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("raised by the template")
    }
}

impl std::error::Error for Raised {}

fn raise_exception(msg: String) -> Result<JValue, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg).with_source(Raised))
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
    /// arguments as objects); `extra` holds template variables such as enable_thinking. An error
    /// is the request's when the template raised it, else a `crate::ServerFault`.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, extra: &serde_json::Map<String, Value>) -> Result<String> {
        // the extras first, so they cannot replace the conversation itself
        let mut ctx = extra.clone();
        ctx.insert("messages".into(), messages.clone());
        match tools {
            Some(t) => ctx.insert("tools".into(), t.clone()),
            None => ctx.remove("tools"),
        };
        ctx.insert("add_generation_prompt".into(), Value::Bool(true));
        let t = self.env.get_template("chat").map_err(|e| anyhow::Error::new(crate::ServerFault(format!("chat template: {e}"))))?;
        t.render(JValue::from_serialize(Value::Object(ctx))).map_err(|e| {
            let mut msg = e.to_string();
            let mut raised = false;
            let mut src = std::error::Error::source(&e);
            while let Some(s) = src {
                if s.is::<Raised>() {
                    raised = true;
                } else {
                    msg.push_str(&format!(": {s}"));
                }
                src = s.source();
            }
            if raised {
                anyhow!("chat template: {msg}")
            } else {
                anyhow::Error::new(crate::ServerFault(format!("the chat template failed to render: {msg}")))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_json_matches_python_dumps() {
        // separators ", " and ": ", insertion order, non-ASCII kept (ensure_ascii=False)
        let v = json!({"b": [true, null, "x"], "a": 1, "c": {"d": "é\"q"}});
        let mut s = String::new();
        py_json(&v, &mut s);
        assert_eq!(s, r#"{"b": [true, null, "x"], "a": 1, "c": {"d": "é\"q"}}"#);
    }

    #[test]
    fn renders_messages_tools_and_python_methods() {
        let t = ChatTemplate::new(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% if m.content.startswith('hi') %}!{% endif %}\n{% endfor %}\
             {% if tools %}T={{ tools | tojson }}\n{% endif %}{% if add_generation_prompt %}<assistant>{% endif %}",
        )
        .unwrap();
        let msgs = json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi there"}]);
        let tools = json!([{"name": "f", "parameters": {"x": 1}}]);
        let out = t.render(&msgs, Some(&tools), &serde_json::Map::new()).unwrap();
        // trim_blocks (as Hugging Face renders chat templates): the newline right after a block
        // tag ({% endif %} in the loop) is dropped; the one after an expression is kept
        assert_eq!(out, "<system>be brief<user>hi there!T=[{\"name\": \"f\", \"parameters\": {\"x\": 1}}]\n<assistant>");
    }

    #[test]
    fn template_kwargs_cannot_replace_the_conversation() {
        let t = ChatTemplate::new("{{ messages | length }} {{ tools is defined }} {{ add_generation_prompt }} {{ enable_thinking }}").unwrap();
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let extra = json!({"messages": [], "tools": [{"name": "x"}], "add_generation_prompt": false, "enable_thinking": false});
        let out = t.render(&msgs, None, extra.as_object().unwrap()).unwrap();
        assert_eq!(out, "1 False True False");   // booleans print as Python does
    }

    #[test]
    fn raise_exception_is_the_requests_error_and_other_failures_the_servers() {
        let none = serde_json::Map::new();
        let t = ChatTemplate::new("{{ raise_exception('bad role') }}").unwrap();
        let e = t.render(&json!([]), None, &none).unwrap_err();
        assert!(e.to_string().contains("bad role") && !e.to_string().contains("raised by"), "{e}");
        assert!(!e.is::<crate::ServerFault>());
        // raised inside a macro whose output is filtered, as the model's template does
        let t = ChatTemplate::new(
            "{% macro rc(c) %}{% if c is string %}{{ c }}{% else %}{{ raise_exception('Unexpected content type.') }}{% endif %}{% endmacro %}\
             {% for m in messages %}{% set x = rc(m.content)|trim %}{{ x }}{% endfor %}",
        )
        .unwrap();
        assert_eq!(t.render(&json!([{"content": " a "}]), None, &none).unwrap(), "a");
        let e = t.render(&json!([{"content": 3}]), None, &none).unwrap_err();
        assert!(!e.is::<crate::ServerFault>() && e.to_string().contains("Unexpected content type"), "{e}");
        // a template that fails on its own (an operator on a type it does not take): the server's
        let t = ChatTemplate::new("{{ messages[0].content + 1 }}").unwrap();
        let e = t.render(&json!([{"content": "x"}]), None, &none).unwrap_err();
        assert!(e.is::<crate::ServerFault>(), "{e}");
        assert_eq!(crate::start_error_status(&e), 500);
    }
}
