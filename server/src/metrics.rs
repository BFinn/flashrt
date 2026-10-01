// SPDX-License-Identifier: Apache-2.0
//! Running totals over the engine's `done` events, served at `/metrics` in Prometheus's text
//! format (phase 6): requests by finish, tokens, prompt and decode time, drafts proposed and
//! accepted, the decode's expert-cache hits and misses, the host's time running the misses, and
//! the cache's slot count; then the last request's rates, for a reader without Prometheus.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;

use serde_json::Value;

#[derive(Default)]
struct Totals {
    finished: BTreeMap<String, u64>,
    errors: u64,
    prompt_tokens: u64,
    reused: u64,
    generated: u64,
    prompt_ms: f64,
    decode_ms: f64,
    drafts_proposed: u64,
    drafts_accepted: u64,
    cache_hits: u64,
    cache_misses: u64,
    miss_ms: f64,
    cache_slots: Option<u64>,
    last: Option<Last>,
}

#[derive(Clone, Copy)]
struct Last {
    decode_tps: f64,
    prefill_tps: f64,
    hit_ratio: Option<f64>,
    acceptance: Option<f64>,
}

#[derive(Default)]
pub struct Metrics {
    t: Mutex<Totals>,
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    let mut x = v;
    for k in path {
        x = x.get(*k)?;
    }
    x.as_f64()
}

impl Metrics {
    /// One finished generation (the engine's `done` event; absent fields count as zero).
    pub fn record_done(&self, done: &Value) {
        let n = |p: &[&str]| num(done, p).unwrap_or(0.0);
        let finish = done.get("finish").and_then(Value::as_str).unwrap_or("stop").to_string();
        let mut t = self.t.lock().unwrap();
        *t.finished.entry(finish).or_default() += 1;
        let (prompt, reused, generated) = (n(&["prompt_tokens"]), n(&["reused"]), n(&["generated"]));
        let (prompt_ms, decode_ms) = (n(&["prompt_ms"]), n(&["decode_ms"]));
        let (proposed, accepted) = (n(&["drafts", "proposed"]), n(&["drafts", "accepted"]));
        let (hits, misses) = (n(&["cache", "hits"]), n(&["cache", "misses"]));
        t.prompt_tokens += prompt as u64;
        t.reused += reused as u64;
        t.generated += generated as u64;
        t.prompt_ms += prompt_ms;
        t.decode_ms += decode_ms;
        t.drafts_proposed += proposed as u64;
        t.drafts_accepted += accepted as u64;
        t.cache_hits += hits as u64;
        t.cache_misses += misses as u64;
        t.miss_ms += n(&["cache", "miss_ms"]);
        if let Some(s) = num(done, &["cache", "slots"]) {
            t.cache_slots = Some(s as u64);
        }
        if generated > 0.0 {
            let rate = |k: f64, ms: f64| if ms > 0.0 { k * 1000.0 / ms } else { 0.0 };
            t.last = Some(Last {
                decode_tps: rate(generated, decode_ms),
                prefill_tps: rate(prompt - reused, prompt_ms),
                hit_ratio: (hits + misses > 0.0).then(|| hits / (hits + misses)),
                acceptance: (proposed > 0.0).then(|| accepted / proposed),
            });
        }
    }

    /// A request the engine answered with an `error` event.
    pub fn record_error(&self) {
        self.t.lock().unwrap().errors += 1;
    }

    pub fn render(&self, engine_up: bool) -> String {
        let t = self.t.lock().unwrap();
        let mut o = String::new();
        let mut metric = |name: &str, kind: &str, help: &str, rows: &[(String, f64)]| {
            let _ = writeln!(o, "# HELP flashrt_{name} {help}\n# TYPE flashrt_{name} {kind}");
            for (labels, v) in rows {
                let _ = writeln!(o, "flashrt_{name}{labels} {v}");
            }
        };
        let one = |v: f64| vec![(String::new(), v)];
        metric("engine_up", "gauge", "1 while the engine process is serving.", &one(f64::from(u8::from(engine_up))));
        let by_finish: Vec<(String, f64)> = t.finished.iter().map(|(k, v)| (format!("{{finish=\"{k}\"}}"), *v as f64)).collect();
        metric("requests_total", "counter", "Generations finished, by how they ended.", &by_finish);
        metric("request_errors_total", "counter", "Generations the engine answered with an error.", &one(t.errors as f64));
        metric("prompt_tokens_total", "counter", "Prompt tokens, reused ones included.", &one(t.prompt_tokens as f64));
        metric("prompt_tokens_reused_total", "counter", "Prompt tokens taken from the previous sequence.", &one(t.reused as f64));
        metric("generated_tokens_total", "counter", "Tokens generated.", &one(t.generated as f64));
        metric("prompt_seconds_total", "counter", "Engine time spent on prompts.", &one(t.prompt_ms / 1000.0));
        metric("decode_seconds_total", "counter", "Engine time spent generating.", &one(t.decode_ms / 1000.0));
        metric("drafts_proposed_total", "counter", "MTP drafts verified.", &one(t.drafts_proposed as f64));
        metric("drafts_accepted_total", "counter", "MTP drafts kept.", &one(t.drafts_accepted as f64));
        metric("expert_cache_hits_total", "counter", "Routed experts found in the VRAM cache while generating.", &one(t.cache_hits as f64));
        metric("expert_cache_misses_total", "counter", "Routed experts computed on the CPU while generating.", &one(t.cache_misses as f64));
        metric("cpu_miss_seconds_total", "counter", "Host time computing the missed experts.", &one(t.miss_ms / 1000.0));
        if let Some(s) = t.cache_slots {
            metric("expert_cache_slots", "gauge", "Expert slots in the VRAM cache.", &one(s as f64));
        }
        if let Some(l) = t.last {
            metric("last_decode_tokens_per_second", "gauge", "The last generation's decode rate.", &one(l.decode_tps));
            metric("last_prefill_tokens_per_second", "gauge", "The last generation's prefill rate over its new tokens.", &one(l.prefill_tps));
            if let Some(h) = l.hit_ratio {
                metric("last_expert_cache_hit_ratio", "gauge", "The last generation's expert-cache hit ratio.", &one(h));
            }
            if let Some(a) = l.acceptance {
                metric("last_draft_acceptance_ratio", "gauge", "The last generation's share of drafts kept.", &one(a));
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn totals_and_last_follow_the_done_events() {
        let m = Metrics::default();
        m.record_done(&json!({"generated": 100, "prompt_tokens": 1000, "reused": 200, "prompt_ms": 400.0, "decode_ms": 1000.0,
            "finish": "length", "drafts": {"proposed": 80, "accepted": 60},
            "cache": {"hits": 900, "misses": 100, "slots": 8800, "miss_ms": 250.0}}));
        m.record_done(&json!({"generated": 0, "finish": "cancelled"}));
        m.record_error();
        let r = m.render(true);
        for line in [
            "flashrt_engine_up 1",
            "flashrt_requests_total{finish=\"cancelled\"} 1",
            "flashrt_requests_total{finish=\"length\"} 1",
            "flashrt_request_errors_total 1",
            "flashrt_prompt_tokens_reused_total 200",
            "flashrt_generated_tokens_total 100",
            "flashrt_drafts_accepted_total 60",
            "flashrt_expert_cache_misses_total 100",
            "flashrt_cpu_miss_seconds_total 0.25",
            "flashrt_expert_cache_slots 8800",
            "flashrt_last_decode_tokens_per_second 100",
            "flashrt_last_prefill_tokens_per_second 2000",
            "flashrt_last_expert_cache_hit_ratio 0.9",
            "flashrt_last_draft_acceptance_ratio 0.75",
        ] {
            assert!(r.lines().any(|l| l == line), "missing {line:?} in\n{r}");
        }
    }

    #[test]
    fn an_engine_without_the_new_fields_still_counts() {
        // an older engine: no slots, no miss time; no generation yet means no last-request gauges
        let m = Metrics::default();
        let r = m.render(false);
        assert!(r.contains("flashrt_engine_up 0") && !r.contains("expert_cache_slots ") && !r.contains("last_decode"));
        m.record_done(&json!({"generated": 4, "decode_ms": 40.0, "finish": "stop", "cache": {"hits": 3, "misses": 1}}));
        let r = m.render(true);
        assert!(r.contains("flashrt_cpu_miss_seconds_total 0\n") && !r.contains("expert_cache_slots "));
        assert!(r.contains("flashrt_last_expert_cache_hit_ratio 0.75") && !r.contains("last_draft_acceptance"));
    }
}
