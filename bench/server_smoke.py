#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of flashrt-server's OpenAI and Anthropic APIs against a running server.

  server_smoke.py [--url http://127.0.0.1:8080] [--key KEY]

Each check prints a one-line result (and timings); the script exits non-zero if any fails.
Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

A = None
FAILED = []


def post(path, body, stream=False, headers=None):
    h = {"Content-Type": "application/json"}
    if A.key:
        h["Authorization"] = f"Bearer {A.key}"
    h.update(headers or {})
    req = urllib.request.Request(A.url + path, data=json.dumps(body).encode(), headers=h, method="POST")
    r = urllib.request.urlopen(req, timeout=1800)
    if not stream:
        return json.loads(r.read())
    return r


def sse(resp):
    """Yields (event, data) of an SSE response; data parsed as JSON when it is."""
    event = None
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            d = line[5:].strip()
            try:
                d = json.loads(d)
            except json.JSONDecodeError:
                pass
            yield event, d
            event = None


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        FAILED.append(name)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city"]}}}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key")
    A = ap.parse_args()

    m = json.loads(urllib.request.urlopen(A.url + "/v1/models").read())
    check("models", m["data"][0]["id"] != "", json.dumps(m["data"][0]))

    # 1. plain chat, reasoning off, greedy
    t = time.time()
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}],
                                      "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]
    check("chat, no thinking", "391" in (c["message"]["content"] or ""),
          f"{c['message']['content']!r}, finish {c['finish_reason']}, {r['usage']}, {time.time() - t:.1f} s")
    tm = r.get("timings", {})
    ec = tm.get("expert_cache", {})
    check("timings", tm.get("prompt_n", -1) + tm.get("cache_n", -1) == r["usage"]["prompt_tokens"]
          and tm.get("predicted_n") == r["usage"]["completion_tokens"] and tm.get("predicted_ms", 0) > 0
          and ec.get("hits", 0) + ec.get("misses", 0) > 0, json.dumps(tm))

    # 2. streaming with reasoning
    t = time.time()
    resp = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Name three primary colours, briefly."}],
                                         "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}}, stream=True)
    reasoning, content, finish, usage, first = "", "", None, None, None
    for _, d in sse(resp):
        if d == "[DONE]":
            break
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                first = first or time.time()
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
            finish = ch.get("finish_reason") or finish
    dt = time.time() - (first or t)
    rate = usage["completion_tokens"] / dt if usage and dt > 0 else 0
    check("chat stream with reasoning", len(reasoning) > 0 and len(content) > 0 and finish == "stop",
          f"{len(reasoning)} chars reasoning, content {content[:80]!r}, finish {finish}, {usage}, {rate:.1f} tok/s after the first token")

    # 3. tool call
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    r = post("/v1/chat/completions", {"messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 0})
    c = r["choices"][0]
    calls = c["message"].get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("tool call", c["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
          and args.get("city", "").lower().startswith("paris"), f"{calls}, usage {r['usage']}")

    # 4. the conversation continues with the tool result: the prompt prefix is reused
    if calls:
        msgs.append(c["message"])
        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_c": 18, "sky": "cloudy"})})
        r = post("/v1/chat/completions", {"messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 0})
        c = r["choices"][0]
        cached = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        check("tool result turn, prefix reuse", "18" in (c["message"]["content"] or "") and cached > 0,
              f"{(c['message']['content'] or '')[:100]!r}, cached {cached} of {r['usage']['prompt_tokens']}")

    # 5. stop string
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from 1 to 20, comma separated."}],
                                      "temperature": 0, "max_tokens": 200, "stop": ["7"],
                                      "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]
    check("stop string", "7" not in c["message"]["content"] and "6" in c["message"]["content"],
          f"{c['message']['content']!r}, finish {c['finish_reason']}")

    # 6. raw completion
    r = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    check("raw completion", "Paris" in r["choices"][0]["text"], f"{r['choices'][0]['text']!r}")

    # 7. Anthropic, thinking shown, non-streaming
    r = post("/v1/messages", {"model": "x", "max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024},
                              "system": "Be brief.", "messages": [{"role": "user", "content": "Is 91 prime?"}]})
    kinds = [b["type"] for b in r["content"]]
    text = " ".join(b.get("text", "") for b in r["content"])
    check("anthropic messages", kinds[:1] == ["thinking"] and "text" in kinds and r["stop_reason"] == "end_turn",
          f"blocks {kinds}, {text[:80]!r}, {r['usage']}")

    # 8. Anthropic streaming tool use
    resp = post("/v1/messages", {"model": "x", "max_tokens": 2048, "stream": True, "tool_choice": {"type": "auto"},
                                 "tools": [{"name": "get_weather", "description": "Current weather for a city",
                                            "input_schema": WEATHER["function"]["parameters"]}],
                                 "messages": [{"role": "user", "content": "Weather in Oslo? Use the tool."}]}, stream=True)
    events, tool_json, stop = [], "", None
    for ev, d in sse(resp):
        events.append(ev)
        if ev == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
            tool_json += d["delta"]["partial_json"]
        if ev == "message_delta":
            stop = d["delta"]["stop_reason"]
        if ev == "message_stop":
            break
    inp = json.loads(tool_json) if tool_json else {}
    check("anthropic stream tool_use", stop == "tool_use" and inp.get("city", "").lower().startswith("oslo")
          and events[0] == "message_start", f"stop {stop}, input {inp}, {len(events)} events")

    # 9. count_tokens
    r = post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": "Hello there"}]})
    check("count_tokens", r["input_tokens"] > 3, json.dumps(r))

    # 10. a client that leaves mid-stream: the next request must still be served promptly
    resp = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a long essay about the sea."}],
                                         "max_tokens": 4000, "stream": True}, stream=True)
    n = 0
    for _, d in sse(resp):
        n += 1
        if n > 20:
            break
    resp.close()
    t = time.time()
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 16, "temperature": 0,
                                      "chat_template_kwargs": {"enable_thinking": False}})
    check("disconnect cancels", time.time() - t < 30, f"next request answered in {time.time() - t:.1f} s: {r['choices'][0]['message']['content']!r}")

    # 11. sampling limits: top_k above the engine's 64 and a negative temperature are 400s, not
    # silent caps; top_k 0 means no limit
    for name, extra in [("top_k 65", {"top_k": 65}), ("temperature -1", {"temperature": -1})]:
        try:
            post("/v1/chat/completions", dict({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}, **extra))
            check(f"{name} rejected", False, "accepted")
        except urllib.error.HTTPError as e:
            check(f"{name} rejected", e.code == 400, f"HTTP {e.code}: {json.loads(e.read())['error']['message']}")
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 16, "top_k": 0,
                                      "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}})
    text = r["choices"][0]["message"]["content"] or ""
    check("top_k 0 accepted", "<|im" not in text, repr(text))

    # 12. /metrics: the totals over the generations above, in Prometheus's text format
    req = urllib.request.Request(A.url + "/metrics", headers={"Authorization": f"Bearer {A.key}"} if A.key else {})
    mt = {}
    for line in urllib.request.urlopen(req, timeout=60).read().decode().splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            mt[k] = float(v)
    done = sum(v for k, v in mt.items() if k.startswith("flashrt_requests_total{"))
    check("metrics", mt.get("flashrt_engine_up") == 1 and done >= 10 and mt.get("flashrt_generated_tokens_total", 0) > 0
          and mt.get("flashrt_expert_cache_hits_total", 0) > 0 and mt.get("flashrt_expert_cache_slots", 0) > 0,
          f"{done:.0f} requests, {mt.get('flashrt_generated_tokens_total', 0):.0f} tokens, "
          f"slots {mt.get('flashrt_expert_cache_slots', 0):.0f}, last hit ratio {mt.get('flashrt_last_expert_cache_hit_ratio', 0):.3f}, "
          f"CPU miss time {mt.get('flashrt_cpu_miss_seconds_total', 0):.2f} s")

    print("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
