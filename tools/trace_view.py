#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reads flashrt-server's request traces (--trace-dir; server/src/trace.rs): one line per request,
or with --full the rendered prompt, the reasoning, the text and the tool calls as the server parsed
them, then the engine's figures.

  trace_view.py TRACE.jsonl [--last N] [--id ID] [--full]

A line: time, request id, prompt tokens (reused), prefill and decode time, decode tok/s, drafts
kept / proposed, the expert-cache hit rate, finish, and the start of the answer or the tool calls.
"""
import argparse
import json
import sys


def summary(t):
    e, o = t.get("engine") or {}, t.get("output") or {}
    gen, dms = e.get("generated", 0), e.get("decode_ms", 0.0)
    tps = f"{gen * 1000 / dms:6.1f}" if dms else "     -"
    d, c = e.get("drafts") or {}, e.get("cache") or {}
    drafts = f"{d.get('accepted', 0)}/{d.get('proposed', 0)}" if d.get("proposed") else "-"
    hits = c.get("hits", 0) / max(1, c.get("hits", 0) + c.get("misses", 0)) if c else None
    tools = ", ".join(f"{x['name']}({json.dumps(x['arguments'], ensure_ascii=False)})" for x in o.get("tool_calls", []))
    what = tools or (o.get("content") or "").replace("\n", " ")
    if o.get("unparsed_tool_calls"):
        what = f"[{len(o['unparsed_tool_calls'])} unparsed tool call(s)] " + what
    if t.get("error"):
        what = f"[error: {t['error']}] " + what
    return (f"{t.get('ts', '')[:19]} {t.get('id', ''):>5} prompt {t.get('prompt', {}).get('tokens', 0):>6} "
            f"(reused {e.get('reused', 0):>6}) {e.get('prompt_ms', 0) / 1000:6.2f} s + {dms / 1000:5.2f} s, "
            f"{tps} tok/s, drafts {drafts:>7}, hits {'-' if hits is None else f'{hits:.0%}'}, "
            f"{o.get('finish', '-'):<13} {what[:80]}")


def full(t):
    o = t.get("output") or {}
    print("=" * 100)
    print(summary(t))
    print(f"sampling: {json.dumps(t.get('sampling'))}")
    print("--- rendered prompt ---")
    print(t.get("prompt", {}).get("text", ""))
    if o.get("reasoning"):
        print("--- reasoning ---")
        print(o["reasoning"])
    print("--- content ---")
    print(o.get("content", ""))
    for x in o.get("tool_calls", []):
        print(f"--- tool call {x['id']}: {x['name']}")
        print(json.dumps(x["arguments"], ensure_ascii=False, indent=2))
    for u in o.get("unparsed_tool_calls", []):
        print("--- tool call that did not parse ---")
        print(u)
    print(f"--- engine: {json.dumps(t.get('engine'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--last", type=int, default=0, help="only the last N requests")
    ap.add_argument("--id", help="only this request id (they restart at r0 with the server)")
    ap.add_argument("--full", action="store_true", help="the rendered prompt and the whole output")
    a = ap.parse_args()
    with open(a.trace, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if a.id:
        rows = [t for t in rows if t.get("id") == a.id]
    if a.last:
        rows = rows[-a.last:]
    for t in rows:
        if a.full:
            full(t)
        else:
            print(summary(t))
    if not rows:
        print("no matching requests", file=sys.stderr)


if __name__ == "__main__":
    main()
