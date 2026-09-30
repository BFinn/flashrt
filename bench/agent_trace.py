#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""An agentic coding session against a running flashrt-server (phase 2 of
docs/improvement-plan.md): a multi-turn conversation with code context, tool calls and tool
results, the regime where the generation routes unlike its prompt.

  agent_trace.py --repo DIR [--url http://127.0.0.1:8090] [--turns 12] [--max-tokens 512]
                 [--temperature 0] [--label X]

The model gets two tools over DIR (a source tree, read-only): read_file and list_dir, and a
review task. Each turn, the tool calls it makes are answered (file contents are cut at
--max-file-chars); when it makes none, the next file of a fixed list comes as a user message, so
the context grows either way. Reasoning is on, and the template drops it from earlier turns, so
each prompt diverges from the previous sequence inside the last assistant turn: reuse comes from
the end-of-prompt checkpoint.

Per turn, from the response's `timings`: prompt tokens (reused and prefilled), time to the
answer's first token (the prompt time), decode tok/s, draft acceptance, the decode's expert-cache
hit rate and misses per generated token. A JSON summary line (SUMMARY ...) ends the output.
Standard library only.
"""
import argparse
import json
import os
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path relative to the repository root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path relative to the repository root"}},
                       "required": ["path"]}}},
]
SYSTEM = ("You are a careful C++, CUDA and Rust reviewer working in a repository through tools. Read the files you need, "
          "then report concrete defects with file and line, the failure they cause, and a fix. Be brief between tool calls.")
TASK = ("Review how the engine reuses a previous conversation's state: engine/session.cpp first, then what it calls in "
        "arch/qwen4exp/forward.cu and arch/qwen4exp/mtp.cu. Look for states that can be restored for the wrong tokens.")
# given in order when the model calls no tool
NEXT_FILES = ["engine/session.hpp", "engine/session.cpp", "arch/qwen4exp/forward.hpp", "arch/qwen4exp/mtp.hpp",
              "engine/main.cpp", "server/src/engine.rs", "server/src/chat.rs", "arch/qwen4exp/moe_fast.hpp",
              "core/cpu_pool.hpp", "arch/qwen4exp/blocks.hpp", "docs/design.md", "bench/engine_smoke.py"]


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-file-chars", type=int, default=24000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="agent")
    a = ap.parse_args()
    root = os.path.realpath(a.repo)

    def inside(p):
        full = os.path.realpath(os.path.join(root, p))
        return full if full == root or full.startswith(root + os.sep) else None

    def tool(name, args):
        path = inside(str(args.get("path", "")))
        if path is None:
            return "error: outside the repository"
        try:
            if name == "list_dir":
                return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(path, e)) else "") for e in os.listdir(path)))
            if name == "read_file":
                with open(path, errors="replace") as f:
                    text = f.read()
                return text if len(text) <= a.max_file_chars else text[:a.max_file_chars] + f"\n... ({len(text) - a.max_file_chars} more characters)"
        except OSError as e:
            return f"error: {e}"
        return f"error: unknown tool {name}"

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    rows, nxt = [], 0
    for turn in range(a.turns):
        t0 = time.time()
        r = post(a.url, {"messages": msgs, "tools": TOOLS, "max_tokens": a.max_tokens, "temperature": a.temperature,
                         "top_k": 20, "top_p": 0.95 if a.temperature > 0 else 1.0, "seed": 1234 + turn})
        wall = time.time() - t0
        tm, c = r.get("timings", {}), r["choices"][0]
        ec = tm.get("expert_cache", {})
        hits, misses = ec.get("hits", 0), ec.get("misses", 0)
        gen = tm.get("predicted_n", 0)
        calls = c["message"].get("tool_calls") or []
        row = {"turn": turn, "prompt": r["usage"]["prompt_tokens"], "reused": tm.get("cache_n", 0), "prefilled": tm.get("prompt_n", 0),
               "ttft_s": round(tm.get("prompt_ms", 0) / 1000, 2), "wall_s": round(wall, 2), "generated": gen,
               "decode_tps": round(tm.get("predicted_per_second", 0), 2),
               "accept": round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
               "hit_rate": round(hits / max(1, hits + misses), 4), "misses_per_token": round(misses / max(1, gen), 2),
               "finish": c["finish_reason"], "tool_calls": [x["function"]["name"] for x in calls]}
        rows.append(row)
        print(f"{a.label} turn {turn:2d} | prompt {row['prompt']:>6} (reused {row['reused']:>6}, prefilled {row['prefilled']:>5}) "
              f"ttft {row['ttft_s']:>6} s | gen {gen:>4} at {row['decode_tps']:>6} tok/s | accept {row['accept']} | "
              f"hits {100 * row['hit_rate']:.1f}%, {row['misses_per_token']} misses/token | {row['finish']} {row['tool_calls']}",
              flush=True)
        msg = {"role": "assistant", "content": c["message"].get("content") or ""}
        if c["message"].get("reasoning_content"):
            msg["reasoning_content"] = c["message"]["reasoning_content"]
        if calls:
            msg["tool_calls"] = calls
        msgs.append(msg)
        if calls:
            for x in calls:
                try:
                    args = json.loads(x["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": x["id"], "content": tool(x["function"]["name"], args)})
        else:
            f = NEXT_FILES[nxt % len(NEXT_FILES)]
            nxt += 1
            msgs.append({"role": "user", "content": f"Continue with {f}:\n\n```\n{tool('read_file', {'path': f})}\n```"})
    gen = sum(r["generated"] for r in rows)
    dec_s = sum(r["generated"] / r["decode_tps"] for r in rows if r["decode_tps"] > 0)
    print("SUMMARY " + json.dumps({"label": a.label, "turns": len(rows), "final_prompt": rows[-1]["prompt"] if rows else 0,
                                   "ttft_total_s": round(sum(r["ttft_s"] for r in rows), 1), "generated": gen,
                                   "decode_tps": round(gen / dec_s, 2) if dec_s else 0, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
