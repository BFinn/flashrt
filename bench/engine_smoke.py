#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test of flashrt-engine's JSON-lines protocol (docs/design.md), with token ids.

Runs four requests against one engine process and prints each one's events summary:
  1. a prompt of N tokens from an ids file;
  2. that prompt plus the tokens request 1 generated plus a few more prompt tokens (the engine
     should reuse the whole previous sequence: "reused" close to the old length);
  3. request 1's prompt again (reuse from the checkpoint at the end of that prompt);
  4. request 1's prompt with a "stop" sent after a few tokens (finish "cancelled").
With --same-seed-check it also compares request 3's tokens with request 1's (same prompt, same
seed: the same tokens wherever the logits agree).

  engine_smoke.py ENGINE MODEL --ids IDS [--n 2048] [--gen 64] [--temp 1.0] [--seed 1] -- [engine args]
"""
import argparse
import json
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    ids = [int(x) for x in open(a.ids).read().split()]
    extra = [x for x in a.rest if x != "--"]
    p = subprocess.Popen([a.engine, a.model] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def read():
        line = p.stdout.readline()
        if not line:
            sys.exit("engine exited")
        return json.loads(line)

    t0 = time.time()
    ready = read()
    print(f"ready after {time.time() - t0:.1f} s: {ready}")
    sampling = {"temperature": a.temp, "top_k": 20, "top_p": 0.95, "seed": a.seed}

    def run(rid, prompt, stop_after=None):
        p.stdin.write(json.dumps({"op": "generate", "id": rid, "prompt": prompt, "max_new": a.gen, "sampling": sampling,
                                  "stop_ids": [248046]}) + "\n")
        p.stdin.flush()
        toks, t_first, t_start = [], None, time.time()
        while True:
            ev = read()
            if ev.get("ev") == "token":
                toks.append(ev["tok"])
                if t_first is None:
                    t_first = time.time()
                if stop_after is not None and len(toks) == stop_after:
                    p.stdin.write(json.dumps({"op": "stop", "id": rid}) + "\n")
                    p.stdin.flush()
            elif ev.get("ev") == "done":
                dec = ev["decode_ms"] / 1000
                print(f"{rid}: prompt {ev['prompt_tokens']} (reused {ev['reused']}) in {ev['prompt_ms'] / 1000:.1f} s, "
                      f"{ev['generated']} tokens in {dec:.2f} s ({ev['generated'] / max(dec, 1e-9):.1f} tok/s), finish {ev['finish']}, "
                      f"drafts {ev['drafts']}, first tokens {toks[:8]}")
                return toks
            elif ev.get("ev") == "error":
                print(f"{rid}: error {ev['msg']}")
                return toks

    prompt1 = ids[:a.n]
    t1 = run("r1", prompt1)
    run("r2", prompt1 + t1 + ids[a.n:a.n + 32])
    t3 = run("r3", prompt1)
    same = sum(1 for x, y in zip(t1, t3) if x == y)
    print(f"r3 vs r1 (same prompt and seed): {same} of {min(len(t1), len(t3))} tokens equal, first difference at "
          f"{next((i for i, (x, y) in enumerate(zip(t1, t3)) if x != y), None)}")
    run("r4", prompt1, stop_after=5)
    p.stdin.write(json.dumps({"op": "quit"}) + "\n")
    p.stdin.flush()
    p.wait(timeout=60)
    print(f"engine exited with {p.returncode}")


if __name__ == "__main__":
    main()
