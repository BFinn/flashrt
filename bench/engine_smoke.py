#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test of flashrt-engine's JSON-lines protocol (docs/design.md), with token ids.

Runs four requests against one engine process and prints each one's events summary:
  1. a prompt of N tokens from an ids file;
  2. that prompt plus the tokens request 1 generated plus a few more prompt tokens (the engine
     should reuse the whole previous sequence: "reused" close to the old length);
  3. request 2's prompt plus different tokens than request 2 generated (the engine should restore
     the checkpoint taken at the end of request 2's prompt: "reused" = that prompt's length);
  4. request 1's prompt with a "stop" sent after a few tokens (finish "cancelled"; no reuse, as
     the checkpoint is at the end of request 3's prompt).

With --faults it instead checks that bad and failing requests leave the engine serving (the
engine runs with FLASHRT_FAULT_INJECTION=1). A greedy reference request A comes first; then:
  - invalid requests (a prompt longer than the context, a token id out of range, bad sampling):
    an error each, and A again reuses the previous sequence (nothing changed);
  - a fault injected after the first prefill chunk, and one in the first decode step, and a
    "stop" during the prefill: A again runs cold (reused 0) and succeeds.
Each follow-up's first 8 greedy tokens must match the reference (later ones may differ: the
expert cache's content decides GPU or CPU arithmetic). Exits 1 when a check fails.

  engine_smoke.py ENGINE MODEL --ids IDS [--n 2048] [--gen 64] [--temp 1.0] [--seed 1] [--faults] -- [engine args]

--faults wants a small --ctx (the too-long prompt is ctx + 1 tokens) and --prefill-chunk 512 (so
the prefill has several chunks to stop between).
"""
import os
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
    ap.add_argument("--faults", action="store_true")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    extra = argv[cut + 1:]
    ids = [int(x) for x in open(a.ids).read().split()]
    env = dict(os.environ, FLASHRT_FAULT_INJECTION="1") if a.faults else None
    p = subprocess.Popen([a.engine, a.model] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)

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

    if a.faults:
        sys.exit(faults(a, ids, ready, p, read))
    prompt1 = ids[:a.n]
    t1 = run("r1", prompt1)
    prompt2 = prompt1 + t1 + ids[a.n:a.n + 32]
    run("r2", prompt2)
    run("r3", prompt2 + ids[a.n + 32:a.n + 48])
    run("r4", prompt1, stop_after=5)
    p.stdin.write(json.dumps({"op": "quit"}) + "\n")
    p.stdin.flush()
    p.wait(timeout=60)
    print(f"engine exited with {p.returncode}")


def faults(a, ids, ready, p, read):
    """The --faults sequence; returns the exit status."""
    greedy = {"temperature": 0.0, "top_k": 20, "top_p": 1.0}
    A = ids[:a.n]
    failed = []

    def send(op):
        p.stdin.write(json.dumps(op) + "\n")
        p.stdin.flush()

    def request(rid, prompt, sampling=greedy, stop_on_progress=False, **extra):
        send(dict({"op": "generate", "id": rid, "prompt": prompt, "max_new": a.gen, "sampling": sampling, "stop_ids": []}, **extra))
        toks, stopped = [], False
        while True:
            ev = read()
            if ev.get("id") != rid:
                continue
            if ev["ev"] == "progress" and stop_on_progress and not stopped:
                send({"op": "stop", "id": rid})
                stopped = True
            elif ev["ev"] == "token":
                toks.append(ev["tok"])
            elif ev["ev"] in ("done", "error"):
                return ev, toks

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failed.append(name)

    ev, ref = request("ref", A)
    check("reference", ev["ev"] == "done" and len(ref) > 0, f"{ev.get('finish') or ev.get('msg')}, {len(ref)} tokens")

    def follow_up(name, want_reused):
        ev, toks = request(name + "-next", A)
        if ev["ev"] != "done":
            return check(name, False, f"the next request failed: {ev.get('msg')}")
        same = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), min(len(toks), len(ref)))
        reuse_ok = ev["reused"] == 0 if want_reused == 0 else ev["reused"] >= want_reused
        check(name, reuse_ok and same >= min(8, len(ref)),
              f"next request reused {ev['reused']} (want {'0' if want_reused == 0 else '>= %d' % want_reused}), "
              f"first {same} of {len(ref)} tokens as the reference")

    n_ctx = ready["max_context"]
    for name, prompt, sampling in [
        ("too-long", ids[:1] * (n_ctx + 1), greedy),
        ("bad-token", A[:100] + [10**9] + A[100:], greedy),
        ("bad-top-k", A, dict(greedy, top_k=65)),
        ("bad-temperature", A, dict(greedy, temperature=-1.0)),
        ("bad-top-p", A, dict(greedy, top_p=0.0)),
    ]:
        ev, _ = request(name, prompt, sampling)
        check(name + " rejected", ev["ev"] == "error", ev.get("msg", ev.get("finish")))
        follow_up(name, len(A) - 1)   # nothing changed: A's checkpoint is still there
    for name, extra in [("fault-prefill", {"debug_fail": 1}), ("fault-decode", {"debug_fail": 2})]:
        ev, _ = request(name, A[:-7] + ids[a.n:a.n + 7], **extra)   # shares less than A's checkpoint: a full prefill
        check(name + " reported", ev["ev"] == "error" and "injected" in ev.get("msg", ""), ev.get("msg", ev.get("finish")))
        follow_up(name, 0)
    ev, _ = request("stop-in-prefill", ids[a.n:2 * a.n], stop_on_progress=True)
    check("stop-in-prefill cancelled", ev["ev"] == "done" and ev["finish"] == "cancelled", ev.get("finish") or ev.get("msg"))
    ev, toks = request("stop-in-prefill-next", A)
    check("stop-in-prefill", ev["ev"] == "done" and toks[:8] == ref[:8], f"{ev.get('finish') or ev.get('msg')}, reused {ev.get('reused')}")
    send({"op": "quit"})
    p.wait(timeout=60)
    print(f"engine exited with {p.returncode}; {len(failed)} check(s) failed{': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed or p.returncode != 0 else 0


if __name__ == "__main__":
    main()
