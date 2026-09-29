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
engine runs with FLASHRT_TEST_HOOKS=1). A greedy reference request A comes first; then:
  - invalid requests (a prompt longer than the context, a token id out of range, bad sampling):
    an error each, and A again reuses the previous sequence (nothing changed);
  - a fault injected after the first prefill chunk, and one in the first decode step, and a
    "stop" during the prefill: A again runs cold (reused 0) and succeeds.
Each follow-up's state after the prompt must match the reference's: the same top token at the
first generated position, and the KL divergence over the reference's top 8 there within --kl-tol.
That catches a broken state (a negative control, a different text, must fail it), not small
numeric differences. Tokens are not compared: a GPU hit and a CPU miss differ in the last bits,
so greedy tokens depend on the expert cache's content (docs/engine.md), and the reference and a
follow-up see different caches (sw92: 1-14 leading tokens shared, relative logits of the less
likely candidates up to 0.8 apart, KL at most 0.0002). Exits 1 when a check fails.

With --reuse it checks prefix reuse through the host checkpoints taken during a prefill
(engine --ckpts): each case's prompt is first run cold (after an unrelated request, so nothing is
reused) as its reference, then after the prompt it should reuse from; the second run must reuse
at least the expected prefix and match the reference's state (as for --faults):
  - tail: A with text inserted before its last 19 tokens (window 9's shape: a growing text with
    a fixed instruction at the end) reuses up to the checkpoint before A's tail (--ckpt-tail);
  - middle: A with one token changed half way reuses from a checkpoint before the change;
  - cancelled: after A, a different text stopped late in its prefill, past A's length, then
    that text's first N + 8 tokens and something else: it must reuse its own checkpoints, never
    A's end-of-prompt checkpoint (at N - 1, inside the shared prefix, but holding A's tokens).

  engine_smoke.py ENGINE MODEL --ids IDS [--n 2048] [--gen 64] [--temp 1.0] [--seed 1] [--faults | --reuse] -- [engine args]

--faults wants a small --ctx (the too-long prompt is ctx + 1 tokens) and --prefill-chunk 512 (so
the prefill has several chunks to stop between). --reuse wants --n 9000 with --prefill-chunk 2048
--ckpt-interval 2048 (no chunk end falls between N - 1 and N + 8).
"""
import os
import argparse
import json
import math
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
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--kl-tol", type=float, default=0.001)
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    extra = argv[cut + 1:]
    ids = [int(x) for x in open(a.ids).read().split()]
    env = dict(os.environ, FLASHRT_TEST_HOOKS="1") if a.faults or a.reuse else None
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
    if a.reuse:
        sys.exit(reuse(a, ids, p, read))
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


def compare_state(ref_top, ev, kl_tol):
    """(ok, detail) of a run's first generated position against a reference's first_top: the same
    top token, and the KL divergence over the reference's top 8 within kl_tol; also shown, the
    log-probabilities of the reference's top 4 relative to the top one (a shift of every logit does
    not count)."""
    top = {t: l for t, l in ev.get("first_top", [])}
    if not top or not ref_top:
        return False, "no logits"
    if ev["first_top"][0][0] != ref_top[0][0]:
        return False, f"top token {ev['first_top'][0][0]} against {ref_top[0][0]}"
    r0, f0 = ref_top[0][1], ev["first_top"][0][1]
    d = max((abs((top[t] - f0) - (l - r0)) if t in top else float("inf")) for t, l in ref_top[:4])
    shift = f0 - r0

    def probs(pairs):
        m = max(l for _, l in pairs)
        e = {t: math.exp(l - m) for t, l in pairs}
        z = sum(e.values())
        return {t: v / z for t, v in e.items()}

    pr, pf = probs(ref_top), probs(ev["first_top"])
    kl = sum(p * math.log(p / pf.get(t, 1e-9)) for t, p in pr.items())
    return kl <= kl_tol, f"KL(top 8) {kl:.5f}, relative logits within {d:.4f} (shift {shift:+.3f})"


def reuse(a, ids, p, read):
    """The --reuse sequence; returns the exit status."""
    greedy = {"temperature": 0.0, "top_k": 20, "top_p": 1.0}
    N = a.n
    A = ids[:N]
    other = ids[3 * N:]   # unrelated text
    failed = []

    def send(op):
        p.stdin.write(json.dumps(op) + "\n")
        p.stdin.flush()

    def request(rid, prompt, stop_past=None):
        """stop_past: send a stop once the prefill reports more than that many prompt tokens done."""
        send({"op": "generate", "id": rid, "prompt": prompt, "max_new": a.gen, "sampling": greedy, "stop_ids": [],
              "debug_first_top": True})
        stopped = False
        while True:
            ev = read()
            if ev.get("id") != rid:
                continue
            if ev["ev"] == "progress" and stop_past is not None and not stopped and ev.get("prompt_done", 0) > stop_past:
                send({"op": "stop", "id": rid})
                stopped = True
            elif ev["ev"] in ("done", "error"):
                if ev["ev"] == "done":
                    print(f"  {rid}: prompt {ev['prompt_tokens']}, reused {ev['reused']}, {ev['prompt_ms'] / 1000:.2f} s, finish {ev['finish']}")
                else:
                    print(f"  {rid}: error {ev['msg']}")
                return ev

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failed.append(name)

    def cold(rid, prompt):
        request(rid + "-flush", other[:300])   # shares nothing with prompt: the next run starts cold
        ev = request(rid + "-cold", prompt)
        if ev["ev"] != "done" or ev["reused"] != 0:
            check(rid + " cold reference", False, ev.get("msg") or f"reused {ev['reused']}")
        return ev.get("first_top", [])

    tail = A[:-19] + other[1000:1000 + N // 4] + A[-19:]
    mid = list(A)
    mid[N // 2] = (mid[N // 2] + 1) % 1000 + 10
    long_other = other[5000:5000 + N + N // 2]
    diverged = long_other[:N + 8] + other[:64]
    ref = {"tail": cold("tail", tail), "middle": cold("middle", mid), "cancelled": cold("cancelled", diverged)}

    for name, prompt, lo, hi in [("tail", tail, N - 1 - 64 - 1, N - 19), ("middle", mid, 1, N // 2)]:
        request(name + "-flush", other[:300])
        request(name + "-A", A)   # A cold: its prefill leaves the checkpoints
        ev = request(name, prompt)
        if ev["ev"] != "done":
            check(name, False, ev.get("msg"))
            continue
        state_ok, detail = compare_state(ref[name], ev, a.kl_tol)
        check(name, lo <= ev["reused"] <= hi and state_ok, f"reused {ev['reused']} (want {lo}..{hi}), {detail}")

    # the cancelled case: A, then another text stopped past A's length (A's checkpoints are then
    # past the shared prefix, 0, and must go), then a prompt sharing N + 8 tokens with that text
    request("cancelled-flush", other[:300])
    request("cancelled-A", A)
    ev = request("cancelled-stop", long_other, stop_past=N + 16)
    check("cancelled stopped", ev["ev"] == "done" and ev["finish"] == "cancelled", ev.get("finish") or ev.get("msg"))
    ev = request("cancelled", diverged)
    if ev["ev"] != "done":
        check("cancelled", False, ev.get("msg"))
    else:
        state_ok, detail = compare_state(ref["cancelled"], ev, a.kl_tol)
        check("cancelled", 0 < ev["reused"] <= N + 8 and ev["reused"] != N - 1 and state_ok,
              f"reused {ev['reused']} (want its own checkpoint, not A's at {N - 1}), {detail}")
    # negative control: another text against the tail reference
    ok, detail = compare_state(ref["tail"], request("control", other[:N]), a.kl_tol)
    check("negative control detected", not ok, detail)
    send({"op": "quit"})
    p.wait(timeout=60)
    print(f"engine exited with {p.returncode}; {len(failed)} check(s) failed{': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed or p.returncode != 0 else 0


def faults(a, ids, ready, p, read):
    """The --faults sequence; returns the exit status."""
    greedy = {"temperature": 0.0, "top_k": 20, "top_p": 1.0}
    A = ids[:a.n]
    failed = []

    def send(op):
        p.stdin.write(json.dumps(op) + "\n")
        p.stdin.flush()

    def request(rid, prompt, sampling=greedy, stop_on_progress=False, **extra):
        send(dict({"op": "generate", "id": rid, "prompt": prompt, "max_new": a.gen, "sampling": sampling, "stop_ids": [],
                   "debug_first_top": True}, **extra))
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
    check("reference", ev["ev"] == "done" and len(ref) > 0 and "first_top" in ev, f"{ev.get('finish') or ev.get('msg')}, {len(ref)} tokens")
    ref_top = ev.get("first_top", [])

    def same_state(ev):
        return compare_state(ref_top, ev, a.kl_tol)

    def follow_up(name, want_reused):
        ev, toks = request(name + "-next", A)
        if ev["ev"] != "done":
            return check(name, False, f"the next request failed: {ev.get('msg')}")
        same = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), min(len(toks), len(ref)))
        reuse_ok = ev["reused"] == 0 if want_reused == 0 else ev["reused"] >= want_reused
        state_ok, detail = same_state(ev)
        check(name, reuse_ok and state_ok,
              f"next request reused {ev['reused']} (want {'0' if want_reused == 0 else '>= %d' % want_reused}), "
              f"{detail}; {same} of {len(ref)} tokens shared")

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
    follow_up("stop-in-prefill", 0)
    # negative control: another text must fail the state check
    ev, _ = request("control", ids[4 * a.n:5 * a.n])
    ok, detail = same_state(ev)
    check("negative control detected", not ok, detail)
    print(f"info reference top 8: {ref_top}")
    send({"op": "quit"})
    p.wait(timeout=60)
    print(f"engine exited with {p.returncode}; {len(failed)} check(s) failed{': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed or p.returncode != 0 else 0


if __name__ == "__main__":
    main()
