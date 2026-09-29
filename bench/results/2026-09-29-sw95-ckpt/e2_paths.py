#!/usr/bin/env python3
# E2: the "cancelled" prompt of engine_smoke --reuse (N = 8217) cold, chunked vs all batches;
# first_top of each, and the KL between them
import json, math, os, subprocess, sys
engine, model, idsf = sys.argv[1:4]
ids = [int(x) for x in open(idsf).read().split()]
N = 8217
other = ids[3 * N:]
diverged = other[5000:5000 + N + 8] + other[:64]
def run(extra):
    p = subprocess.Popen([engine, model, "--ctx", "32768", "--prefill-chunk", "2048"] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, env=dict(os.environ, FLASHRT_TEST_HOOKS="1"))
    rd = lambda: json.loads(p.stdout.readline())
    rd()
    p.stdin.write(json.dumps({"op": "generate", "id": "x", "prompt": diverged, "max_new": 1, "sampling": {"temperature": 0.0, "top_k": 20, "top_p": 1.0},
                              "stop_ids": [], "debug_first_top": True}) + "\n"); p.stdin.flush()
    while True:
        ev = rd()
        if ev.get("ev") in ("done", "error"): break
    p.stdin.write('{"op":"quit"}\n'); p.stdin.flush(); p.wait(timeout=120)
    return ev
a = run([]); b = run(["--chunk-min", "1000000"])
print("chunked:", a.get("first_top", a)[:4], a.get("prompt_ms")); print("batches:", b.get("first_top", b)[:4], b.get("prompt_ms"))
def probs(p):
    m = max(l for _, l in p); e = {t: math.exp(l - m) for t, l in p}; z = sum(e.values()); return {t: v / z for t, v in e.items()}
pa, pb = probs(a["first_top"]), probs(b["first_top"])
print("KL(top 8):", sum(p * math.log(p / pb.get(t, 1e-9)) for t, p in pa.items()))
