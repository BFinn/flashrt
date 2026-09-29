#!/usr/bin/env python3
# The first generated position's top 8 logits (engine test hook) of one prompt: the first
# `--n` ids of an ids file, optionally with other ids inserted before its last 19 (sw95e's
# segmentation control). Prints one JSON line.
import argparse, json, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("engine"); ap.add_argument("model"); ap.add_argument("--ids", required=True)
ap.add_argument("--n", type=int, required=True); ap.add_argument("--insert", type=int, default=0)
argv = sys.argv[1:]; cut = argv.index("--") if "--" in argv else len(argv)
a = ap.parse_args(argv[:cut]); extra = argv[cut + 1:]
ids = [int(x) for x in open(a.ids).read().split()]
A = ids[:a.n]
prompt = A[:-19] + ids[3 * a.n + 1000:3 * a.n + 1000 + a.insert] + A[-19:] if a.insert else A
p = subprocess.Popen([a.engine, a.model] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                     env=dict(os.environ, FLASHRT_TEST_HOOKS="1"))
def read():
    return json.loads(p.stdout.readline())
read()
p.stdin.write(json.dumps({"op": "generate", "id": "x", "prompt": prompt, "max_new": 1, "sampling": {"temperature": 0.0, "top_k": 20, "top_p": 1.0},
                          "stop_ids": [], "debug_first_top": True}) + "\n"); p.stdin.flush()
while True:
    ev = read()
    if ev.get("ev") in ("done", "error"):
        print(json.dumps({"n": len(prompt), "args": extra, "first_top": ev.get("first_top"), "msg": ev.get("msg")}))
        break
p.stdin.write(json.dumps({"op": "quit"}) + "\n"); p.stdin.flush(); p.wait(timeout=60)
