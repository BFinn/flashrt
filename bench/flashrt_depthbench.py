#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Depth sweep of flashrt-engine with the protocol of the reference runs (window 9:
bench/results/2026-09-27-w9-validation, strata_depthbench.py and depthbench.py).

  flashrt_depthbench.py --engine BIN --model GGUF --ids ids.json --label X [--gen 384]
                        [--sampling "temperature=1.0 top_p=0.95 top_k=20"] -- [engine args]

One growing conversation: each depth's prompt (the token ids of strata-ids.json, the same ids
the reference engines got) extends the previous one, so only the new tokens are prefilled
(prefix reuse). Each depth generates --gen tokens without stopping at end-of-sequence (as
ignore_eos in the reference runs). The engine is started fresh for each run, in a systemd scope
capped at 56 GB where systemd-run exists (--no-scope: a plain child process).

The decode rate is timed two ways, as in strata_depthbench.py: wall clock between the first
and the last streamed token, and the engine's own done event (generated tokens / decode_ms).
The prefill rate is the new tokens over prompt_ms. Prints one line per depth, then a JSON
summary line.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time


def gpu_state():
    """SM clock (MHz), temperature (C) and power (W) now, for the row's record."""
    try:
        f = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], text=True).split(",")
        return {"sm_mhz": int(f[0]), "temp_c": int(f[1]), "power_w": float(f[2])}
    except Exception:
        return None


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help='JSON [{"depth": D, "ids": [...]}]')
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--sampling", default="", help='e.g. "temperature=1.0 top_p=0.95 top_k=20" (default greedy)')
    ap.add_argument("--log", default="", help="engine stderr log (default: LABEL.log)")
    ap.add_argument("--no-scope", action="store_true", help="run the engine without a systemd scope")
    a = ap.parse_args(argv[:cut])
    extra = argv[cut + 1:]

    sampling = {"temperature": 0.0, "top_k": 20, "top_p": 1.0}
    for kv in a.sampling.split():
        k, v = kv.split("=", 1)
        sampling[k] = int(v) if k == "top_k" else float(v)
    for _ in range(150):   # the previous loader's VRAM can take a few seconds to come back
        if vram() < 600:
            break
        time.sleep(2)
    log = open(a.log or f"{a.label}.log", "w")
    cmd = [a.engine, a.model] + extra
    log.write("CMD " + " ".join(cmd) + "\n")
    log.flush()
    t0 = time.time()
    scope = [] if a.no_scope or not shutil.which("systemd-run") else \
        ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=56G", "-p", "MemorySwapMax=0", "--"]
    p = subprocess.Popen(scope + cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

    def read():
        line = p.stdout.readline()
        if not line:
            print(f"{a.label}: engine exited (see {log.name})", flush=True)
            sys.exit(2)
        return json.loads(line)

    ready = read()
    if ready.get("ev") != "ready":
        print(f"{a.label}: {ready}", flush=True)
        sys.exit(2)
    load_s = time.time() - t0
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                samples.append(vram())
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()
    rows = []
    try:
        for n, item in enumerate(json.load(open(a.ids))):
            ids = item["ids"]
            sp = dict(sampling, seed=random.randrange(1, 2**31)) if sampling["temperature"] > 0 else dict(sampling)
            rid = f"d{n}"
            t_send = time.time()
            p.stdin.write(json.dumps({"op": "generate", "id": rid, "prompt": ids, "max_new": a.gen, "sampling": sp, "stop_ids": []}) + "\n")
            p.stdin.flush()
            times, done = [], None
            while True:
                ev = read()
                if ev.get("id") not in (rid, None, ""):
                    continue
                if ev.get("ev") == "token":
                    times.append(time.time())
                elif ev.get("ev") == "done":
                    done = ev
                    break
                elif ev.get("ev") == "error":
                    print(f"{a.label}: error {ev.get('msg')}", flush=True)
                    sys.exit(3)
            k = len(times)
            wall_tps = (k - 1) / (times[-1] - times[0]) if k > 1 else 0.0
            eng_tps = done["generated"] / done["decode_ms"] * 1000 if done.get("decode_ms") else 0.0
            new = done["prompt_tokens"] - done.get("reused", 0)
            pre_tps = new / done["prompt_ms"] * 1000 if done.get("prompt_ms") else 0.0
            dr = done.get("drafts") or {}
            ca = done.get("cache") or {}
            hit = ca["hits"] / max(1, ca["hits"] + ca["misses"]) if "hits" in ca else None
            row = {"depth": len(ids), "new": new, "reused": done.get("reused", 0), "generated": k,
                   "ttft_s": round(times[0] - t_send, 2) if times else None, "prompt_s": round(done.get("prompt_ms", 0) / 1000, 2),
                   "decode_wall_tps": round(wall_tps, 2), "decode_engine_tps": round(eng_tps, 2), "prefill_tps": round(pre_tps, 1),
                   "drafts": dr, "cache": ca, "hit_rate": round(hit, 4) if hit is not None else None, "finish": done.get("finish"),
                   "gpu": gpu_state()}
            rows.append(row)
            acc = f" | drafts {dr.get('accepted')}/{dr.get('proposed')}" if dr.get("proposed") else ""
            acc += f" | hits {100 * hit:.1f}%" if hit is not None else ""
            print(f"{a.label:28s} depth {row['depth']:>7} | prefill {row['prefill_tps']:>7} t/s ({new} new, {row['prompt_s']} s) | decode "
                  f"{row['decode_wall_tps']:>6} t/s wall, {row['decode_engine_tps']:>6} engine | gen {k} {row['finish']}{acc}"
                  f" | vram_max {max(samples) if samples else 0}", flush=True)
        print("SUMMARY " + json.dumps({"label": a.label, "load_s": round(load_s, 1), "sampling": sampling, "engine_args": extra,
                                       "vram_peak_mib": max(samples) if samples else 0, "rows": rows}), flush=True)
    finally:
        stop.set()
        try:
            p.stdin.close()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        time.sleep(3)


if __name__ == "__main__":
    main()
