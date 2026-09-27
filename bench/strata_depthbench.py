#!/usr/bin/env python3
"""Depth sweep for the Strata engine with the exact prompts of depthbench.py.

  strata_depthbench.py tokenize --url http://HOST:PORT --api-key-file F --out ids.json
      build depthbench's prompts and tokenize them with a running llama-server (same token ids)
  strata_depthbench.py run --cfg strata-q2_0.json --ids ids.json --label X [--no-mtp] [--gen 384]
      start `strata --serve` with the setup's arguments and time each depth

One growing conversation, as in depthbench.py: each depth extends the previous prompt.
Decode rate is timed two ways: wall clock between the first and last streamed token, and the
engine's own DONE line (generated tokens / decode_ms).
"""
import argparse, json, os, random, subprocess, sys, threading, time, urllib.request

DEPTHS = [1000, 32000, 131000, 245000]
SEEDS = [
 "The reconciliation of the logistics ledger cross-references bill-of-lading identifiers against customs declarations filed before departure.",
 "Maintenance protocols specify torque sequences, thermal tolerances, and an inspection interval tied to operating hours rather than calendar dates.",
 "Archival records indicate the survey office renumbered parcels twice, complicating any retrospective join between tax rolls and deeds.",
 "The firmware changelog describes a race in the interrupt handler that appeared only under sustained load with fragmented packets.",
 "Tidal gauges along the estuary were recalibrated after the dredging, which shifted the datum used by every later flood model.",
 "The cooperative's bylaws let members defer dues during harvest, so cash flow peaks in late autumn and troughs in spring.",
]
TAIL = "\n\nContinue the text above with a detailed, new paragraph about how these records are audited."


def prompts(depths):
    # identical to depthbench.py: same seed, same word stream, same tail
    random.seed(20260926)
    words, out = [], []
    for d in depths:
        need = int(d / 1.24)
        while len(words) < need:
            words.extend(random.choice(SEEDS).split())
        out.append(" ".join(words[:need]) + TAIL)
    return out


def tokenize(a):
    key = open(os.path.expanduser(a.api_key_file)).read().strip() if a.api_key_file else ""
    res = []
    for d, p in zip(DEPTHS, prompts(DEPTHS)):
        r = urllib.request.Request(a.url + "/tokenize", data=json.dumps({"content": p, "add_special": True}).encode(),
                                   headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        ids = json.load(urllib.request.urlopen(r, timeout=600))["tokens"]
        res.append({"depth": d, "ids": ids})
        print(f"depth {d}: {len(ids)} tokens", flush=True)
    json.dump(res, open(a.out, "w"))


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).split()[0])


def run(a):
    cfg = json.load(open(a.cfg))
    args = list(cfg["args"])
    def setarg(flag, val):
        if flag in args:
            args[args.index(flag) + 1] = val
        else:
            args.extend([flag, val])
    if a.no_mtp:
        # serve mode requires --mtp and --spec >= 2; no draft reaches p >= 1.1, so every window is
        # one token (the draft step still runs: a slight underestimate of plain decode)
        setarg("--spec", "2")
        setarg("--spec-min-p", "1.1")
    if a.vram_reserve_mib:
        setarg("--vram-reserve-mib", str(a.vram_reserve_mib))
    for kv in a.set:
        if "=" not in kv:              # a bare switch such as --stats
            if kv not in args:
                args.append(kv)
            continue
        k, v = kv.split("=", 1)
        setarg(k, v)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs") or []) + ":" + env.get("LD_LIBRARY_PATH", "")
    # the previous loader's VRAM can take a few seconds to come back
    for _ in range(90):
        if vram() < 600:
            break
        time.sleep(2)
    log = open(f"{a.out_dir}/{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.log", "w")
    log.write("CMD " + " ".join([cfg["exe"], "--serve", *args]) + "\n"); log.flush()
    t0 = time.time()
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=56G", "-p", "MemorySwapMax=0",
                          "--", cfg["exe"], "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
    ready = None
    for line in p.stdout:
        if line.startswith("READY"):
            ready = line.strip(); break
    if not ready:
        print(f"{a.label}: engine exited before READY (see {log.name})"); sys.exit(2)
    load_s = time.time() - t0
    samples = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            try: samples.append(vram())
            except Exception: pass
            time.sleep(0.5)
    threading.Thread(target=sampler, daemon=True).start()
    rows = []
    try:
        items = json.load(open(a.ids))
        for item in items[:a.n_depths] if a.n_depths else items:
            ids = item["ids"]
            t_send = time.time()
            samp = f"{a.sampling} seed={random.randrange(1, 2**31)} " if a.sampling else ""
            p.stdin.write(f"GEN {a.gen} {samp}{','.join(str(t) for t in ids)}\n"); p.stdin.flush()
            times, done, out_ids = [], None, []
            for line in p.stdout:
                if line.startswith("T "):
                    times.append(time.time())
                    out_ids.append(int(line[2:]))
                elif line.startswith("DONE"):
                    done = line.split(); break
                elif line.startswith("ERR"):
                    print(f"{a.label}: ERR {line.strip()}"); sys.exit(3)
            n = len(times)
            wall_tps = (n - 1) / (times[-1] - times[0]) if n > 1 else 0.0
            eng = {"generated": int(done[1]), "prompt_tokens": int(done[2]), "prompt_ms": float(done[3]),
                   "decode_ms": float(done[4]), "finish": done[5]} if done else {}
            eng_tps = eng["generated"] / eng["decode_ms"] * 1000 if eng.get("decode_ms") else 0.0
            pre_tps = eng["prompt_tokens"] / eng["prompt_ms"] * 1000 if eng.get("prompt_ms") else 0.0
            row = {"depth": len(ids), "generated": n, "ttft_s": round(times[0] - t_send, 2) if times else None,
                   "decode_wall_tps": round(wall_tps, 2), "decode_engine_tps": round(eng_tps, 2),
                   "prefill_engine_tps": round(pre_tps, 1), "engine": eng, "out_ids": out_ids}
            rows.append(row)
            print(f"{a.label:28s} depth {row['depth']:>7} | prefill {row['prefill_engine_tps']:>7} t/s "
                  f"({eng.get('prompt_tokens')} new) | decode {row['decode_wall_tps']:>6} t/s wall, "
                  f"{row['decode_engine_tps']:>6} engine | gen {n} {eng.get('finish')} | vram_max {max(samples) if samples else 0}",
                  flush=True)
        print("SUMMARY " + json.dumps({"label": a.label, "load_s": round(load_s, 1), "ready": ready,
                                       "vram_peak_mib": max(samples) if samples else 0, "rows": rows}), flush=True)
    finally:
        stop.set()
        try:
            p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=20)
        except Exception:
            p.kill()
        time.sleep(3)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("tokenize"); t.add_argument("--url", required=True); t.add_argument("--api-key-file"); t.add_argument("--out", required=True)
r = sub.add_parser("run"); r.add_argument("--cfg", required=True); r.add_argument("--ids", required=True)
r.add_argument("--label", required=True); r.add_argument("--no-mtp", action="store_true"); r.add_argument("--gen", type=int, default=384)
r.add_argument("--out-dir", default=os.path.expanduser("$BENCH"))
r.add_argument("--vram-reserve-mib", type=int, default=0)
r.add_argument("--set", action="append", default=[], help="--flag=value to set or override, or a bare --switch to add")
r.add_argument("--n-depths", type=int, default=0, help="only the first N depths")
r.add_argument("--sampling", default="", help='e.g. "temperature=1.0 top_p=0.95 top_k=20"')
a = ap.parse_args()
tokenize(a) if a.cmd == "tokenize" else run(a)
