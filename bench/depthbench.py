#!/usr/bin/env python3
"""Depth sweep against a llama-server started by this script.

  depthbench.py --label X --bin DIR --depths 1000,32000,128000,240000 -- <server args>

One growing conversation: each depth extends the previous prompt, so only the delta is
prefilled (prefix cache), and decode is timed at that depth. Prints one line per depth
and a JSON summary. Starts its own server on 127.0.0.1:8299 and kills it on exit.
"""
import argparse, json, os, random, signal, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--model", default=os.path.expanduser(
    "$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"))
ap.add_argument("--depths", default="1000,32000,128000,240000")
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--port", type=int, default=8299)
ap.add_argument("--out", default=os.path.expanduser("$BENCH"))
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--ids", help='JSON [{"depth": D, "ids": [...]}] (strata_depthbench format): send these '
                'token ids instead of the built-in filler text; --depths is then ignored')
ap.add_argument("srv", nargs=argparse.REMAINDER)
a = ap.parse_args()
srv_args = [x for x in a.srv if x != "--"]
os.makedirs(a.out, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
log = f"{a.out}/{a.label}-{stamp}.log"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = a.bin
for kv in a.env:
    k, v = kv.split("=", 1); env[k] = v

def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).split()[0])

for _ in range(60):
    if vram() < 600: break
    time.sleep(2)

cmd = [f"{a.bin}/llama-server", "-m", a.model, "--host", "127.0.0.1",
       "--port", str(a.port)] + srv_args
lf = open(log, "w")
lf.write("CMD " + " ".join(cmd) + "\nENV " + " ".join(a.env) + "\n"); lf.flush()
p = subprocess.Popen(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=50G",
                      "-p", "MemorySwapMax=0", "--", "choom", "-n", "800", "--"] + cmd,
                     stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
def kill():
    try: os.killpg(p.pid, signal.SIGKILL)
    except Exception: pass
signal.signal(signal.SIGTERM, lambda *_: (kill(), sys.exit(1)))

def post(path, body, timeout=3600):
    r = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

try:
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); break
        except Exception:
            if p.poll() is not None:
                print(f"{a.label}: server died, see {log}"); sys.exit(2)
            if time.time() - t0 > 900:
                print(f"{a.label}: not healthy after 900s"); sys.exit(2)
            time.sleep(2)
    load_s = time.time() - t0
    idle = vram()

    # continuous VRAM sampling: the prefill peak is what decides whether a config OOMs
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=subprocess.PIPE, text=True)
    import threading
    samples = []
    threading.Thread(target=lambda: [samples.append(int(l)) for l in smi.stdout if l.strip().isdigit()],
                     daemon=True).start()
    random.seed(20260926)
    seeds = [
     "The reconciliation of the logistics ledger cross-references bill-of-lading identifiers against customs declarations filed before departure.",
     "Maintenance protocols specify torque sequences, thermal tolerances, and an inspection interval tied to operating hours rather than calendar dates.",
     "Archival records indicate the survey office renumbered parcels twice, complicating any retrospective join between tax rolls and deeds.",
     "The firmware changelog describes a race in the interrupt handler that appeared only under sustained load with fragmented packets.",
     "Tidal gauges along the estuary were recalibrated after the dredging, which shifted the datum used by every later flood model.",
     "The cooperative's bylaws let members defer dues during harvest, so cash flow peaks in late autumn and troughs in spring.",
    ]
    words = []
    tail = "\n\nContinue the text above with a detailed, new paragraph about how these records are audited."
    rows = []
    peak = idle
    items = json.load(open(a.ids)) if a.ids else [{"depth": int(x)} for x in a.depths.split(",")]
    for item in items:
        if a.ids:
            prompt = item["ids"]                      # llama-server accepts a token array
        else:
            need = int(item["depth"] / 1.24)
            while len(words) < need:
                words.extend(random.choice(seeds).split())
            prompt = " ".join(words[:need]) + tail
        t = time.time()
        r = post("/completion", {"prompt": prompt, "n_predict": a.gen, "temperature": 0,
                                 "ignore_eos": True, "cache_prompt": True})
        tm = r["timings"]
        peak = max(peak, vram())
        row = {"depth": r.get("tokens_evaluated", tm.get("prompt_n")),
               "prompt_n": tm["prompt_n"], "prefill_tps": round(tm["prompt_per_second"], 1),
               "decode_tps": round(tm["predicted_per_second"], 2),
               "decode_ms": round(tm["predicted_per_token_ms"], 2), "wall_s": round(time.time() - t, 1),
               "text": r.get("content", "")}
        if "draft_n" in tm:
            row["draft_n"] = tm["draft_n"]; row["draft_acc"] = tm.get("draft_n_accepted")
        rows.append(row)
        print(f"{a.label:28s} depth {row['depth']:>7} | prefill {row['prefill_tps']:>7} t/s "
              f"({row['prompt_n']} new) | decode {row['decode_tps']:>6} t/s "
              f"{'| draft %s/%s' % (row.get('draft_acc'), row.get('draft_n')) if 'draft_n' in row else ''}"
              f" | vram_max {max(samples) if samples else 0}",
              flush=True)
    smi.terminate()
    peak = max([peak] + samples)
    summ = {"label": a.label, "load_s": round(load_s, 1), "vram_idle_mib": idle,
            "vram_peak_mib": peak, "rows": rows, "log": log}
    print("SUMMARY " + json.dumps(summ), flush=True)
finally:
    kill()
    time.sleep(3)
