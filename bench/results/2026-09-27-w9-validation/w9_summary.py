#!/usr/bin/env python3
"""Aggregate window 9 (background validation) into a mean +- sd table per arm and depth.

Decode: llama.cpp rows report decode_tps, Strata rows decode_wall_tps.
Prefill: llama.cpp prefill_tps (new tokens only); Strata new tokens / time to first token.
"""
import json, statistics as st, sys, collections

rows = collections.defaultdict(lambda: collections.defaultdict(lambda: {"dec": [], "pre": [], "acc": []}))
for line in open(sys.argv[1] if len(sys.argv) > 1 else "w9.out"):
    if not line.startswith("SUMMARY "):
        continue
    s = json.loads(line[8:])
    arm = s["label"].rsplit("-r", 1)[0]
    prev = 0
    for r in s["rows"]:
        d = r["depth"]
        c = rows[arm][d]
        if "decode_wall_tps" in r:                      # Strata
            if r.get("generated", 0) >= 100:             # a sampled EOS can end a run early: no decode rate
                c["dec"].append(r["decode_wall_tps"])
            else:
                c["short"] = c.get("short", 0) + 1
            if r.get("ttft_s"):
                c["pre"].append((d - prev) / r["ttft_s"])
        else:                                          # llama.cpp depthbench
            c["dec"].append(r["decode_tps"])
            c["pre"].append(r["prefill_tps"])
        prev = d


def fmt(xs):
    if not xs:
        return "-"
    m = st.mean(xs)
    return f"{m:6.1f} +-{st.stdev(xs):4.1f}" if len(xs) > 1 else f"{m:6.1f}      "


depths = sorted({d for a in rows.values() for d in a})
print("DECODE tok/s (mean +- sd, n runs)")
print(f"{'arm':26s}" + "".join(f"{d:>16}" for d in depths))
for arm, a in rows.items():
    n = max(len(a[d]["dec"]) for d in a)
    print(f"{arm + f' (n={n})':26s}" + "".join(f"{fmt(a[d]['dec']) if d in a else '-':>16}" for d in depths))
    short = {d: a[d].get("short", 0) for d in a if a[d].get("short")}
    if short:
        print(f"{'':26s}  (runs ended early by a sampled EOS, excluded: {short})")
print("\nPREFILL tok/s of the new tokens (mean +- sd)")
print(f"{'arm':26s}" + "".join(f"{d:>16}" for d in depths))
for arm, a in rows.items():
    print(f"{arm:26s}" + "".join(f"{fmt(a[d]['pre']) if d in a else '-':>16}" for d in depths))
