#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarise flashrt_depthbench.py runs: per arm (the label without its -rN suffix) and depth,
mean +- sd over the runs of decode tok/s (wall clock), prompt seconds, prefill tok/s of the new
tokens, reused tokens, the decode's expert-cache hit rate and draft acceptance; GPU clock and
temperature ranges at the end.

  depthsum.py OUT [OUT ...]      (files with the SUMMARY lines)
"""
import collections
import json
import statistics as st
import sys


def fmt(xs, width=15, prec=1):
    if not xs:
        return f"{'-':>{width}}"
    m = st.mean(xs)
    s = f"{m:.{prec}f} +- {st.stdev(xs):.{prec}f}" if len(xs) > 1 else f"{m:.{prec}f}"
    return f"{s:>{width}}"


def main():
    runs = collections.defaultdict(list)
    for path in sys.argv[1:]:
        for line in open(path):
            if line.startswith("SUMMARY "):
                s = json.loads(line[8:])
                runs[s["label"].rsplit("-r", 1)[0]].append(s)
    depths = sorted({r["depth"] for ss in runs.values() for s in ss for r in s["rows"]})
    metrics = [
        ("DECODE tok/s (wall clock)", lambda r: r["decode_wall_tps"], 1),
        ("PROMPT seconds (engine prompt_ms)", lambda r: r.get("prompt_s"), 2),
        ("PREFILL tok/s of the new tokens", lambda r: r["prefill_tps"], 0),
        ("REUSED prompt tokens", lambda r: r.get("reused"), 0),
        ("EXPERT-CACHE hit rate in the decode, %", lambda r: 100 * r["hit_rate"] if r.get("hit_rate") is not None else None, 1),
        ("DRAFT acceptance, %", lambda r: 100 * r["drafts"]["accepted"] / r["drafts"]["proposed"] if (r.get("drafts") or {}).get("proposed") else None, 1),
    ]
    for title, get, prec in metrics:
        print(f"{title} (mean +- sd)")
        print(f"{'arm':28s}" + "".join(f"{d:>17}" for d in depths))
        for arm, ss in runs.items():
            cells = []
            for d in depths:
                xs = [get(r) for s in ss for r in s["rows"] if r["depth"] == d]
                cells.append(fmt([x for x in xs if x is not None], 17, prec))
            print(f"{arm + f' (n={len(ss)})':28s}" + "".join(cells))
        print()
    gpu = [r["gpu"] for ss in runs.values() for s in ss for r in s["rows"] if r.get("gpu")]
    if gpu:
        print(f"GPU at the rows' ends: SM clock {min(g['sm_mhz'] for g in gpu)}-{max(g['sm_mhz'] for g in gpu)} MHz, "
              f"{min(g['temp_c'] for g in gpu)}-{max(g['temp_c'] for g in gpu)} C, "
              f"{min(g['power_w'] for g in gpu):.0f}-{max(g['power_w'] for g in gpu):.0f} W")


if __name__ == "__main__":
    main()
