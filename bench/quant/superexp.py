#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Super-expert profile from a ref_dump capture of ffn_moe_down and ffn_moe_topk.

"Super experts" (arXiv 2507.23279) produce rare, huge down_proj outputs, and pruning or
crushing them collapses the model. For each (layer, expert) this records the largest
|down_proj output| over the prompt, then flags experts far above their layer's typical
maximum. The allocator pins flagged experts at the high tier and never prunes them.

  superexp.py CAPTURE.frd [--top 64] > superexperts.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.expandvars("$FLASHRT/tools"))
from frd import records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frd")
    ap.add_argument("--top", type=int, default=64)
    a = ap.parse_args()
    down, topk = {}, {}
    for name, step, _typ, _ne, arr in records(a.frd):
        if step != -1:
            continue
        base, _, layer = name.rpartition("-")
        if base == "ffn_moe_down":
            # ggml ne [n_embd, n_used, n_tok] -> numpy (1, n_tok, n_used, n_embd)
            down[int(layer)] = np.abs(arr.reshape(arr.shape[-3:])).max(axis=-1)  # [n_tok, n_used]
        elif base == "ffn_moe_topk":
            topk[int(layer)] = arr.reshape(arr.shape[-2:])                       # [n_tok, n_used]
    rows = []
    per_layer = {}
    for layer in sorted(down):
        m, ids = down[layer], topk[layer]
        best = defaultdict(float)
        for e, v in zip(ids.ravel().tolist(), m.ravel().tolist()):
            if v > best[e]:
                best[e] = v
        vals = np.array(list(best.values()))
        med = float(np.median(vals))
        per_layer[layer] = {"experts_seen": len(best), "median_max": med, "p99_max": float(np.percentile(vals, 99))}
        for e, v in best.items():
            rows.append({"layer": layer, "expert": e, "max_abs": v, "x_layer_median": v / med if med else 0.0})
    rows.sort(key=lambda r: -r["x_layer_median"])
    json.dump({"tokens": int(next(iter(down.values())).shape[0]) if down else 0,
               "per_layer": per_layer, "top": rows[: a.top]}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
