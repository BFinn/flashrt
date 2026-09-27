#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""QSA selection locality: would a small VRAM hot set of KV blocks serve decode?

    qsa_stats.py PREFIX [--block 4]      (reads PREFIX.decode_qsa.npy and PREFIX.meta.json)

The trace holds, per decoded token and attention layer, the KV cells the indexer selected.
Cells are grouped into blocks of --block cells (QSA selects whole blocks). Per layer:
  overlap   fraction of this token's blocks that the previous token also selected
  lru B     miss rate of a per-layer LRU hot set of B blocks, for B = 1x, 2x, 4x the
            selection size, counting only blocks that existed before decoding began
            (blocks the decode itself writes are resident by construction)
  recent    share of selected blocks among the newest 1/8 of the context
"""
import argparse
import json
from collections import OrderedDict

import numpy as np


def lru_miss(rows, cap):
    cache, miss, total = OrderedDict(), 0, 0
    for blocks in rows:
        for b in blocks:
            total += 1
            if b in cache:
                cache.move_to_end(b)
            else:
                miss += 1
                cache[b] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return miss / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("--block", type=int, default=4)
    a = ap.parse_args()
    q = np.load(a.prefix + ".decode_qsa.npy")               # [G, n_attn, W]
    meta = json.load(open(a.prefix + ".meta.json"))
    n_prompt = meta["n_prompt"]
    G, A, W = q.shape
    print(f"{a.prefix}: {G} decoded tokens x {A} QSA layers, selection width {W} cells, context {n_prompt}+")
    first_new_block = n_prompt // a.block
    print(f"{'layer':>6} {'blocks':>6} {'overlap':>8} {'lru 1x':>7} {'lru 2x':>7} {'lru 4x':>7} {'recent':>7}")
    agg = []
    for j, layer in enumerate(meta.get("qsa_layers", list(range(A)))):
        rows = []
        for g in range(G):
            cells = q[g, j]
            blocks = np.unique(cells[cells >= 0] // a.block)
            rows.append(blocks)
        nsel = int(np.median([len(r) for r in rows]))
        ov = np.mean([len(np.intersect1d(rows[g], rows[g - 1], assume_unique=True)) / max(len(rows[g]), 1)
                      for g in range(1, G)])
        old = [r[r < first_new_block].tolist() for r in rows]
        m = [lru_miss(old, k * nsel) for k in (1, 2, 4)]
        recent = np.mean([(r >= first_new_block - first_new_block // 8).mean() for r in rows])
        agg.append((ov, *m, recent))
        print(f"{layer:>6} {nsel:>6} {ov:8.1%} {m[0]:7.1%} {m[1]:7.1%} {m[2]:7.1%} {recent:7.1%}")
    agg = np.array(agg).mean(axis=0)
    print(f"{'mean':>6} {'':>6} {agg[0]:8.1%} {agg[1]:7.1%} {agg[2]:7.1%} {agg[3]:7.1%} {agg[4]:7.1%}")


if __name__ == "__main__":
    main()
