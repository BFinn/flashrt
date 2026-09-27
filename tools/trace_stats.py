#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Routing-trace statistics: the numbers behind the cache and speculation design.

    trace_stats.py PREFIX            (reads PREFIX.decode_topk.npy, optional PREFIX.prefill_topk.npy)

Reports:
  skew      share of routes that go to the hottest 5/10/20% of experts, per layer (median/min/max)
  reuse     fraction of a token's experts that the previous token (same layer) also used
  window    distinct experts per layer across W consecutive tokens, vs W*k (the verify-window
            union: how many expert reads a speculative window of W tokens costs)
  prefill   distinct experts touched per layer by a chunk of C prompt tokens
"""
import os
import sys

import numpy as np


def skew(topk, n_exp):
    T, L, k = topk.shape
    out = {}
    for frac in (0.05, 0.10, 0.20):
        n = max(1, int(round(frac * n_exp)))
        shares = []
        for layer in range(L):
            c = np.bincount(topk[:, layer].ravel(), minlength=n_exp)
            shares.append(np.sort(c)[::-1][:n].sum() / c.sum())
        out[frac] = np.array(shares)
    return out


def reuse(topk):
    T, L, k = topk.shape
    s = np.zeros(L)
    for t in range(1, T):
        for layer in range(L):
            s[layer] += len(np.intersect1d(topk[t, layer], topk[t - 1, layer], assume_unique=True)) / k
    return s / (T - 1)


def window_union(topk, W):
    T, L, k = topk.shape
    n = T // W
    u = [len(np.unique(topk[i * W:(i + 1) * W, layer])) for i in range(n) for layer in range(L)]
    return np.mean(u)


def chunk_coverage(topk, C, n_exp):
    T, L, k = topk.shape
    res = []
    for s in range(0, T - C + 1, C):
        blk = topk[s:s + C]
        for layer in range(L):
            ids = blk[:, layer].ravel()
            ids = ids[ids >= 0]
            if ids.size:
                res.append(len(np.unique(ids)) / n_exp)
    return np.mean(res) if res else float("nan")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pre = sys.argv[1]
    topk = np.load(pre + ".decode_topk.npy")
    T, L, k = topk.shape
    n_exp = int(topk.max()) + 1
    n_exp = 1 << (n_exp - 1).bit_length()          # round up to the power of two the model uses
    print(f"{pre}: decode {T} tokens x {L} layers x top-{k}, n_expert~{n_exp}")
    if (topk < 0).any():
        print(f"  warning: {(topk < 0).sum()} unobserved entries")

    sk = skew(topk, n_exp)
    for frac, v in sk.items():
        print(f"  skew   hottest {frac:4.0%} of experts take {np.median(v):.1%} of routes "
              f"(layer min {v.min():.1%}, max {v.max():.1%})")
    r = reuse(topk)
    print(f"  reuse  previous token shares {r.mean():.1%} of experts (layer min {r.min():.1%}, max {r.max():.1%})")
    for W in (1, 2, 3, 4, 6, 8):
        u = window_union(topk, W)
        print(f"  window W={W}: {u:6.1f} distinct experts/layer = {u / (W * k):.2f} of W*k")

    pf = pre + ".prefill_topk.npy"
    if os.path.exists(pf):
        p = np.load(pf)
        for C in (256, 512, 1024, 2048, 4096, 8192):
            if p.shape[0] >= C:
                print(f"  prefill chunk {C:5d}: touches {chunk_coverage(p, C, n_exp):.1%} of experts per layer")


if __name__ == "__main__":
    main()
