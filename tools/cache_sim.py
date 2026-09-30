#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expert-cache simulator: hit rate vs cache size over a routing trace.

    cache_sim.py --trace routing.npy --slots 2000,4000,6000 [--policies lru,dlfu,belady]
    cache_sim.py --synthetic --tokens 2000 --slots 2000,4000,6000

Trace format: a .npy integer array of shape [n_tokens, n_layers, k] holding the routed
expert ids per token and layer (decode order). Cache keys are (layer, expert) pairs and the
capacity is global, in slots of one expert each.

Policies:
  lru     evict the least recently used pair
  dlfu    decayed LFU with hysteresis: every access adds 1; every `--decay-every` tokens all
          counts are multiplied by `--decay`; a missing pair is admitted only if its count
          >= `--admit` and it beats the weakest resident by `--margin`
  win     LRU with windowed admission: a missing pair is admitted only after `--win-admit`
          sightings within the last `--win` tokens (the llama.cpp expert-cache policy)
  static  the `slots` most-used pairs of the whole trace, never changed (an oracle profile:
          the best any fixed pre-fill can do)
  belady  evict the resident pair used furthest in the future (the optimum; needs the trace)
  engine  flashrt-engine's policy: primed from the prompt's routing (--prime, a prefill trace:
          the `slots` pairs of highest count, counts halved every --prime-half-life tokens from
          the prompt's start, as the engine's prefill counts are), those counts seeding the
          decayed LFU above; at most --budget uploads started per token, each resident from
          the next token on. With --windows W, also the hit rate of each W-token window.

Hits are counted per access. The hit rate is what one gets from the cache; misses are what
the CPU or PCIe has to serve.
"""
import argparse
import heapq
import sys
from collections import OrderedDict

import numpy as np


def keys_of(trace):
    n_tok, n_layer, k = trace.shape
    layer = np.arange(n_layer, dtype=np.int64)[None, :, None]
    return (layer * 100_000 + trace.astype(np.int64)).reshape(n_tok, n_layer * k)


def sim_lru(keys, cap):
    cache, hits, total = OrderedDict(), 0, 0
    for row in keys:
        for key in row.tolist():
            total += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return hits / total


def sim_dlfu(keys, cap, decay=0.7, decay_every=4, admit=2.0, margin=1.5):
    count, resident, hits, total = {}, set(), 0, 0
    for t, row in enumerate(keys):
        for key in row.tolist():
            total += 1
            count[key] = count.get(key, 0.0) + 1.0
            if key in resident:
                hits += 1
            elif len(resident) < cap:
                resident.add(key)
            elif count[key] >= admit:
                victim = min(resident, key=lambda r: count.get(r, 0.0)) if len(resident) < 4096 else None
                if victim is None:                      # large caches: sample victims instead of a full scan
                    sample = [resident.pop() for _ in range(64)]
                    resident.update(sample)
                    victim = min(sample, key=lambda r: count.get(r, 0.0))
                if count[key] >= margin * count.get(victim, 0.0):
                    resident.discard(victim)
                    resident.add(key)
        if (t + 1) % decay_every == 0:
            for key in list(count):
                count[key] *= decay
                if count[key] < 1e-3 and key not in resident:
                    del count[key]
    return hits / total


def sim_win(keys, cap, window=32, need=3):
    from collections import deque
    cache, hits, total = OrderedDict(), 0, 0
    seen, recent = {}, deque()                 # key -> sightings in the window; per-token key lists
    for row in keys:
        row = row.tolist()
        recent.append(row)
        for key in row:
            seen[key] = seen.get(key, 0) + 1
        if len(recent) > window:
            for key in recent.popleft():
                seen[key] -= 1
        for key in row:
            total += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            elif len(cache) < cap or seen[key] >= need:
                cache[key] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return hits / total


def sim_static(keys, cap):
    uniq, counts = np.unique(keys, return_counts=True)
    return np.sort(counts)[::-1][:cap].sum() / keys.size


def sim_belady(keys, cap):
    flat = keys.reshape(-1).tolist()
    n = len(flat)
    nxt, last = [n] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(flat[i], n)
        last[flat[i]] = i
    resident, heap, hits = {}, [], 0          # key -> its next use; max-heap of (-next_use, key)
    for i, key in enumerate(flat):
        if key in resident:
            hits += 1
        elif len(resident) >= cap:
            while True:                        # drop stale heap entries until a live one surfaces
                neg, victim = heapq.heappop(heap)
                if resident.get(victim) == -neg:
                    del resident[victim]
                    break
        resident[key] = nxt[i]
        heapq.heappush(heap, (-nxt[i], key))
    return hits / n


def prime_counts(prefill, n_exp, half_life):
    """The engine's prefill routing counts over (layer, expert): halved every half_life tokens."""
    n_tok, n_layer, k = prefill.shape
    c = np.zeros(n_layer * n_exp)
    base = np.arange(n_layer)[:, None] * n_exp
    for t in range(n_tok):
        if half_life and t and t % half_life == 0:
            c *= 0.5
        row = prefill[t]
        ok = row >= 0
        np.add.at(c, (base + row)[ok], 1.0)
    return c


def sim_engine(trace, cap, n_exp, prime=None, budget=32, decay=0.7, decay_every=4, admit=2.0, margin=1.5, windows=0, seed_scale=1.0,
               early_budget=0, early_tokens=0):
    """Returns (hit rate, per-window hit rates, uploads started)."""
    n_tok, n_layer, k = trace.shape
    keys = (np.arange(n_layer)[None, :, None] * n_exp + trace).reshape(n_tok, n_layer * k)
    count = np.zeros(n_layer * n_exp) if prime is None else prime.astype(float).copy()
    resident = np.zeros(n_layer * n_exp, dtype=bool)
    if prime is not None:
        resident[np.argsort(-count, kind="stable")[:cap]] = True
        count *= seed_scale   # the fill follows the prompt; the policy's counts start at this fraction
    pending = []
    hits, total, win, uploads = 0, 0, [], 0
    wh = wt = 0
    for t in range(n_tok):
        resident[pending] = True   # last token's uploads have landed
        uploads += len(pending)
        pending = []
        row = keys[t]
        h = int(resident[row].sum())
        hits += h
        total += row.size
        wh += h
        wt += row.size
        np.add.at(count, row, 1.0)
        n_res = int(resident.sum())
        missed = [key for key in dict.fromkeys(row.tolist()) if not resident[key]]
        missed.sort(key=lambda key: -count[key])
        bud = early_budget if t < early_tokens else budget
        for key in missed:
            if len(pending) >= bud:
                break
            if n_res + len(pending) < cap:
                pending.append(key)
                continue
            if count[key] < admit:
                continue
            res_idx = np.flatnonzero(resident)
            victim = res_idx[np.argmin(count[res_idx])]
            if count[key] >= margin * count[victim]:
                resident[victim] = False
                pending.append(key)
        if (t + 1) % decay_every == 0:
            count *= decay
        if windows and (t + 1) % windows == 0:
            win.append(wh / wt)
            wh = wt = 0
    return hits / total, win, uploads


def synthetic(n_tok, n_layer=48, n_exp=512, k=10, seed=1):
    """Zipf-skewed popularity plus temporal locality: a crude stand-in until a real trace exists."""
    rng = np.random.default_rng(seed)
    pop = 1.0 / np.arange(1, n_exp + 1) ** 0.6
    out = np.empty((n_tok, n_layer, k), dtype=np.int32)
    for layer in range(n_layer):
        perm = rng.permutation(n_exp)
        p = pop[np.argsort(perm)] / pop.sum()
        prev = rng.choice(n_exp, k, replace=False, p=p)
        for t in range(n_tok):
            keep = rng.random(k) < 0.45                 # locality: some experts repeat token to token
            fresh = rng.choice(n_exp, k, replace=False, p=p)
            row = np.where(keep, prev, fresh)
            _, idx = np.unique(row, return_index=True)  # de-duplicate, refill from fresh draws
            row = list(row[np.sort(idx)])
            for e in fresh:
                if len(row) == k:
                    break
                if e not in row:
                    row.append(e)
            out[t, layer] = row[:k]
            prev = out[t, layer]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tokens", type=int, default=1000)
    ap.add_argument("--slots", default="2304,4000,5000,6000,8000")
    ap.add_argument("--policies", default="lru,win,dlfu,static,belady")
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--decay-every", type=int, default=4)
    ap.add_argument("--admit", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--win-admit", type=int, default=3)
    ap.add_argument("--prime", help="engine policy: a prefill trace [N, n_layer, k] (route_trace --prefill-trace)")
    ap.add_argument("--prime-half-life", type=int, default=4096)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--n-expert", type=int, default=512)
    ap.add_argument("--windows", type=int, default=0)
    a = ap.parse_args()

    if a.trace:
        trace = np.load(a.trace)
    elif a.synthetic:
        trace = synthetic(a.tokens)
    else:
        sys.exit("need --trace or --synthetic")
    keys = keys_of(trace)
    print(f"trace: {trace.shape[0]} tokens x {trace.shape[1]} layers x top-{trace.shape[2]}"
          f"{' (synthetic)' if a.synthetic else ''}")

    slots = [int(s) for s in a.slots.split(",")]
    pols = a.policies.split(",")
    print(f"{'slots':>8} " + " ".join(f"{p:>8}" for p in pols))
    for cap in slots:
        res = []
        for p in pols:
            if p == "lru":
                res.append(sim_lru(keys, cap))
            elif p == "dlfu":
                res.append(sim_dlfu(keys, cap, a.decay, a.decay_every, a.admit, a.margin))
            elif p == "win":
                res.append(sim_win(keys, cap, a.win, a.win_admit))
            elif p == "static":
                res.append(sim_static(keys, cap))
            elif p == "belady":
                res.append(sim_belady(keys, cap))
            elif p == "engine":
                if not hasattr(a, "_prime"):
                    a._prime = prime_counts(np.load(a.prime), a.n_expert, a.prime_half_life) if a.prime else None
                r, win, up = sim_engine(trace, cap, a.n_expert, a._prime, a.budget, a.decay, a.decay_every, a.admit, a.margin, a.windows)
                res.append(r)
                print(f"{'':>8} engine: {up} uploads ({up / trace.shape[0]:.1f} per token)"
                      + (f"; {a.windows}-token windows: " + " ".join(f"{w:.3f}" for w in win) if win else ""))
            else:
                sys.exit(f"unknown policy {p}")
        print(f"{cap:>8} " + " ".join(f"{r:8.3f}" for r in res), flush=True)


if __name__ == "__main__":
    main()
