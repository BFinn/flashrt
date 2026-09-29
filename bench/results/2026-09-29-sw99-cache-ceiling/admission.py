#!/usr/bin/env python3
# sw99: cache primes (primes.py) / admission settings (admission.py) in the engine-policy simulation;
# argument: the folder with sw99.sh's traces (w9.*.npy, wiki.*.npy)
import sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../tools"))
from cache_sim import prime_counts, sim_engine
D = sys.argv[1]; E = 512
dec = {p: np.load(f"{D}/{p}.decode_topk.npy") for p in ("w9", "wiki")}
base = {p: prime_counts(np.load(f"{D}/{p}.prefill_topk.npy"), E, 4096) for p in ("w9", "wiki")}
print(f"{'admit':>5} {'margin':>6} {'budget':>6} | {'w9 hits':>8} {'uploads/tok':>11} | {'wiki hits':>9} {'uploads/tok':>11}")
for admit, margin, budget in [(2, 1.5, 32), (1.5, 1.5, 32), (1, 1.5, 32), (1, 1.2, 32), (1, 1.0, 32), (0.5, 1.2, 32), (1, 1.2, 16), (1, 1.2, 64)]:
    out = []
    for p in ("w9", "wiki"):
        r, _, up = sim_engine(dec[p], 7780, E, base[p], budget=budget, admit=admit, margin=margin)
        out += [100 * r, up / dec[p].shape[0]]
    print(f"{admit:5.1f} {margin:6.1f} {budget:6d} | {out[0]:7.1f}% {out[1]:11.1f} | {out[2]:8.1f}% {out[3]:11.1f}", flush=True)
