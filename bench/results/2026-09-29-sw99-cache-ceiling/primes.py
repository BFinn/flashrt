#!/usr/bin/env python3
# sw99: cache primes (primes.py) / admission settings (admission.py) in the engine-policy simulation;
# argument: the folder with sw99.sh's traces (w9.*.npy, wiki.*.npy)
import sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../tools"))
from cache_sim import prime_counts, sim_engine
D = sys.argv[1]
E, CAP = 512, 7780
dec = {p: np.load(f"{D}/{p}.decode_topk.npy") for p in ("w9", "wiki")}
pre = {p: np.load(f"{D}/{p}.prefill_topk.npy") for p in ("w9", "wiki")}
def dec_counts(p):
    c = np.zeros(48 * E)
    t = dec[p]
    np.add.at(c, (np.arange(48)[None, :, None] * E + t).reshape(-1), 1.0)
    return c
def run(name, p, prime, **kw):
    r, win = sim_engine(dec[p], CAP, E, prime, windows=64, **kw)
    print(f"{p:5s} {name:42s} {100*r:5.1f}%  windows " + " ".join(f"{100*w:4.1f}" for w in win), flush=True)
for p in ("w9", "wiki"):
    other = "wiki" if p == "w9" else "w9"
    base = prime_counts(pre[p], E, 4096)
    run("prompt, half-life 4096 (engine)", p, base)
    for K in (64, 512, 4096):
        run(f"prompt's last {K} tokens only", p, prime_counts(pre[p][-K:], E, 0))
    g = dec_counts(other)
    for w in (0.25, 1.0, 4.0):
        run(f"prompt + {w} x the other text's generation", p, base / base.sum() + w * g / g.sum())
    run("the other generation alone", p, g)
    run("engine, admit 1, margin 1.2", p, base, admit=1.0, margin=1.2)
    run("engine, decay 0.5 per 4", p, base, decay=0.5)
