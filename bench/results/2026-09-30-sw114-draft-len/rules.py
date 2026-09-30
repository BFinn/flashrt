#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sw114: one draft-length rule for every context, replayed on the spec 3 rounds (simulate.py's
cost model), relative to fixed K = 2: thresholds on the drafts' probabilities, then K chosen from
the previous round's kept count (every map to 1..3), with and without the threshold.
Usage: rules.py LOGDIR"""
import math, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, 'simulate.py')); sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
import os
d = sys.argv[1]
ctxs = [(c, m) for c in ['32k', '245k', 'w9'] for m in ['greedy', 't1']]
data = {}
for c, m in ctxs:
    runs = {k: sim.rounds(os.path.join(d, f'spec{k}_{c}_{m}.rounds')) for k in (1, 2, 3)}
    dk = [(k, st.mean(r['draft'] for r in runs[k])) for k in (1, 2, 3)]
    a0, b0 = sim.fit_line([k for k, _ in dk], [v for _, v in dk])
    ver = {1: sim.plain_ms(os.path.join(d, f'plain_{c}_{m}.txt'))}
    for k in (1, 2, 3): ver[k + 1] = st.mean(r['verify'] for r in runs[k])
    data[(c, m)] = (runs[3], a0, b0, ver)
def run(pol, c, m):
    r3, a0, b0, ver = data[(c, m)]
    tok = tim = 0.0
    for r in r3:
        kp, made = pol(r['q'])
        tok += min(r['a'], kp) + 1
        tim += a0 + b0 * made + ver[kp + 1]
    return tok / tim * 1000
def thr(th, kmax, th2=None):
    def pol(q):
        kp = 0
        while kp < kmax and q[kp] >= (th if kp == 0 or th2 is None else th2):
            kp += 1
        return kp, min(kmax, kp + 1)
    return pol
def fixed(k): return lambda q: (k, k)
base = {cm: run(fixed(2), *cm) for cm in ctxs}
rows = []
cands = [('fixed K=1', fixed(1)), ('fixed K=3', fixed(3))]
for th in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    for kmax in (2, 3):
        cands.append((f'stop<{th:.2f} Kmax{kmax}', thr(th, kmax)))
# first draft always, then threshold on later drafts
for th in [0.3, 0.5, 0.7]:
    for kmax in (2, 3):
        cands.append((f'd1 always, later<{th:.1f} Kmax{kmax}', thr(0.0, kmax, th)))
for name, pol in cands:
    rel = [run(pol, *cm) / base[cm] for cm in ctxs]
    gm = math.exp(sum(math.log(x) for x in rel) / len(rel))
    rows.append((gm, name, rel))
rows.sort(reverse=True)
print('vs fixed K=2 (sim):', '  '.join(f'{c}-{m}' for c, m in ctxs))
for gm, name, rel in rows[:14]:
    print(f'{name:28s} gm {100*(gm-1):+5.1f}%  ' + '  '.join(f'{100*(x-1):+5.1f}' for x in rel))

# K from the previous round's observed outcome (kept count, and whether it was a full accept)
import itertools
def run_state(kmap, th, c, m, kmax_th=None):
    r3, a0, b0, ver = data[(c, m)]
    tok = tim = 0.0
    state = 2   # the start: K = 2
    for r in r3:
        K = state
        kp = 0
        if th is None: kp, made = K, K
        else:
            while kp < K and r['q'][kp] >= th: kp += 1
            made = min(K, kp + 1)
        kept = min(r['a'], kp)
        tok += kept + 1
        tim += a0 + b0 * made + ver[kp + 1]
        state = kmap[(kept, kept == kp)]
    return tok / tim * 1000
keys = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, True)]
res = []
for vals in itertools.product((1, 2, 3), repeat=len(keys)):
    kmap = dict(zip(keys, vals))
    for th in (None, 0.3):
        rel = [run_state(kmap, th, *cm) / base[cm] for cm in ctxs]
        gm = math.exp(sum(math.log(x) for x in rel) / len(rel))
        res.append((gm, kmap, th, rel))
res.sort(key=lambda x: -x[0])
print('\nK from the last round (kept, full): K ;  gm vs fixed 2; per context')
for gm, kmap, th, rel in res[:10]:
    print(' '.join(f'{k[0]}{"f" if k[1] else ""}:{v}' for k, v in kmap.items()), f'th={th}', f'gm {100*(gm-1):+5.1f}%  ' + '  '.join(f'{100*(x-1):+5.1f}' for x in rel))
simple = {(0, False): 1, (0, True): 1, (1, False): 2, (1, True): 2, (2, False): 2, (2, True): 3, (3, True): 3}
for th in (None, 0.3):
    rel = [run_state(simple, th, *cm) / base[cm] for cm in ctxs]
    gm = math.exp(sum(math.log(x) for x in rel) / len(rel))
    print('simple (next K = kept + 1, capped 1..3)', f'th={th}', f'gm {100*(gm-1):+5.1f}%  ' + '  '.join(f'{100*(x-1):+5.1f}' for x in rel))
