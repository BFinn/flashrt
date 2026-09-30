#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sw114: a per-round draft length, simulated on logged rounds.

For each context (32k, 245k, w9) and mode (greedy, t1), from the fr_bench --round-log files of
--spec 1, 2 and 3 and the plain run:
- costs: the draft phase as a + b * drafts (a least-squares fit over the three runs' round means;
  it includes the catch-up of the rows kept last round), the verify phase per window length T
  (T = 1 is the plain run's step);
- calibration: in the spec 3 rounds, how often draft j is kept, given drafts 1 .. j-1 were, by its
  probability under the head (bins);
- policies replayed on the spec 3 rounds. A round drafts until a draft's probability falls below
  theta (or K_max) and verifies the drafts before it: the stopping draft was drafted (paid) but not
  verified. Kept drafts = min(a, K'), since acceptance of a prefix does not depend on later drafts.
  Tokens per round = kept + 1.
Fixed K = 1, 2, 3 replayed the same way check the model against the measured runs.

Usage: simulate.py DIR
"""
import glob
import os
import re
import statistics as st
import sys


def rounds(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) < 6:
            continue
        out.append({"kd": int(f[2]), "a": int(f[3]), "draft": float(f[4]), "verify": float(f[5]), "q": [float(x) for x in f[6:]]})
    return out


def plain_ms(path):
    # one decode line per window; mean ms per token over windows 2.. (window 1 warms up)
    v = [float(m.group(1)) for m in re.finditer(r"^decode: .*?: ([0-9.]+) tok/s", open(path).read(), re.M)]
    v = v[1:] if len(v) > 1 else v
    return 1000.0 / st.mean(v)


def fit_line(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    d = sys.argv[1]
    for ctx in ["32k", "245k", "w9"]:
        for mode in ["greedy", "t1"]:
            runs = {}
            for k in (1, 2, 3):
                p = os.path.join(d, f"spec{k}_{ctx}_{mode}.rounds")
                if os.path.exists(p):
                    runs[k] = rounds(p)
            pp = os.path.join(d, f"plain_{ctx}_{mode}.txt")
            if len(runs) < 3 or not os.path.exists(pp):
                continue
            # costs
            dk = [(k, st.mean(r["draft"] for r in runs[k])) for k in (1, 2, 3)]
            a0, b0 = fit_line([k for k, _ in dk], [v for _, v in dk])
            ver = {1: plain_ms(pp)}
            for k in (1, 2, 3):
                ver[k + 1] = st.mean(r["verify"] for r in runs[k])
            meas = {k: sum(r["a"] + 1 for r in runs[k]) / sum(r["draft"] + r["verify"] for r in runs[k]) * 1000 for k in (1, 2, 3)}
            print(f"\n== {ctx} {mode}: plain {1000 / ver[1]:.1f} tok/s; draft {a0:.2f} + {b0:.2f}/draft ms; verify ms by T: "
                  + ", ".join(f"{t}:{ver[t]:.2f}" for t in sorted(ver)))
            print("   measured (draft + verify only): " + ", ".join(f"K={k} {meas[k]:.1f} tok/s ({st.mean(r['a'] + 1 for r in runs[k]):.2f} tok/round)" for k in (1, 2, 3)))
            r3 = runs[3]
            # calibration: P(keep draft j | kept 1..j-1) by the draft's probability
            bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01]
            for j in range(3):
                row = []
                for lo, hi in zip(bins, bins[1:]):
                    sel = [r for r in r3 if r["a"] >= j and lo <= r["q"][j] < hi]
                    if sel:
                        row.append(f"[{lo:.1f},{min(hi, 1):.1f}) {sum(r['a'] > j for r in sel) / len(sel):.2f} (n={len(sel)})")
                print(f"   keep draft {j + 1} | earlier kept, by its q: " + "; ".join(row))

            def sim(policy):
                tok = tim = 0.0
                for r in r3:
                    kp, made = policy(r["q"], r["a"])
                    tok += min(r["a"], kp) + 1
                    tim += (a0 + b0 * made if made else 0.0) + ver[kp + 1]
                return tok / tim * 1000, tok / len(r3)

            res = []
            for k in (1, 2, 3):
                v, t = sim(lambda q, a, k=k: (k, k))
                res.append((f"fixed K={k}", v, t))
            v, t = sim(lambda q, a: (a, min(3, a + 1)))   # knows which drafts will be kept
            res.append(("fixed oracle", v, t))
            for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                for kmax in (2, 3):
                    def pol(q, a, th=th, kmax=kmax):
                        kp = 0
                        while kp < kmax and q[kp] >= th:
                            kp += 1
                        return kp, min(kmax, kp + 1)   # the stopping draft was drafted too
                    v, t = sim(pol)
                    res.append((f"stop below {th:.1f}, K_max {kmax}", v, t))
            best = max((x for x in res if x[0] != "fixed oracle"), key=lambda x: x[1])
            for name, v, t in res:
                if name.startswith("fixed") or name == best[0]:
                    print(f"   sim {name:24s} {v:6.1f} tok/s  {t:.2f} tok/round")
            thr = sorted([x for x in res if not x[0].startswith("fixed")], key=lambda x: -x[1])[:4]
            # the oracle is printed with the fixed rows ("fixed oracle": an upper bound)
            print("   best rules: " + "; ".join(f"{n} {v:.1f}" for n, v, _ in thr))


if __name__ == "__main__":
    main()
