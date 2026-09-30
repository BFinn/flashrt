#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sw115: prompt-lookup drafts (P-4), replayed on greedy text.

For each context: the sequence is the prompt's first N ids plus every generated token
(fr_bench --save-tokens), and the head's 3 drafts at every position (fr_bench --mtp --draft 3
--round-log). A round at position p (x_p known, drafting p + 1 ..):
- the head's drafts are kept while they equal the text (greedy decoding is exact, so the text is
  what the target produces);
- a lookup draft takes the most recent earlier occurrence of the last g tokens and proposes what
  followed it, up to M tokens; it is kept as far as it equals the text.
Policies (tokens per round = kept + 1; costs from sw114's greedy runs of the same context, the
verify phase per window length T measured for T <= 4 and extrapolated linearly above: an
estimate):
  A  head, K = 2 (the default);
  B  lookup when a match exists (proposal up to M), else the head with K = 2;
  C  as B, but only when the lookup's first token equals the head's first draft (the head drafts
     one token in any case);
  D  the head's 2 drafts, extended by lookup from the context plus those drafts, up to M in all.
Usage: lookup.py DIR SW114_LOGDIR NAME:IDS:N:COST ...
  NAME.gen and NAME.drafts in DIR; IDS the ids file under DIR/ids; N the prompt length; COST the
  sw114 context whose costs apply (32k, 245k, w9). A sequence is cut at its first end-of-turn
  token (248046, <|im_end|>) or the first repeated 64-gram: greedy text past that loops.
"""
import os
import statistics as st
import sys



EOT = 248046


def ints(path):
    return [int(x) for x in open(path).read().split()]


def costs(logdir, ctx):
    # draft phase a + b * drafts and verify ms per T from sw114's greedy runs (see its simulate.py)
    def rounds(p):
        return [l.split() for l in open(p)]
    dk = []
    ver = {}
    for k in (1, 2, 3):
        r = rounds(os.path.join(logdir, f"spec{k}_{ctx}_greedy.rounds"))
        dk.append((k, st.mean(float(x[4]) for x in r)))
        ver[k + 1] = st.mean(float(x[5]) for x in r)
    import re
    txt = open(os.path.join(logdir, f"plain_{ctx}_greedy.txt")).read()
    v = [float(m.group(1)) for m in re.finditer(r"^decode: .*?: ([0-9.]+) tok/s", txt, re.M)][1:]
    ver[1] = 1000.0 / st.mean(v)
    mx = st.mean(k for k, _ in dk)
    my = st.mean(v for _, v in dk)
    b = sum((k - mx) * (y - my) for k, y in dk) / sum((k - mx) ** 2 for k, _ in dk)
    slope = (ver[4] - ver[2]) / 2
    return my - b * mx, b, lambda T: ver[T] if T in ver else ver[4] + slope * (T - 4)


def main():
    d, logdir = sys.argv[1], sys.argv[2]
    for spec in sys.argv[3:]:
        name, idf, n, ctx = spec.split(":")
        n = int(n)
        gen = ints(os.path.join(d, name + ".gen"))
        cut = len(gen)
        seen = set()
        for i in range(len(gen)):
            if gen[i] == EOT:
                cut = i
                break
            if i >= 63:
                k = tuple(gen[i - 63:i + 1])
                if k in seen:
                    cut = i - 63
                    break
                seen.add(k)
        seq = ints(os.path.join(d, "ids", idf))[:n] + gen[:cut]
        print(f"{name}: {cut} of {len(gen)} generated tokens used")
        heads = {}
        for line in open(os.path.join(d, name + ".drafts")):
            f = [int(x) for x in line.split()]
            heads[f[0]] = f[1:]
        a0, b0, ver = costs(logdir, ctx)

        def head_kept(p, k):
            dr = heads.get(p, [])[:k]
            m = 0
            while m < len(dr) and p + 1 + m < len(seq) and dr[m] == seq[p + 1 + m]:
                m += 1
            return m

        for g in (2, 3, 4):
            # index of g-grams: the most recent end position, built as the walk advances
            last = {}
            for e in range(g - 1, n):
                last[tuple(seq[e - g + 1:e + 1])] = e
            built = n - 1

            def extend(upto):
                nonlocal built
                while built < upto:
                    built += 1
                    if built >= g - 1:
                        last[tuple(seq[built - g + 1:built + 1])] = built

            def lookup(ctx_tail, before):
                # proposal after the most recent earlier occurrence of ctx_tail (ending before `before`)
                e = last.get(tuple(ctx_tail))
                if e is None or e >= before:
                    return None
                return e

            for M in (2, 4, 8):
                res = {}
                for pol in "ABCD":
                    last.clear()
                    for e in range(g - 1, n):
                        last[tuple(seq[e - g + 1:e + 1])] = e
                    built = n - 1
                    p = n
                    tok = tim = 0.0
                    nlook = 0
                    end = len(seq) - M - 2
                    while p < end and p in heads:
                        # the index holds g-grams ending before p (the current one would find itself)
                        extend(p - 1)
                        e = lookup(seq[p - g + 1:p + 1], p)
                        prop = seq[e + 1:min(e + 1 + M, p + 1)] if e is not None else []   # known text only
                        kept = 0
                        if pol == "A" or (pol in "BC" and not prop) or (pol == "C" and prop[:1] != heads[p][:1]):
                            kept = head_kept(p, 2)
                            t = a0 + b0 * 2 + ver(3)
                        elif pol in "BC":
                            K = len(prop)
                            while kept < K and prop[kept] == seq[p + 1 + kept]:
                                kept += 1
                            t = (a0 + b0 if pol == "C" else a0) + ver(K + 1)
                            nlook += 1
                        else:   # D: head drafts, then lookup from the context plus them
                            dr = heads[p][:2]
                            ctxt = seq[: p + 1] + dr
                            e2 = lookup(ctxt[-g:], p)
                            ext = seq[e2 + 1:min(e2 + 1 + (M - 2), p + 1)] if e2 is not None and M > 2 else []
                            drafts = dr + ext
                            nlook += bool(ext)
                            while kept < len(drafts) and drafts[kept] == seq[p + 1 + kept]:
                                kept += 1
                            t = a0 + b0 * 2 + ver(len(drafts) + 1)
                        tok += kept + 1
                        tim += t
                        p += kept + 1
                    res[pol] = (tok / tim * 1000, nlook)
                base = res["A"][0]
                print(f"{name:8s} g={g} M={M}: " + "  ".join(
                    f"{pol} {v:6.1f} ({100 * (v / base - 1):+5.1f}%, lookup rounds {k})" for pol, (v, k) in res.items()))


if __name__ == "__main__":
    main()
