#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel time per plain token against per speculative round, from two nsys cuda_gpu_kern_sum CSVs
and the fr_bench outputs of the same runs (token and round counts).

  kcmp2.py PLAIN_PREFIX SPEC_PREFIX   (each: PREFIX_cuda_gpu_kern_sum.csv and PREFIX.txt)
"""
import csv
import re
import sys


def load(prefix):
    d = {}
    for r in csv.DictReader(open(prefix + "_cuda_gpu_kern_sum.csv")):
        n = r["Name"].replace("flashrt::qwen4exp::<unnamed>::", "").replace("flashrt::", "").split("(")[0][:60]
        d[n] = d.get(n, 0.0) + float(r["Total Time (ns)"]) / 1e6
    return d


def count(prefix, pattern):
    m = re.search(pattern, open(prefix + ".txt").read())
    return int(m.group(1)) if m else None


def main():
    plain, spec = sys.argv[1], sys.argv[2]
    p, s = load(plain), load(spec)
    tokens = count(plain, r"decode: (\d+) tokens")
    rounds = count(spec, r"speculative: (\d+) rounds")
    print(f"plain: {tokens} tokens, spec: {rounds} rounds (kernel ms per token / per round)")
    keys = sorted(set(p) | set(s), key=lambda k: -s.get(k, 0.0) / rounds)
    for k in keys[:28]:
        print(f"{k:62s} {p.get(k, 0.0) / tokens:8.3f} {s.get(k, 0.0) / rounds:8.3f}")
    print(f"{'total':62s} {sum(p.values()) / tokens:8.3f} {sum(s.values()) / rounds:8.3f}")


if __name__ == "__main__":
    main()
