#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read ref_dump files (.frd): llama.cpp intermediate tensors recorded for parity tests.

    frd.py FILE.frd                  summary: one line per tensor name, with steps and shapes
    frd.py FILE.frd NAME [STEP]      print statistics of one record (first match)

As a module: records(path) yields (name, step, ggml_type, shape, numpy array); the array has
ggml's shape reversed (numpy's last axis is ggml's ne[0]).
"""
import struct
import sys
from collections import OrderedDict

import numpy as np


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if not h:
                return
            (nl,) = struct.unpack("<I", h)
            name = f.read(nl).decode()
            step, typ = struct.unpack("<ii", f.read(8))
            ne = struct.unpack("<4q", f.read(32))
            (nb,) = struct.unpack("<Q", f.read(8))
            data = f.read(nb)
            dt = np.float32 if typ == 0 else np.int32
            arr = np.frombuffer(data, dtype=dt).reshape(ne[::-1])
            yield name, step, typ, ne, arr


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    if len(sys.argv) == 2:
        seen = OrderedDict()
        for name, step, typ, ne, _ in records(path):
            key = name
            s = seen.setdefault(key, {"steps": [], "ne": ne, "n": 0})
            s["n"] += 1
            if step not in s["steps"]:
                s["steps"].append(step)
        for k, v in seen.items():
            print(f"{k:32s} x{v['n']:<3d} steps {v['steps'][:6]}{'...' if len(v['steps']) > 6 else ''} ne {v['ne']}")
        return
    want, step = sys.argv[2], (int(sys.argv[3]) if len(sys.argv) > 3 else None)
    for name, st, typ, ne, arr in records(path):
        if name == want and (step is None or st == step):
            a = arr.astype(np.float64)
            print(f"{name} step {st} ne {ne}: mean {a.mean():.6g} std {a.std():.6g} min {a.min():.6g} max {a.max():.6g}")
            print(arr.reshape(-1)[:16])
            return
    sys.exit(f"{want} not found")


if __name__ == "__main__":
    main()
