#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replace machine-specific strings (home directories, host and user names) in result files
before they are committed, with the placeholders of docs/engine.md (Runbook).

  scrub.py [--rules FILE] PATH...

The rules come from an untracked file (default `.scrub.local` at the repository root), one per
line: `REGEX => REPLACEMENT`, applied in order; `#` starts a comment. Text files are rewritten in
place with their bytes otherwise unchanged (no newline translation); binary files are skipped.
"""
import argparse
import os
import re
import sys


def rules_of(path):
    rules = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rx, rep = line.split(" => ", 1)
        rules.append((re.compile(rx), rep))
    return rules


def files_of(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    yield os.path.join(root, n)
        else:
            yield p


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=os.path.join(here, ".scrub.local"))
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if not os.path.exists(a.rules):
        sys.exit(f"no rules file {a.rules}")
    rules = rules_of(a.rules)
    changed = 0
    for f in files_of(a.paths):
        try:
            with open(f, encoding="utf-8", newline="") as fh:
                s = fh.read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        t = s
        for rx, rep in rules:
            t = rx.sub(lambda m, rep=rep: rep, t)
        if t != s:
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(t)
            changed += 1
    print(f"scrubbed {changed} file(s)")


if __name__ == "__main__":
    main()
