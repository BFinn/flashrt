# Window 9, reference engines: an earlier copy of `2026-09-27-w9-validation` (2026-09-27, 11:34-13:02)

**This folder duplicates part of `2026-09-27-w9-validation`.** Its three files, `w9-all.out`,
`w9-summary.txt` and `w9b.sh`, are byte-identical to the files of the same name there (checked
with `cmp`). Two sessions committed the window 9 results five minutes apart on 2026-09-27. This
folder came first and has no per-run logs, no `w9.sh` and no `w9_summary.py`. The other folder
has all of those and the README. The docs (`README.md`, `docs/engine.md`, `docs/background.md`,
`docs/sweet-spots.md`) cite only `2026-09-27-w9-validation`, so cite that one.

## What was measured

Window 9 is the baseline flashrt is measured against: llama.cpp and Strata on the target box
(RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600). All arms process one growing conversation that
reuses its prefix, at 4 depths. Every arm gets the same token ids and generates 384 tokens, and
the arms are interleaved.

| Arm | Engine and flags (from `w9b.sh`) | Runs |
|---|---|---:|
| L | llama.cpp dev tree (`$LLAMA_CPP`, QSA block selection and pooled-key cache), no MTP, 48-slot expert cache, experts on the CPU, q8_0 KV | 4 |
| G | Strata (`$STRATA`, config `strata-q2_0.json`), tuned: `--vram-reserve-mib 1024`, `--pcie-frac 0.35`, `--pool-workers 8`; MTP, greedy | 4 |
| S | as G, temperature 1.0, top_p 0.95, top_k 20 | 4 |
| S06 | as G, temperature 0.6, top_p 0.95, top_k 20 | 2 |

The engine builds are recorded in the w9-validation README: llama.cpp branch `mtp` at e4c893841
with uncommitted patches, and Strata 0.1.6 with PR #19 (5ffa807).

`w9b.sh` is the second half of the window, run after an interruption; `w9.sh` (only in
w9-validation) is the first half. `w9-all.out` concatenates both outputs, and `w9-summary.txt` is
`w9_summary.py` run over it. Decode is llama.cpp's `decode_tps` and Strata's `decode_wall_tps`.
Prefill counts only the new tokens: llama.cpp reports it, and for Strata it is new tokens divided
by the time to first token.

**Caveat on Strata's numbers** (as in the w9-validation README, added 2026-09-29): with
`--expert-cache` on, Strata 0.1.6 logs a warning that its GPU hit path "is NOT CORRECT", and its
tokens diverge from a cache-off run. Its timings are real. Its draft acceptance, and so its MTP
speed, come from outputs that differ from the model's.

## Results (`w9-summary.txt`)

The columns are the actual prompt lengths: 1,055 / 32,793 / 134,029 / 250,712 tokens.

Decode tok/s, mean ± sd:

| Arm | 1,055 | 32,793 | 134,029 | 250,712 |
|---|---:|---:|---:|---:|
| L (n=4) | 37.4 ± 0.6 | 37.2 ± 1.1 | 32.8 ± 1.6 | 30.7 ± 1.1 |
| G (n=4) | 87.0 ± 0.7 | 96.0 ± 2.2 | 85.0 ± 2.7 | 80.4 ± 3.6 |
| S (n=4) | 80.5 ± 4.1 | 79.1 ± 1.3 | 73.9 ± 1.8 | 69.9 ± 7.2 (one run excluded, see below) |
| S06 (n=2) | 75.7 ± 1.1 | 80.3 ± 4.8 | 74.6 ± 3.6 | 72.2 ± 0.9 |

Prefill tok/s of the new tokens, mean ± sd:

| Arm | 1,055 | 32,793 | 134,029 | 250,712 |
|---|---:|---:|---:|---:|
| L | 616.0 ± 6.5 | 1107.4 ± 4.0 | 723.2 ± 0.9 | 442.4 ± 0.7 |
| G | 677.4 ± 4.2 | 1133.8 ± 0.8 | 1089.0 ± 0.5 | 944.9 ± 0.3 |
| S | 674.1 ± 2.5 | 1133.8 ± 0.8 | 1089.2 ± 0.4 | 945.1 ± 0.1 |
| S06 | 667.8 ± 12.0 | 1134.3 ± 0.6 | 1089.6 ± 0.2 | 945.3 ± 0.0 |

Notes from `w9-all.out`:
- **G's first attempt failed:** "engine exited before READY". The window restarted at 11:44:28.
- **One S run ended early at 250,712** on a sampled end-of-sequence token. The summary excludes
  runs that generated fewer than 100 tokens.
- **One S06 run generated 300 tokens at 1,055,** not 384. It is above the 100-token cut, so the
  summary includes it.
- **At exit, `w9b.sh` restarted the live server** (`flashnext-262k-server`). That was the box
  practice on 2026-09-27. Since then no live service runs on the box.

## Where the docs use it

The docs take window 9 from `2026-09-27-w9-validation`. That is the baseline in `README.md`
and `docs/engine.md`'s reference-engine row, and the background in `docs/background.md` and
`docs/sweet-spots.md`.

## Files

- `w9b.sh`: the second half of the window (G 1, S 1; G 2, S 2, L 2; S 3, L 3, G 3; L 4, G 4,
  S 4; S06 1, S06 2).
- `w9-all.out`: per-depth lines and one `SUMMARY` JSON line per run, for both halves.
- `w9-summary.txt`: the tables above.
