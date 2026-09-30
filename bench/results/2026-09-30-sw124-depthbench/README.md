# sw124: window 9's protocol after P-5's kernels (2026-10-01)

sw110's protocol and arms (`sw124.sh`, n = 5 each, interleaved) on the build with the QSA selection
on a thread-block cluster (sw121) and the hot set's CLOCK in parallel (sw122). Both are
output-identical, so the greedy arm generates the same tokens as sw119: its hit rates (85.9 / 81.2 /
83.1 / 81.9%) and draft acceptance (55.8 / 54.6 / 55.5 / 55.8%) are sw119's exactly.
`sw124-summary.txt` is `bench/depthsum.py` over sw124, sw119 and sw110.

| decode tok/s | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P, no head, greedy | 101.3 ± 0.7 | 94.9 ± 1.4 | 93.5 ± 1.3 | 90.3 ± 1.5 |
| G, head, greedy | 125.8 ± 3.4 | 107.8 ± 0.7 | 104.2 ± 2.0 | 99.1 ± 1.8 |
| S, head, temperature 1.0 | 124.6 ± 9.8 | 106.7 ± 3.9 | 102.3 ± 3.1 | 102.3 ± 3.5 |
| sw119 (P n = 2 / G / S) | 102.0 / 125.7 / 118.3 | 94.0 / 106.0 / 106.1 | 87.7 / 99.5 / 99.5 | 82.3 / 92.3 / 96.8 |

- **At depth:** P +6.6% at 134K and +9.7% at 250K; G +4.7% and +7.4%. At 1K and 32K, where the
  selection is small, level. The S arm samples its own text, so its changes carry the text's
  variance too.
- **Against Strata 0.1.6** (w9-validation) greedy: G ahead by 45%, 12%, 23%, 23%; P ahead at 1K,
  134K and 250K by 10-16%, level at 32K. At temperature 1.0, S ahead by 35-55%.
- **Against llama.cpp** greedy: P 2.6-2.9x, G 2.9-3.4x.
