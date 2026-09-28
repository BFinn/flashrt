# sw33: the P2 gate measured; window KLD on the current kernels; Q2_0 head at depth (2026-09-28)

After the graph-scratch fix. Settings as sw31 (temperature 1.0, top-k 20, top-p 0.95; q8 host
KV + hot set 4,096; head vocabulary 32,768 ranked + prompt), 6 windows of 128 tokens.

## KLD, windows of 3 with rewinds (`fr_kld --fast --window 3 --kv-hot 512`)

| | KLD mean | median | p99 | same top-1 | windows (rewound) |
|---|---|---|---|---|---|
| current kernels (fused hc, grouped hits, window graphs) | **0.009232** | 0.001191 | 0.1158 | 96.76% | 4,067 (2,689) |

At the noise floor (llama.cpp against itself with a Q8_0 KV cache: 0.0078; flashrt q8 KV plain
fast path: 0.0092).

## Speed at temperature 1.0

| Arm | tok/s min / max / mean | tokens/round | hit rate |
|---|---|---|---|
| 32K `--spec 2` (Q4_0 head) | 103.9 / 122.5 / 115.3 | 1.856 | 91.5% |
| 32K `--spec 2`, Q2_0 head | 95.4 / 124.8 / 107.1 | 1.787 | 90.3% |
| 245K `--spec 2` (Q4_0 head) | 75.9 / 90.5 / 83.2 | 1.992 | 80.4% |
| 245K `--spec 2`, 11 workers | 74.0 / 86.0 / 81.7 | 1.992 | 80.4% |
| 245K `--spec 2`, Q2_0 head | 72.7 / 102.1 / 86.9 | 1.832 | 86.8% |
| 245K `--spec 1`, Q2_0 head | 83.5 / 96.0 / 89.4 | 1.584 | 84.9% |
| 245K greedy `--spec 3`, Q2_0 head | 76.8 / 118.0 / 91.4 | 3.484 | 61.7% |

With sw31's plain arms (32K 97.9, 245K 78.5 mean) and `--spec 1` (32K 114.2, 245K 90.2):

- **P2 gate (≥ 80 at 32K, ≥ 72 at 250K, temperature 1.0): met,** by plain decoding already and
  with margin by `--spec 1` (+17% at 32K, +15% at 245K on the means).
- **At temperature 1.0 one draft is the best K:** a second draft's acceptance does not pay for
  the wider window's extra expert misses. Greedy at depth (the text there repeats earlier
  articles) keeps 3.5 tokens per round with K = 3.
- **The Q2_0 head** frees about 500 slots; its acceptance is within run-to-run spread of Q4_0's.
  Not adopted as the default yet: the evidence is mixed at 32K.
- **11 CPU workers** do not help at depth.

**Distribution test at 245K** (`--dist-test 300`, `--spec 2`): the first token equal in 300 of
300 (TV 0), the second in 59 of 59.
