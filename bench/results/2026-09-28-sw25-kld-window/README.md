# sw25: KLD gate for speculative verify windows (2026-09-28)

**What:** `fr_kld --fast --window 4`: the scored half of each 8K chunk runs in windows of 4
tokens where only the first j (random, 1..4) are the chunk's and the rest random tokens; the j
real rows are scored and committed, the rest rewound. Every rewind path (GDN state, conv and
PLE histories, KV caches and the indexer ring) is exercised: 3,241 windows, 2,400 rewound.

| | KLD mean | median | p99 | same top-1 |
|---|---|---|---|---|
| windows of 4 with rewinds | **0.008718** | 0.001139 | 0.1110 | 96.50% |
| plain fast path (sw17-sw22) | 0.0087-0.0092 | | | |
| llama.cpp against itself (noise floor) | 0.0078-0.0086 | | | |

The window path is exact within the gate. (Kernels at sw24: eager windows, generic hc path.)
