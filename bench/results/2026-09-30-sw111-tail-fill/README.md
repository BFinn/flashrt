# sw111: a cache fill weighted toward the prompt's last tokens (2026-09-30)

The expert cache is filled from a prompt's routing counts. An answer's first tokens route more
like the prompt's last tokens (the chat template, the instruction) than like its bulk. sw99's
simulator under the current policy: adding the routing of the last 16 tokens at twice the prompt's
total lifted window 9's first 64 tokens from 67% to 77% hits, and cost wikitext 0.2 points.

**The option** (`--cache-tail-tokens 16 --cache-tail-weight W`, engine and `fr_bench`; 0 = off):
during a chunked prefill the routing kernels and the batch path also count the prompt's last N
tokens into a separate buffer. `ForwardRef::fill_counts` adds it, scaled to W times the whole
count's total, for the fill and the policy's seed.

**Teacher-forced** (as sw104, 3 runs per arm, rotating; `sw111.out`):

| Tail weight | Window 9, plain | Wikitext | Window 9 with the head |
|---|---:|---:|---:|
| 0 | 87.0 ± 0.6 (82.1% hits) | 107.9 ± 0.9 (93.6%) | 107.9 ± 1.1 (72.2%) |
| 0.5 | 87.5 ± 0.7, +0.6% (84.0%) | 101.9 ± 2.3, −5.6% (93.6%) | 116.3 ± 3.0, +7.8% (77.7%) |
| 2 | 87.3 ± 0.7, +0.4% (84.1%) | 103.7 ± 0.2, −3.9% (93.7%) | 119.0 ± 0.9, +10.3% (78.4%) |

**The agentic session** (greedy, 2 runs per arm; the two arms read the same repository state):

| Tail weight | Decode, whole session | Hits | Generated | Turns 0 / 1: hits, decode |
|---|---:|---:|---:|---|
| 0 | 125.8, 125.5 tok/s | 84.3% | 3,141 | 74.1% / 56.7%, 121 / 94-101 tok/s |
| 2 | 124.0, 124.1 | 84.2% | 3,516 | 78.2% / 75.8%, 133 / 129 tok/s |

- **It does what it was built for:** the short turns right after a prompt hit far more (turn 1:
  56.7 → 75.8%, +33% decode), and window 9 with the head gains 10%.
- **It is not a net gain elsewhere:** the whole agent session is flat (−1.3%, different text after
  the first turns), window 9 plain is flat despite +2 hit points, and wikitext loses 4-6% at an
  unchanged hit rate. There its misses cost more: 47.2 against 43.3 µs for a layer with one
  miss, and the host waits 7.65 against 7.28 ms per token for the GPU's routing.
- **So it stays off by default.** Which experts miss changes the cost as well as how many do
  (compare sw104: the seed scale sped wikitext by 5% at an unchanged hit rate). A hypothesis, not
  measured: repeatedly missed experts are served from the CPU's L3 cache.

Agent runs are comparable only within one build: `agent_trace.py` reads this repository's source
files, which change with every commit.
