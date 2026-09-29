# sw96: window 9's protocol with prefix reuse (2026-09-29)

Phase 2 of `docs/improvement-plan.md`: sw91 again, now that the engine reuses a prompt's shared
prefix through host checkpoints (sw95). These are the same token ids as the reference runs
(`strata-ids.json`: 1K / 32K / 134K / 250K). Each depth's prompt keeps the previous one's text and
ends with the same 19-token instruction. The depths run as one sequence, 384 tokens per depth
without stopping at end-of-sequence, with a fresh engine per run. Five runs per arm, interleaved
in a rotating order (`sw96.sh`). Every row records the reused tokens, prompt time, draft
acceptance, the decode's expert-cache hit rate, and the GPU's clock, temperature and power. The
build includes the MTP head's KV mirror (sw98) and its fast load.

Arms: **P** no MTP head, greedy; **G** `--mtp --spec 2`, greedy (argmax drafts); **S** `--mtp --spec 2`,
temperature 1.0, top-p 0.95, top-k 20 (sampled drafts). Otherwise the engine's defaults: 262K
context, q8 host KV with a hot set of 4,096, swap budget 32, 8 host checkpoints, no cache prior.
`sw96-summary.txt` is `bench/depthsum.py` on `sw96.out`.

## Decode tok/s (wall clock, mean ± sd, n = 5)

| Arm | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P, no MTP, greedy | 94.5 ± 2.2 | 83.1 ± 0.6 | 78.3 ± 0.5 | 73.4 ± 0.7 |
| G, MTP, greedy | 105.2 ± 2.6 | 84.0 ± 1.1 | 78.8 ± 1.8 | 75.2 ± 1.6 |
| S, MTP, temperature 1.0 | 100.7 ± 8.3 | 81.0 ± 4.3 | 79.5 ± 2.6 | 77.3 ± 4.8 |
| sw91 (no reuse), n = 3: P / G / S | 94.7 / 106.6 / 82.2 | 81.7 / 83.7 / 77.7 | 77.4 / 81.0 / 81.0 | 73.0 / 74.8 / 76.4 |
| Strata 0.1.6 greedy / t 1.0 (w9, n = 4) | 87.0 / 80.5 | 96.0 / 79.1 | 85.0 / 73.9 | 80.4 / 69.9 |
| llama.cpp greedy (w9, n = 4) | 37.4 | 37.2 | 32.8 | 30.7 |

Strata's build warns that its expert-cache path "is NOT CORRECT" (its tokens diverge from a
cache-off run). Its timings are real; its outputs are not the model's.

## Prefill and reuse

| | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| reused prompt tokens (every arm, every run) | 0 | 0 | 32,768 | 134,004 |
| new tokens | 1,055 | 32,793 | 101,261 | 116,708 |
| prompt time P, s | 1.23 | 5.79 | 17.59 | 22.31 |
| prompt time G and S, s | 1.29 | 6.57 | 20.13 | 25.77 |
| prefill of the new tokens P / G, tok/s | 855 / 820 | 5,662 / 4,988 | 5,756 / 5,031 | 5,232 / 4,529 |

- **Reuse now matches the reference engines'.** Strata reused 32,768 and 131,072 tokens at 134K
  and 250K, and llama.cpp about 30,700 and 132,000. flashrt reuses 32,768 and 134,004: up to 24
  tokens before the previous prompt's end. It takes that tail checkpoint once it has seen a
  prompt keep the previous text but not its last tokens. The step to 32K is the first such
  prompt, so it prefills cold.
- **Time to the first token at 250K:** 25.8 s with the head and 22.3 s without it. sw91 took about
  117 s with the head, re-prefilling all 250,712 tokens at 2,150 tok/s. Strata took 123 s for its
  119,640 new tokens.
- **With the head, prefill at depth is 4,529-5,031 tok/s** (sw91: 2,149-2,429), from the head's
  KV mirror (sw98).

## Decode did not move: the expert cache is the depth gap

| Expert-cache hit rate in the decode, % | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P | 82.1 | 75.0 | 76.6 | 76.6 |
| G | 77.0 | 66.0 | 67.5 | 68.0 |
| S | 76.7 | 63.7 | 66.7 | 66.6 |

| Draft acceptance, % | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| G | 56.7 | 55.8 | 57.3 | 55.2 |
| S | 52.6 | 56.5 | 59.6 | 61.1 |

- Decode is within the spread of sw91 in every cell. Reusing the prefix changes the prompt's time,
  not what decode costs. The expert cache is still filled from the prompt's routing counts, and the
  generation routes unlike the prompt: 66-68% hits with the head, 75-77% without, against 93% on
  wikitext (sw88).
- The head's arms hit less: the head takes VRAM from the cache (7,780 slots against 8,634), and each
  verify window routes up to three tokens at once.
- **Against Strata, greedy:** ahead at 1K; behind by 12.5% at 32K, 7% at 134K and 6.5% at 250K.
  **At temperature 1.0:** ahead at 1K, 134K and 250K, and within the spread at 32K.
- So P-2 (the cache's warm-up and adaptation) is next: sw99 measures the ceiling any cache policy has
  on this generation.

GPU at the rows' ends: SM clock 2,805-2,820 MHz, 48-52 °C, 138-177 W (no throttling).
