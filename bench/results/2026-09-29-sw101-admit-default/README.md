# sw101: admit 1 / margin 1.2 as the expert cache's default (2026-09-29)

sw100 made the adaptive cache admit a missed expert at count 1, and at 1.2 times the weakest
resident's count (was 2 and 1.5). This checks the gate and the configurations that sw100 did not
measure (`sw101.sh`, one build).

**KLD gate** (2 × 8K wikitext against the FP16-KV llama.cpp base): the fast path's value depends on
what the cache holds (sw94), so it moves. It moved down:

| Configuration | Old admission | New |
|---|---:|---:|
| `--fast` | 0.008688 (sw92) | 0.008602 |
| `--fast --window 3 --prefill-chunk 2048 --kv-hot 512` | 0.009124 (sw92) | 0.008763 |

**The MTP head, teacher-forced** on window 9's generation (as sw100, `--spec 2`, 2 runs each):
old 88.3 / 86.7 tok/s at 57.6% hits, new 89.8 / 89.3 at 58.9% (+2.3%).

**Window 9's protocol** (as sw96, arms G and S, n = 3; `sw101-summary.txt`):

| Decode tok/s | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| G, greedy, old admission (sw96, n = 5) | 105.2 ± 2.6 | 84.0 ± 1.1 | 78.8 ± 1.8 | 75.2 ± 1.6 |
| G, greedy, new | **112.1 ± 1.7** | **87.0 ± 0.5** | **83.7 ± 0.1** | 74.6 ± 0.8 |
| S, temperature 1.0, old (sw96) | 100.7 ± 8.3 | 81.0 ± 4.3 | 79.5 ± 2.6 | 77.3 ± 4.8 |
| S, temperature 1.0, new | 102.3 ± 13.2 | 83.8 ± 3.2 | 82.2 ± 3.9 | 76.9 ± 3.4 |
| Strata 0.1.6 greedy / temperature 1.0 (w9) | 87.0 / 80.5 | 96.0 / 79.1 | 85.0 / 73.9 | 80.4 / 69.9 |

| Hit rate, G | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| old (sw96) | 77.0% | 66.0% | 67.5% | 68.0% |
| new | 80.2% | 68.4% | 69.0% | 70.2% |

- **Greedy with the head: +6.6%, +3.6% and +6.2% at 1K, 32K and 134K.** 250K is flat, within the
  spread, although its hit rate rose by 2.2 points.
- Against Strata's greedy runs (its cache path "is NOT CORRECT", see sw96): behind by 9% at 32K,
  1.5% at 134K and 7% at 250K; ahead at 1K.
- Prefill and reuse are unchanged (the admission acts in decode).
- The rest of the gap is still the warm-up: sw99 puts the ceiling near 90% hits, and the new
  policy reaches 68-70% on this protocol with the head.
