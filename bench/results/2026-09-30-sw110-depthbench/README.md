# sw110: window 9's protocol on the final build (2026-09-30)

sw96's protocol and arms (`sw110.sh`: the same token ids, one growing conversation, 384 tokens per
depth, a fresh engine per run, 5 runs per arm in rotating order), on the build that ends the
expert-cache warm-up work (sw99-sw109). The cache admits at count 1 and 1.2 times the weakest
resident, starts from 0.03 times the prompt's routing counts, starts up to 64 uploads per step
and commits them at the next step. The head's prompt pass runs in chunk calls (sw102).
`sw110-summary.txt` is `bench/depthsum.py` on `sw110.out`.

| Decode tok/s (mean ± sd, n = 5) | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P, no head, greedy | 102.2 ± 1.1 | 94.8 ± 1.1 | 87.5 ± 0.3 | 82.8 ± 0.3 |
| G, head, greedy | 131.9 ± 2.8 | 107.1 ± 2.1 | 99.1 ± 1.0 | 87.0 ± 0.8 |
| S, head, temperature 1.0 | 119.5 ± 10.7 | 108.3 ± 1.6 | 99.8 ± 4.3 | 91.4 ± 8.1 |
| sw96 (P / G / S) | 94.5 / 105.2 / 100.7 | 83.1 / 84.0 / 81.0 | 78.3 / 78.8 / 79.5 | 73.4 / 75.2 / 77.3 |
| Strata 0.1.6 greedy / temperature 1.0 (w9) | 87.0 / 80.5 | 96.0 / 79.1 | 85.0 / 73.9 | 80.4 / 69.9 |
| llama.cpp greedy (w9) | 37.4 | 37.2 | 32.8 | 30.7 |

| Hit rate | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P: sw96 → sw110 | 82.1 → 89.5% | 75.0 → 88.6% | 76.6 → 88.7% | 76.6 → 89.2% |
| G: sw96 → sw110 | 77.0 → 84.6% | 66.0 → 81.1% | 67.5 → 82.1% | 68.0 → 81.8% |

- **Over sw96:** P +8%, +14%, +12%, +13%; G +25%, +28%, +26%, +16%; S +19%, +34%, +26%, +18%.
- **Against Strata greedy:** G ahead by 52%, 12%, 17%, 8%; P ahead at 1K, 134K and 250K and level at
  32K. At temperature 1.0, S is ahead by 31-48%. Strata's build warns that its cache path "is NOT
  CORRECT"; its timings are real.
- **Against llama.cpp:** P 2.5-2.7x, G 2.8-3.5x.
- Prefill, reuse (32,768 / 134,004) and time to the first token are unchanged (250K: 22.3 s without
  the head, 24.1 s with it).
- Draft acceptance with the head, greedy: 64.3%, 55.2%, 57.0%, 49.7%. At 250K the head gains
  less (87.0 against 82.8 without it).

GPU at the rows' ends: 2,797-2,820 MHz, 48-53 °C, 157-192 W.
