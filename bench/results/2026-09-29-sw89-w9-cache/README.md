# sw89: expert-cache warm-up on window 9's protocol (2026-09-29)

`flashrt_depthbench.py` on window 9's protocol, plain decode, greedy, one run each (`sw89.out`):

| Engine options | 1K | 32K | 134K | 250K |
|---|---|---|---|---|
| defaults (sw87 P, mean of 3) | 91.8 | 73.0 | 70.0 | 67.1 |
| `--cache-prior` (the calibration routing prior) | 87.8 | 73.1 | 71.5 | 67.2 |
| `--swap-budget 32` (default 8) | 95.1 | **82.0** | **76.5** | **73.4** |
| both | 91.8 | 82.7 | 77.0 | 73.4 |
| both, `--mtp --spec 2`, greedy | 101.6 | 81.4 | 80.0 | 74.4 |

- **The routing prior does not help.** The prompt's counts dominate it.
- **A larger swap budget does:**
  - +12% at 32K, +9% at 134K and at 250K;
  - the budget is the number of expert uploads the adaptive cache may have in flight per step;
  - with it, the cache follows the generation's experts sooner.
- sw90 checks the cost on the wikitext runs the default was tuned on.
