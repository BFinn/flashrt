# sw108: the expert cache's warm-up policy, end to end (2026-09-30)

The policy from sw104-sw107: the counts start at 0.03 times the prompt's routing counts, 64 uploads
start per step, and uploads commit a fixed number of steps after they are issued, so runs repeat.
This build commits two steps after issue; sw109 tests one. Measured against sw102, the same engine
before the warm-up work (`sw108.sh`; `sw108-summary.txt`).

**Window 9's protocol with the MTP head** (n = 3):

| Decode tok/s | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| greedy, sw102 | 109.2 ± 0.6 | 87.3 ± 0.2 | 83.1 ± 0.1 | 81.8 ± 1.4 |
| **greedy, sw108** | **118.3 ± 1.6** | **105.4 ± 2.2** | **104.2 ± 1.8** | **92.5 ± 1.5** |
| temperature 1.0, sw101 | 102.3 ± 13.2 | 83.8 ± 3.2 | 82.2 ± 3.9 | 76.9 ± 3.4 |
| **temperature 1.0, sw108** | **110.2 ± 9.8** | **106.6 ± 5.6** | **100.2 ± 1.9** | **92.9 ± 4.1** |
| Strata 0.1.6 greedy / temperature 1.0 (w9) | 87.0 / 80.5 | 96.0 / 79.1 | 85.0 / 73.9 | 80.4 / 69.9 |

| | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| hit rate, greedy: sw102 → sw108 | 80.2 → 84.2% | 67.4 → 80.3% | 70.0 → 81.7% | 70.3 → 81.0% |
| draft acceptance, greedy | 52.1% | 56.4% | 62.9% | 56.7% |

- **Greedy: +8%, +21%, +25% and +13%.** flashrt is now ahead of Strata's greedy runs at every depth
  (+36%, +10%, +23%, +15%) and of its temperature-1.0 runs by 33-37%. Strata's build warns that its
  cache path "is NOT CORRECT"; its timings are real.
- Prefill, reuse and time to the first token are unchanged (the policy acts in decode).

**The agentic session** (greedy, 2 runs): 114.3 / 114.2 tok/s at 78.7% hits, against 108.3 at
76.1% with the old settings (sw105) and 125.4 at 83.5% on sw105's timing-dependent build. Greedy
agent runs are not teacher-forced: this build generated 3,244 tokens, sw105's 3,648-3,861. Turns 0
and 1 have the same prompts and lengths in every run, and hit 1-2 points below the
timing-dependent build: 73.2% / 54.6% against 74.0% / 56.6% (old settings: 70.4% / 47.4%). That
build often committed an upload a step sooner; sw109 commits at the next step.

**Server smoke:** all checks pass (`smoke.txt`).
