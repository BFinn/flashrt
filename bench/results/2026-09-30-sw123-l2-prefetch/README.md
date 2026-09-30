# sw123: L2 prefetch of the next hc mix during the miss wait (P-5): rejected (2026-09-30)

**Idea.** In a decode layer, `k_moe_combine_db` waits for the CPU misses with the GPU otherwise idle
(sw120 at 32K: 11.5 µs per layer on average). The next layer's first hc mix reads about 10 MB of
weights from DRAM (`k_hc_up_mix2` 15.3 µs, above its bandwidth floor). Prefetching those weights into
L2 during the wait should speed up the mix.

**Tried** (e2a7567, 3f78eb9; `FLASHRT_L2_PREFETCH`, same binary, modes rotated; teacher-forced
from the saved states, 256 tokens, n = 3; `sw123.out`, `sw123b.out`):

| tok/s | off | 1: bulk prefetch (`cp.async.bulk.prefetch.L2`), attention mix | 2: bulk, both mixes | off (2nd run) | 3: per-line hints (`prefetch.global.L2`) over 168 CTAs |
|---|---:|---:|---:|---:|---:|
| 245K plain | 95.87 | 90.92 (−5.2%) | 86.36 (−9.9%) | 94.30 | 93.61 (−0.7%) |
| 32K plain | 111.34 | 102.79 (−7.7%) | 95.76 (−14.0%) | 114.82 | 110.87 (−3.4%) |
| 32K `--spec 2` | 104.80 | 102.82 (−1.9%) | 97.07 (−7.4%) | 107.26 | 103.22 (−3.8%) |

**Why** (nsys at 32K plain, `p_m*_cuda_gpu_kern_sum.csv`):
- The prefetch reaches L2: with mode 1, `k_hc_up_mix2` fell from 15.3 to 11.1 µs per call on average
  (400 µs per token).
- But the bulk-prefetch kernel took 29 µs per call: it does not retire before its transfers are
  done, so it holds the stream longer than the average wait it was meant to fill (the combine's
  wait fell from 11.5 to 6.3 µs, the rest of the prefetch time added to the token).
- The per-line hints end quickly (4.8 µs), but the mix barely gained (14.4 µs), and the combine's
  wait doubled (22.5 µs). The extra traffic appears to delay the step. The cause is not isolated;
  one guess is the mapped-memory polling of the doorbell.

**Conclusion.** Not adopted; the code is removed. A side stream could hold the bulk prefetch without
blocking the main stream, but the per-line result suggests the traffic itself costs more than the
mix gains.

The fingerprint with mode 1 (`fp-mode1.txt`) equals sw122's: a prefetch never changes outputs.
