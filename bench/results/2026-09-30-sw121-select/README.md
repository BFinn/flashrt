# sw121: the QSA block selection on a thread-block cluster (P-5) (2026-09-30)

**Why.** sw120's decode profile at 245K (plain, q8 host KV with the hot set) put `k_idx_select` at
1,245 µs per token: 103 µs per call in each of the 12 QSA layers, 10.7% of GPU time. One CTA per
token ran a 32-bit radix select (4 passes of 8 bits, then a pass to write the cells) over all
61,440 block scores.

**Change** (06a0274, 7e5824c, 2a4f741). One cluster of 8 CTAs per token (`__cluster_dims__`;
sm_90+ thread-block clusters):
- CTA k owns an eighth of the blocks and holds their keys in shared memory for all passes.
- Each pass, every CTA builds its digit histogram, and after one cluster barrier every CTA sums
  all eight through distributed shared memory. So all eight choose the same digit. The histograms
  are double-buffered, so one barrier per pass suffices.
- The final pass counts each CTA's blocks above the threshold and at it. The counts of the CTAs
  before it give each CTA's offsets, so the cells are written in block order, ties taken in block
  order: exactly the single CTA's output.

## Checks

| Check | Result |
|---|---|
| `test_idx_select` (new): 40 token selections, dense to 250K, T = 1, 3, 16, many exact ties, all scores equal | 40 / 40 match a CPU reference |
| Its time at 245K (isolated) | 22.7 µs per call, T = 1 or 3 |
| sw112's fingerprint (7 runs: KLD fast, windows, chunks; teacher-forced window 9 with and without the head; sampled and greedy wikitext) against the build before (0a581c5, `fp-base.txt`) | **identical** |
| ctest | 20 / 20 |

**Speed** (teacher-forced from the saved states, 256 tokens, the build before and after
interleaved, 3 pairs, `ab/`):

| tok/s | before | after | change |
|---|---:|---:|---:|
| 245K, plain | 85.96 ± 0.75 | 94.49 ± 0.14 | **+9.9%** |
| 245K, `--spec 2` | 84.20 ± 0.52 | 87.31 ± 2.78 | +3.7% |
| 32K, plain | 111.19 ± 1.88 | 114.44 ± 0.06 | +2.9% |

The selection sits before the MoE and its CPU misses, outside the miss window, so its time
comes off the token. At 245K that was about 1 ms of 11.6.

## Files

- `sw121.sh`, `sw121.out`; `fp-base.txt`, `fp-new.txt`; `ab/`; `test_idx_select.txt`.
