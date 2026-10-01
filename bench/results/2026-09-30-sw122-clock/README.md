# sw122: the hot set's CLOCK sweep in parallel (P-5) (2026-09-30)

**Why.** `k_hot_select` (sw120: 22.4 µs per call at 245K, 12 per token) marks the blocks the
selection chose and then assigns victims for up to 1,024 missed blocks, with one thread stepping
the CLOCK hand slot by slot.

**Change** (the commit after 2a4f741). The sweep goes a chunk of the ring at a time (up to 1,024
slots, no slot twice). A slot is a candidate when not referenced since the hand last passed and not
pinned this step. A block scan ranks the chunk's candidates in clock order; the first `need` take
the missed blocks in order; the hand stops after the last one taken, and the slots passed before it
lose their reference bit. That is the same victims, pairing and hand as the serial sweep, including
its give-up after 2B slots without a victim. Where a block sits never changes a value, so outputs
cannot change either way.

| Check | Result |
|---|---|
| sw112's fingerprint against sw121's build (2a4f741) | **identical** (`fp-new.txt`) |
| ctest | 20 / 20 |
| nsys, 245K plain (`p_plain_245_cuda_gpu_kern_sum.csv`) | `k_hot_select` 22.4 → **4.4 µs** per call; `k_idx_select` 103 → 25.5 µs (sw121) |

**Speed** (teacher-forced from the saved states, 256 tokens, 3 pairs interleaved, `ab/`):

| tok/s | before | after | change |
|---|---:|---:|---:|
| 245K, plain | 93.83 ± 0.92 | 95.57 ± 0.57 | +1.9% |
| 245K, `--spec 2` | 89.59 ± 0.58 | 87.82 ± 4.58 | −2.0% (one run at 82.7; the other two 89.4, 91.4) |
| 32K, plain | 112.73 ± 2.37 | 114.63 ± 0.60 | +1.7% |

The plain gain matches the time saved (18 µs × 12 per token). Part of what sw121 and sw122 save
before the MoE reappears as waiting in `k_moe_combine_db` (1,104 → 1,434 µs per token at 245K): the
CPU misses are the critical path more often.

## Files

`sw122.sh`, `sw122.out`, `fp-new.txt`, `ab/`, the profile's kernel summary.
