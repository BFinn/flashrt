# Speed work 2: split-K flash-decode attention (2026-09-27)

`k_attn_dense` (one block per query head; each thread took whole cells, then one thread per
output dimension looped serially over every cell) is replaced by `k_attn_part<G>` and
`k_attn_combine`:
- **Partials:** one block per (64-cell split, KV head, token) handles the 12 query heads that
  share the KV head. K rows are read coalesced by one warp per cell, and V is read coalesced
  by one thread per dimension.
- **Combine:** one block per (token, head) merges the partials with the log-sum-exp rule, then
  applies the sigmoid output gate.

Arm: `fr_bench`, 2,048-token wikitext prompt (`p0c-20260927/wiki.prompt_ids.txt`), 128 greedy
decode tokens on the fast path, VRAM expert cache from the prompt's routing counts (8,059
slots, 32.8% of experts), 8 CPU workers. Script: `sw2.sh`.

## Correctness

`fr_parity qsa` (tolerance 2e-3 relative L2; the misses come from llama.cpp's FP16
flash-attention accumulation):

| Dump | Checks | Over tolerance, old kernel | Over tolerance, new kernel | Worst, new kernel |
|---|---:|---:|---:|---:|
| `ar65` | 780 | 2 | 2 | 2.3e-3 |
| `long2216` | 26,592 | 234 | 209 | 5.0e-3 (unchanged) |

Indexer selection is unchanged at 99.948% of cells identical. The greedy tokens are identical to
the reference path and to the pre-change fast path.

## Speed (3 runs)

| Run | tok/s | Hit rate | Misses |
|---|---:|---:|---:|
| 1 | 55.36 | 86.30% | 8,418 |
| 2 | 61.23 | 88.17% | 7,269 |
| 3 | 55.58 | 87.30% | 7,800 |
| Before (1 run, `k_attn_dense`) | 42.75 | 87.19% | 7,873 |

## nsys decode profile (64 tokens at 2K)

- **Attention:** 7,145 µs per token → 234 µs per token (`k_attn_part<12>`), plus the combine
  kernel.
- **GPU kernel time:** 13.1 ms per token, against 16-18 ms per token wall time.
- **Host:** `cudaEventSynchronize` (the 48 routing waits) takes 7.4 ms per token, and kernel
  launches take 4.4 ms per token.
- **Top GPU kernels now:** `k_gdn_delta` 1.93 ms, BF16 mat-vecs 1.78 + 1.43 ms, Q4 MMVQ
  1.46 ms, MoE grouped mat-vecs 0.66 + 0.62 ms.
- **Misses are on the critical path:** tok/s follows the miss count, and the hit rate varies
  run to run. The CPU miss time is the next thing to measure.
