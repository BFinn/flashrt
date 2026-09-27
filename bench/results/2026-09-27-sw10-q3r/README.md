# Speed work 10: Q3R, a decode layout for Q3_K (2026-09-27). Parked: net loss so far

ggml's Q3_K MMVQ runs at 364-389 GB/s on sm_120 for qwen4exp's large Q3_K matrices
(`attn_gate`, `attn_qkv`, `attn_q`; about 475 MB per token, 1.47 ms). Other types reach
745-780 GB/s. Q3R (`kernels/cuda/q3r.h`) is a lossless split of Q3_K into coalesced planes,
with a float-activation mat-vec for single tokens. `test_gemv` checks it against the exact
dequantized weights: relative L2 2.3e-7.

## Microbenchmark (`bench_q3r.txt`, `bench_gemv --q3r`, 300 calls, rotating copies)

| Shape (K x rows) | ggml Q3_K | Q3R v1 | v2 (2 groups per lane, 2 rows per warp) | v3 (x prepared once, global) | Read-only probe |
|---|---:|---:|---:|---:|---:|
| 2560 x 6144 | 18.6 µs (364 GB/s) | 16.8 | 16.4 (412) | 17.3 | 10.2 (663) |
| 2560 x 10240 | 29.4 (383) | 23.3 | 22.4 (504) | 22.1 | 16.4 (685) |
| 2560 x 12288 | 34.8 (389) | 26.7 | 25.9 (522) | 25.6 | 18.5 (732) |
| 6144 x 2560 | 13.2 (512) | 19.8 | 14.4 (469) | 15.5 | 10.8 (627) |

- **The access pattern is not the limit:** reading the planes alone reaches 627-732 GB/s.
- **The decode arithmetic is the limit.** Bit extraction and int-to-float conversion take about
  6 µs per call. Each warp has only 2-3 iterations, so the loads cannot hide behind the compute.

## End to end (v1, 3 runs at 2K; `db_2k_r*.txt`)

| | tok/s | Hit rate | Q3_K kernel time per token |
|---|---|---:|---:|
| sw8 (ggml MMVQ) | 82.9 / 83.3 / 83.3 | 88.97% | 1,469 µs |
| Q3R copies | 78.7 / 79.4 / 79.4 | 84.79% | 1,156 µs |

- **The Q3R copies cost cache slots.** They add about 480 MB of VRAM, the hit rate falls, and
  the extra misses cost more than the kernel saves.
- **Fast-path KLD with Q3R:** 0.008924 (`kl8k-fast.log`), so it passes. Numerics are not the issue.

## Status and what would make it pay

The copies are now off by default (`GpuWeights::load(..., q3r_copies = false)`). The code and
the benchmark stay. Two changes are needed:
- a dp4a formulation (int8 activations, a layout where one shift and mask yields 4 elements'
  3-bit values in bytes), which should approach the read-only probe;
- a batched Q3R path, so the ggml Q3_K copy can be dropped and no VRAM is duplicated.
