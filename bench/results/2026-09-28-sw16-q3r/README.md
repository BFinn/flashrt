# Speed work 16: Q3R v2, planar dp4a Q3_K for tall matrices (2026-09-28)

`kernels/cuda/q3r.h` stores Q3_K weights (q in 0..7, weight = d · sc · (q − 4)) in a layout that
one shift, mask and OR per 4 elements turns into dp4a operands:
- a 2-bit plane;
- a packed high-bit plane;
- int8 group scales;
- fp16 super-block scales.

Activations are int8 per 64 values. Up to 8 tokens go through one pass, so the same kernel also
serves prefill. The conversion happens in place, in slots sized for the larger layout, so no
VRAM is duplicated (the Q3R v1 mistake, sw10).

**Which matrices:** only tall Q3_K matrices (≥ 4096 rows, K ≤ 4096: `attn_gate`, `attn_qkv`,
`attn_q`, about 430 of the 475 MB of Q3_K read per token). There it beats ggml's MMVQ. On
`ssm_out` (K = 6144) and the small shared-expert matrices ggml stays faster, so those stay Q3_K.

## Microbenchmark (`bench_q3r.txt`, 300 calls, VRAM-resident)

| Shape (K x rows) | ggml, 1 token | Q3R, 1 token | ggml, 4 tokens | Q3R, 4 tokens |
|---|---:|---:|---:|---:|
| 2560 x 6144 | 18.6 µs (364 GB/s) | 12.3 µs (549 GB/s) | 18.5 | 20.6 |
| 2560 x 10240 | 29.4 (383) | 18.5 (611) | 29.0 | 24.1 |
| 2560 x 12288 | 34.8 (388) | 23.0 (587) | 34.3 | 33.7 |
| 6144 x 2560 (stays ggml) | 13.1 | 16.4 | 15.1 | 29.4 |
| 2560 x 640 (stays ggml) | 4.3 | 8.2 | 4.2 | 14.4 |

## Checks

- **`test_gemv`:** Q3R against the exact dequantized weights at 1 and 4 tokens, relative L2
  0.57-0.65% (int8 activations, like the other quantized types).
- **Fast-path KLD:** 0.008744, median 0.00114, same top-1 96.73%, PPL ratio 0.9989. The gate holds.

## Speed

| Arm | Runs | tok/s |
|---|---:|---|
| 2K, 256 tokens (sw14: 88.75 / 88.38 / 88.53) | 3 | **89.82 / 88.50 / 89.67** |
| 245,760 from the sw15 state, 3 windows of 128 (sw15, PCIe off: 57.40 / 55.05 / 60.20) | 1 | **57.94 / 57.17 / 58.28** |

- **Kernel time (nsys at 2K):** the ggml Q3_K MMVQ took 1,470 µs per token for all Q3_K matrices;
  Q3R now takes 738 µs for the tall ones, and total GPU kernel time is 9.77 ms per token (10.27
  in sw14).
- **The wall-clock gain is small.** The remaining token time is dominated by the GPU waiting on
  CPU misses (`k_moe_combine_db`, 1.24 ms) and by gaps between kernels.
