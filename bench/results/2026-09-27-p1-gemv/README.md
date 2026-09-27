# P1: GPU dense mat-vec over vendored ggml kernels (2026-09-27)

RTX 5080 16 GB (sm_120, 64 MB L2, ~960 GB/s), CUDA 12.9. `kernels/cuda/ggml_gemv` wraps
ggml's MMVQ (quantized weights x Q8_1 activations), MMVF (F32/F16/BF16 weights) and Q8_1
quantization kernels, vendored unmodified from llama.cpp 187664b (MIT, `third_party/ggml`).

## Correctness (`test_gemv.txt`)

All 13 weight types in qwen4exp's dense tensors pass, at 2,560 and 6,144 columns, 384 rows,
for 1-4 tokens. The reference is the exact dequantized weights from the llama.cpp dev tree's
libggml (`ggml_quantize_chunk` + `to_float`), times random activations, in double precision.
- F32 / F16 / BF16: relative L2 about 1e-7.
- Quantized types: 0.5% (Q2_0, Q8_0, IQ4_NL, IQ4_XS, Q3_K-Q6_K) or 1.2-1.6% (Q4_0, Q5_0).
  This is the Q8_1 activation rounding that llama.cpp's GPU path also applies.

## Speed (`bench_gemv.txt`, 300 calls per case, rotating over >= 256 MB of copies so reads come from VRAM)

| Tensor (type) | Shape (cols x rows) | MB | µs, 1 token | GB/s | µs, 4 tokens |
|---|---|---:|---:|---:|---:|
| attn_qkv, GDN (IQ4_XS) | 2560 x 10240 | 13.9 | 17.8 | 782 | 20.6 |
| attn_gate (Q4_K) | 2560 x 6144 | 8.9 | 11.9 | 745 | 14.9 |
| ssm_out (Q5_K) | 6144 x 2560 | 10.8 | 14.2 | 761 | 15.1 |
| attn_q, QSA (Q2_0) | 2560 x 12288 | 8.9 | 12.0 | 735 | 17.9 |
| attn_output (Q6_K) | 6144 x 2560 | 12.9 | 16.8 | 766 | 17.8 |
| hc_*_down (BF16) | 10240 x 320 | 6.6 | 9.0 | 729 | 11.2 |
| hc_*_up (BF16) | 320 x 10240 | 6.6 | 13.3 | **494** | 23.6 |
| ffn_gate_inp, router (BF16) | 2560 x 512 | 2.6 | 4.7 | 558 | 6.8 |
| ffn_up_shexp (Q3_K) | 2560 x 640 | 0.7 | 4.3 | 165 | 4.2 |
| ffn_down_shexp (Q4_0) | 640 x 2560 | 0.9 | 3.3 | 283 | 4.3 |
| output head (Q5_K) | 2560 x 248320 | 437 | 483 | 905 | 544 |

- **Large matrices reach 735-782 GB/s** (about 80% of peak) including the activation
  quantize. The head reaches 905 GB/s. Four tokens cost 1.1-1.5x one token, which is what
  makes a speculative verify window cheap.
- **Small matrices take 3-5 µs each: latency, not bandwidth.** They need CUDA-graph capture
  and fusion.
- **The thin BF16 hyper-connection up-projection (K = 320) is inefficient** in MMVF, at
  494 GB/s. It is a candidate for a custom kernel.
- **Estimate, not measured:** 3.47 GB of dense weights at about 760 GB/s is about 4.6 ms per
  token, plus the head (0.48 ms) and small-kernel latency. Strata's dense GPU path measured
  7.65 ms per token without the head (P0 window C, `--gpu-stages`).
- **Early-run note:** before copy rotation, repeated calls on one matrix read from L2 and
  showed up to 1,800 GB/s. Any GPU microbenchmark here must defeat the 64 MB L2.
