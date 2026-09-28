# sw51: expert grouping without ggml's helper; mid-layer hc combine fused into the norm (2026-09-28)

Two changes, both meant to be bit-identical:
- **`moe_prepare` groups the (token, slot) pairs by expert itself**, in three kernels:
  - per-block expert histograms (integer atomics);
  - a scan;
  - a stable placement, one warp per 512 tokens (`__match_any_sync`).

  The outputs are the same, including token order within each expert. ggml's `mm_ids_helper`
  had one warp per expert scan all T x 16 slots: 0.67 s at 64K. `mmid.cu` is no longer vendored.
- **`hc_combine_mix`:** in prefill, the combine after the mixer is fused into the FFN mix's RMS
  norm (`k_hc_combine_norm4`, one block per token), which saves a pass over x.

`test_gemm` passes, including a new case of 1,200 tokens over all 512 experts. q8 KV, chunks of
8,192, one run each:

| Prompt | sw50 | now | greedy tokens after |
|---|---|---|---|
| 32,768 | 3,530.8 tok/s | **3,689.4 tok/s** | identical to sw50 |
| 65,536 | 3,505.7 tok/s | **3,655.2 tok/s** | identical to sw50 |

**64K profile** (`p64k_cuda_gpu_kern_sum.csv`): 16.9 s of kernels (sw48: 19.4 s; sw41: 26.9 s):

| Kernel | time |
|---|---|
| expert MMQ (Q2_0) | 5.42 s |
| Q3_K MMQ | 1.23 s |
| GDN delta rule (v3) | 0.97 s |
| attention (`k_attn_tc`) | 0.95 s |
| hc gated mean / combine+norm / norm / combine | 0.67 / 0.59 / 0.42 / 0.36 s |
| planar-to-ggml expert conversion | 0.65 s |
| IQ4_XS MMQ | 0.60 s |
| cuBLAS BF16 GEMMs (hc projections, router) | 1.25 s |
| expert combine / SwiGLU | 0.45 / 0.29 s |
| indexer scores (`k_idx_scores_tc`) | 0.16 s (was 1.65) |
| expert grouping | below 0.1 s (was 0.67) |
