# sw70: the chunked (WY) GDN delta rule (2026-09-28): exact, but slower on this GPU

`k_gdn_chunk_prep` + `k_gdn_chunk_state` (arch/qwen4exp/blocks.cu, `gdn_delta_prefill`), chunks
of 64 tokens:
- **Per (chunk, head), in parallel:** the cumulative log decay G, A[t][s] = beta_t
  exp(G_t - G_s) k_t.k_s (s < t) and P[t][s] = exp(G_t - G_s) q_t.k_s (s <= t). Then U~ and W by
  forward substitution of (I + A) X = [beta V | beta gamma K].
- **Per (head, 32 value columns), sequential over chunks:**
  - U = U~ - W S0;
  - O = diag(gamma) Q S0 + P U;
  - S0 = gamma_C S0 + K^T diag(exp(G_C - G)) U.
- All fp32.

**`test_gdn`** (`test_gdn.txt`), synthetic inputs of the model's shape (48 heads, 16 key groups,
state 128), against the column kernel:

| T | outputs (relative) | final state | column kernel | chunked |
|---|---|---|---|---|
| 1,000 | 6.1e-7 | 5.8e-7 | 0.42 ms | 2.02 ms |
| 3,000 | 6.2e-7 | 6.5e-7 | 1.24 ms | 5.98 ms |

Correct, but 4.8x slower.

**Why it cannot win in fp32** (`bench_mma.txt`, tools/bench_mma):
- **Cost:** the chunked form needs about 2.4x the recurrence's FLOPs (80K against 33K MAC per
  token and head).
- **The column kernel** already runs at about 27% of the CUDA cores' fp32 peak (~56 TFLOPS).
- **Tensor cores with fp32 accumulation** on this GeForce part: TF32 61 TFLOPS, BF16 122
  (int8: 490 TOPS). TF32 buys nothing over the CUDA cores.
- **A BF16 tensor-core version** at a realistic 40% of peak would reach about 0.2-0.25 µs per
  token against 0.41 now. That saves roughly 0.3-0.4 s at 64K, ~3% of prefill, and brings BF16
  rounding into a recurrent state.

The fp32 version stays as a tested, opt-in baseline (`FLASHRT_GDN_CHUNK=1`); the column kernel
stays the default.
