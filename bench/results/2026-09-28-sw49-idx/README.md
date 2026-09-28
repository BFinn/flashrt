# sw49: tensor-core indexer scores (2026-09-28)

`k_idx_scores_tc` (arch/qwen4exp/blocks.cu) scores QSA sub-batches of 16 or more tokens (4
indexer heads, dim 128):
- a CTA takes 32 tokens: their 128 (token, head) query rows are the M side of fp16
  `mma.m16n8k16` with fp32 accumulation;
- it walks tiles of 64 pooled keys, converted to fp16 in shared memory;
- relu and the sum over heads are fused (two shuffles).

Decode keeps `k_idx_scores128`. `FLASHRT_IDX_TC=0` turns it off. This run also has the column
GDN with 8 warps per block, which turned out no faster than 4 (see sw50).

q8 KV, chunks of 8,192, one run each:

| Prompt | FP32 scores | tensor cores |
|---|---|---|
| 32,768 | 2,898.8 tok/s | **3,001.8 tok/s** (+3.6%) |
| 65,536 | 2,983.8 tok/s | **3,197.0 tok/s** (+7.1%) |

The gain grows with depth, as scoring is linear in the number of pooled blocks.

**KLD gate:** `fr_kld --ctx 8192 --chunks 2 --prefill-chunk 1024`, logits from prefill chunks.
The fp16 rounding of queries and keys flips near-ties at the selection boundary, and the KLD
does not move:

| KV | KLD mean | same top-1 |
|---|---|---|
| fp16 | 0.008228 | 96.98% |
| q8 | 0.008401 | 96.81% |
