# sw46: tensor-core prefill attention (2026-09-28)

`k_attn_tc` (arch/qwen4exp/blocks.cu) replaces the split-K FP32 attention (`k_attn_part` +
`k_attn_combine`) for QSA sub-batches of 16 or more tokens:
- one CTA per (kv head, token), 4 warps;
- the GQA group's 12 query heads are the M = 16 rows of `mma.m16n8k16` (fp16 in, fp32
  accumulate);
- K/V are gathered from the token's cell list straight into fragments (q8 dequantized exactly to
  fp16, times the block scale);
- online softmax in base 2, no partials.

Decode and verify windows keep the old kernels. `FLASHRT_ATTN_TC=0` turns it off.
One run per arm (`sw46.sh`):

| Prompt | KV | chunks | split-K FP32 | tensor cores |
|---|---|---|---|---|
| 8,192 | fp16 | 4,096 | 1,677.2 tok/s | 1,837.4 tok/s |
| 32,768 | q8 in VRAM | 8,192 | 2,192.5 tok/s | **2,643.7 tok/s** (+21%) |

The greedy tokens after prefill agree for the first 9 of 24 (8K) and 8 of 9 (32K), then diverge.
That is numerics. The KLD gate is in sw47.
