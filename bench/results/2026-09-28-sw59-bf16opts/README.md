# sw59: deferred layer-end combine; BF16 expert outputs and hc gate (2026-09-28)

**The layer-end combine is deferred.** In a prefill chunk, the hyper-connection combine that ends
a layer goes into the next layer's first mix, fused into its norm (`combine_pending_` in
ForwardRef). It is done on its own only before a PLE block and before the head. This is exact:
the greedy tokens are identical to sw58's.

**Two BF16 options:**
- `FLASHRT_MOE_YD16`: moe_q2's per-slot expert outputs (T x 10 x 2,560 values) in BF16. This
  halves the down kernel's writes and the combine's reads, and frees 51 KB per token of chunk
  memory.
- `FLASHRT_HC_GATE16`: the hc up product writes its gate in BF16 (`gemm::gemm_bf16_out`), and
  the gated mean reads it that way.

32,768 tokens, q8 KV, automatic chunks (16,384), one run each:

| Arm | prefill | chunk buffers | greedy tokens after |
|---|---|---|---|
| deferred combine (sw58: 5,376.9) | 5,431.6 tok/s | 8,206 MiB | = sw58 |
| + BF16 expert outputs | 5,524.9 tok/s | 7,406 MiB | differ from token 9 |
| + BF16 gate | 5,514.1 tok/s | 8,206 MiB | differ from token 9 |
| + both | **5,605.8 tok/s** (+3.2%) | 7,406 MiB | = sw58 |

**KLD with both** (`fr_kld --ctx 8192 --chunks 2 --prefill-chunk 1024`):

| KV | both BF16 options | without (sw58) |
|---|---|---|
| fp16 | 0.008708 (96.85%) | 0.008405 |
| q8 | 0.008675 (96.92%) | 0.008685 |

The two differences have opposite signs (fp16 +0.0003, q8 0.0000), unlike per-64 activation
scales in sw57 (+0.0003 and +0.0007). Both options are now **on by default**; `=0` turns each off.
