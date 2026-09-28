# sw56: KLD gate for moe_q2 (2026-09-28)

`fr_kld --ctx 8192 --chunks 2`, with moe_q2 on (the default):

| Run | KLD mean | same top-1 |
|---|---|---|
| fp16 KV, `--prefill-chunk 1024` (logits from chunks) | 0.008385 | 96.91% |
| q8 KV, `--prefill-chunk 1024` | 0.008176 | 96.79% |
| `--fast --prefill-chunk 2048 --kv-hot 512` (chunked prefill, then decode) | 0.008560 | 96.87% |

All at the noise floor (sw50, MMQ path: 0.0085-0.0087).
