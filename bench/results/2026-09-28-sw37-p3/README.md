# sw37: KLD of logits straight from prefill chunks (2026-09-28)

`fr_kld --ctx 8192 --chunks 2 --prefill-chunk C` (no `--fast`): every scored logit comes from a
prefill chunk (MMQ dense layers, streamed experts, QSA selection in sub-batches), fp16 KV.

| Chunk | KLD mean | median | p99 | same top-1 | PPL ratio |
|---|---|---|---|---|---|
| 1,024 | **0.008492** | 0.001208 | 0.1003 | 96.87% | 1.00324 |
| 4,096 | **0.008559** | 0.001164 | 0.1132 | 96.89% | 0.99948 |

At the noise floor (reference path 0.0089; llama.cpp against itself 0.0078-0.0086).
