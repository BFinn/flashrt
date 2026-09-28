# sw62: KLD of the sw61 build (2026-09-28)

`fr_kld --ctx 8192 --chunks 2 --prefill-chunk 1024` (logits from prefill chunks), every prefill
default of sw61:

| KV | KLD mean | same top-1 |
|---|---|---|
| fp16 | 0.008545 | 96.56% |
| q8 | 0.008983 | 96.73% |

The only change from sw59's "both" build (fp16 0.008708, q8 0.008675) is the GDN kernel's 4
accumulators per sum, a pure reordering of the sums. So +0.0003 on q8 is the spread that small
numeric changes cause. This session's q8 values: 0.0082-0.0090. That variant gave no speed
(sw60), so the previous 2-accumulator kernel is restored after this run.
