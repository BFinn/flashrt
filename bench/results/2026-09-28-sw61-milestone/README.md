# sw61: prefill milestone after the expert kernel and the traffic work (2026-09-28)

Every default of this round is on:
- moe_q2 with activation scales per 32;
- BF16 per-slot expert outputs and BF16 hc gate;
- the hc fusions (1/rms instead of xn, inject in the norm block, deferred layer-end combine);
- automatic chunk length.

One run each:

| Run | chunk | prefill | before this round (sw54) |
|---|---|---|---|
| 32,768, q8 KV | 16,384 | **5,580.1 tok/s** (5.9 s) | 4,005.6 |
| 65,536, q8 KV | 16,384 | **5,628.6 tok/s** (11.6 s) | 3,975.4 (sw52) |
| 245,760, q8 KV in VRAM | 16,384 (fits now) | **5,308.9 tok/s** (46.3 s) | 3,623.2 (11,264) |
| 245,760, host KV + hot set 4,096 (mirror) | 15,360 | **5,169.6 tok/s** (47.5 s) | 3,549.0 (10,240) |

Decode after prefill (16 tokens, plain): 95.6 (32K), 61.2 (64K), 64.5 / 66.5 tok/s (245K). These
are single short windows and in the normal range.

**Fast-path KLD after a chunked prefill** (`--fast --prefill-chunk 2048 --kv-hot 512`): 0.008394
(96.58%). The chunk-logit KLD of this build is in sw62.

**Engine** (`engine_smoke.txt`: 32,000-token prompt, `--mtp ... --spec 1 --ctx 65536`,
temperature 1.0):
- r1 prefilled in **7.9 s** (sw54: 10.2 s), including the head's catch-up and the cache rebuild,
  with chunks of 16,128;
- decode 98-110 tok/s;
- prefix reuse and cancellation as before.
