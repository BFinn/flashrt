# sw68: hc down matrices as Q8P, the default (2026-09-28)

`FLASHRT_HC_Q8=down` (the default: the down matrices as Q8P, the up ones BF16) against all BF16
(`=0`).

**KLD** (`fr_kld --ctx 8192 --chunks 2`):
- logits from prefill chunks, fp16 KV: **0.008670** (96.83%);
- verify windows of 3, hot set 512: **0.009167** (96.79%; sw33, BF16: 0.009232).

Unchanged.

**Speed**, teacher-forced, P2 conditions, from the saved states, 6 windows of 128 tokens, one run
each:

| Arm | BF16 | down as Q8P (+214-284 slots) |
|---|---|---|
| 32K plain | mean 97.4 | mean **99.6** (+2.3%) |
| 32K `--spec 1` | mean 106.0 (verify 12.32 ms) | mean 106.4 (verify 12.29 ms) |
| 245K `--spec 1` | mean 80.8 (verify 16.90 ms) | mean **82.9** (+2.6%; verify 16.46 ms) |

A small, free gain. Converting the up matrices too would add about 5% at 32K, at +0.0011 KLD
(sw66, sw67); that stays opt-in (`FLASHRT_HC_Q8=1`).
