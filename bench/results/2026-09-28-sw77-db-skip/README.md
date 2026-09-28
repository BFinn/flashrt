# sw77: tokens without CPU misses skip the doorbell round trip (2026-09-28)

**Before:** every MoE layer of a decode step did a full host round trip:
- `k_route` copied x (10 KB) into the mapped mailbox and fenced;
- the host saw the flag, and for zero-miss layers memset a zero CPU part and raised done;
- `k_moe_combine_db` polled for done and read the (zero) CPU part back over PCIe.

At 32K, 61% of layers have no CPU miss (sw76: combine median 7.7 µs, `k_route` 8.0 µs).

**Now:**
- `k_route` also writes each token's miss count to device memory (`miss_n`).
- A token without misses skips the x copy, the wait and the mailbox read.
- The host still serves every layer (routing statistics), and `doorbell_end_token` still waits
  for the whole step, so the GPU cannot overtake the host between steps.
- `FLASHRT_DB_SKIP=0` restores the full round trip.

**KLD**, verify windows of 3, hot set 512: **0.008931**, identical to sw75: the skipped CPU part
is exactly zero.

**Teacher-forced decode**, P2 conditions, 6 windows of 128 tokens (skip off ran first):

| Arm | full round trip | skip | |
|---|---|---|---|
| 32K plain | mean 105.7 | mean **106.4** | +0.7% |
| 32K `--spec 1` | mean 110.4 (verify 11.79 ms) | mean **112.4** (verify 11.56 ms) | +1.8% |
| 245K `--spec 1` | mean 87.0 (verify 15.66 ms) | mean 87.0 (verify 15.65 ms) | 0 |

- **Smaller than the ~0.3 ms per token estimated** from the kernel times: those kernels overlap
  with other work less than their durations suggest.
- **At 245K** most layers have a miss in at least one of the window's tokens.
- **Kept:** free, and exact.
