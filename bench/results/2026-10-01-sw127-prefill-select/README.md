# sw127: the QSA selection for prefill sub-batches, by depth (P-5) (2026-10-01)

**Why.** sw124's window 9 run showed prefill 9.5-12% slower than sw119's (32K: 5,665 → 5,060 tok/s of
new tokens, the 250K prompt 22.3 → 24.7 s). The only engine change between them was P-5's first
round. sw121's 8-CTA selection also runs on prefill's 128-token sub-batches, and sw121 measured
decode only.

**Measured in isolation** (`test_idx_select`, now with any cluster size; `test_idx_select.txt`).
The time per 128-token sub-batch, by CTAs per token:

| position | 8 (sw121) | 4 | 2 | 1 |
|---|---:|---:|---:|---:|
| 32K | 274.9 µs | 72.2 | 45.9 | **33.0** |
| 64K | 298.1 | 90.9 | **67.9** | 68.7 |
| 128K | 330.3 | **117.5** | 119.2 | 152.5 |
| 192K | 367.9 | **167.9** | 215.0 | 222.1 |
| 245K | 398.0 | **212.5** | 292.8 | 272.6 |

The sub-batch's 128 tokens fill the GPU already. Eight CTAs per token make 1,024 CTAs of 1,024
threads, and clusters of 8 schedule about 10 at a time: many waves of fixed work (5 radix passes,
each with a cluster barrier). At depth, one CTA's 61K keys no longer fit in its shared memory
and are read again each pass, and 4 CTAs per token balance the two.
Decode keeps 8 (T = 1: 24 µs against 103 µs for one CTA).

**Change** (38ac581, d4aeb23, 5e9860d). `k_idx_select` is a template on the cluster size.
- Prefill sub-batches of 64 tokens or more, outside graphs, take 1 CTA per token below 24,576
  blocks (98K positions) and 4 above.
- Decode and verify windows keep 8.
- `FLASHRT_SELECT_CL1=0` keeps 8 everywhere.
- The output does not depend on the cluster size: `test_idx_select` checks sizes 8, 4, 2 and 1
  (160 / 160 token selections equal the CPU reference).

**Prefill** (same binary, toggle rotated, 2 rounds; q8 host KV, hot set 4096, automatic chunks;
`sw127.out`, `runs/`):

| tok/s | 8 CTAs per token | 1 or 4 by depth | change |
|---|---:|---:|---:|
| 32K | 5,370 / 5,355 | 5,987 / 5,975 | **+11.6%** |
| 128K | 5,294 / 5,297 | 5,932 / 5,938 | **+12.0%** |
| 245K | 5,125 / 5,121 | 5,714 / 5,706 | **+11.5%** (48.0 → 43.0 s) |

**Fingerprint** against sw126's new build: **identical** (`fp-new.txt`).

**Lesson.** A kernel shared by decode and prefill needs both paths measured. sw121's A/B was
teacher-forced decode from saved states, which run no prefill at all.
