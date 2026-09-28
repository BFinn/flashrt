# sw60: GDN column kernel, chain depth and lanes per column (2026-09-28): no gain

The prefill GDN kernel (`k_gdn_delta_col`) takes about 0.41 µs per token-layer (about 1,100
cycles). It has only 384 warps (48 heads x 128 columns, 2 columns per thread, 4 lanes per
column), so I took it for latency-bound on the per-token chain (2 accumulators of 16 dependent
FMAs per column sum, then shuffles). Two changes, at 32K prefill under nsys (q8 KV, automatic
chunks), one run each:

| Variant | GDN kernel time (72 calls) | prefill |
|---|---|---|
| before (2 accumulators, 4 lanes per column; sw58 profile, scaled) | ~0.48 s | |
| 4 accumulators, 4 lanes per column (`lpc4`) | 0.505 s | 5,548.5 tok/s |
| 4 accumulators, 8 lanes per column (`lpc8`: twice the warps) | 0.640 s | 5,442.3 tok/s |

Neither helps, so the chain is not the bound. 4 lanes stays the default (`FLASHRT_GDN_LPC=8`
keeps the variant). The kernel runs at about 3x its instruction-issue estimate. Without
performance counters (admin-only on this box) the cause is open; the chunked (WY) form of the
delta rule remains the structural fix.

**Also measured:** how much adjacent tokens' QSA selections overlap, 64K prefill, 24 samples
(a temporary diagnostic, not committed). The union of G consecutive tokens' cell lists, relative
to one list:

| G | union / one list |
|---|---|
| 2 | 1.30 (1.24-1.41) |
| 4 | 1.72 (1.56-2.04) |
| 8 | 2.28 (1.98-3.03) |
| 16 | 3.08 (2.51-4.48) |

An attention CTA serving 4 tokens from their union would read K/V about 2.3x less often. But
building the union per group costs microseconds, which is about what a whole attention CTA takes
now (~0.6 µs per token and head). Not pursued.
