# sw73: linear_multi v1, decode's small BF16 mat-vecs in one launch (2026-09-28)

`linear_multi` multiplies up to four BF16 matrices of one input in one launch, with optional
epilogues. It is used for:
- GDN `ssm_alpha` + `ssm_beta`, with the gates (softplus decay, sigmoid) as the epilogue: one
  launch instead of three, 36 layers;
- the router + `ffn_gate_inp_shexp`: one instead of two, 48 layers;
- the indexer q + k: one instead of two, 12 layers.

Run order: fused first, then `FLASHRT_LINEAR_MULTI=0`.

**KLD**, verify windows of 3, hot set 512 (`fast-win3-hot512.log`): **0.008951** (sw68:
0.009167). Unchanged.

**Teacher-forced decode**, P2 conditions, 6 windows of 128 tokens, means:

| Arm | separate | v1 fused |
|---|---|---|
| 32K plain | 98.3 | 102.6 |
| 32K `--spec 1` | 109.1 (verify 11.94 ms) | 107.4 (verify 12.23 ms) |
| 245K `--spec 1` | 85.7 (verify 15.88 ms) | 84.5 (verify 16.03 ms) |

Mixed: faster on single tokens, slower on 2-token verify windows. `test_linear_multi` (a new
test, decode shapes, weights from VRAM) found the cause. v1 gave each output row one warp: 96
warps for alpha + beta, which is latency-bound. It lost to two MMVF launches at T >= 2, for
example 16.4 against 7.9 µs at T = 8.

**v2** gives each row a block of 4 warps, with the partial sums meeting in shared memory
(`test_linear_multi_v2.txt`):

| shape | T = 1 | T = 2 | T = 4 | T = 8 |
|---|---|---|---|---|
| router + shexp gate (fused / separate, µs) | 6.2 / 6.3 | 6.1 / 7.2 | 6.2 / 10.3 | 8.2 / 12.9 |
| alpha + beta | 4.1 / 3.7 | 4.1 / 4.5 | 4.3 / 6.8 | 6.2 / 7.9 |
| indexer q + k | 6.2 / 6.9 | 6.2 / 7.6 | 6.2 / 10.5 | 8.2 / 13.7 |

- The "separate" column is two back-to-back launches.
- For alpha + beta, the fused kernel also replaces the gates kernel.
- v2 is measured end to end in sw74.
