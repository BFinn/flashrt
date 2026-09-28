# sw75: the decode hyper-connection kernels, v2 (2026-09-28)

The decode hc mix (97 per token: norm, down + inject, up, silu gate, gated mean) ran at about 500
GB/s, against an 11.4 µs floor per mix (10.1 MB of weights at ~890 GB/s). A new test,
`test_hc_decode`, checks it against a double-precision CPU reference and times it with the
weights coming from VRAM.

The changes:
- **`k_hc_down2`:**
  - a warp takes 2 rows, so each value of x read from shared memory serves 2 weights;
  - each warp loads all its weights before the norm preamble;
  - the 1 / rms is applied after the dot product (W (x * w_norm) * inv), so the block needs one
    barrier instead of 2 per token.
- **`k_hc_up_mix2`:**
  - rank fixed at compile time (320);
  - launched as a programmatic dependent (PDL) of the down kernel, with the down kernel
    triggering at its start: the up blocks are scheduled while down runs;
  - for one token it loads its weights before the dependency wait, for windows after its
    preamble (earlier loads were slower there);
  - at most 64 registers, so its 320 blocks fit in one wave (at 80 registers they took two,
    which first made windows slower).
- **`FLASHRT_HC_DOWN2=0` / `FLASHRT_HC_UP2=0`** bring back the old kernels.

**`test_hc_decode`** (`test_hc_decode.txt`; relative error about 1e-7 in both versions):

| tokens | old | v2 | nsys, old: down / up |
|---|---|---|---|
| 1 | 20.2 µs per mix (501 GB/s) | **17.1** (590 GB/s) | 7.9 / 9.2 µs |
| 2 | 23.2 | **20.4** | 9.8 / 9.7 |
| 3 | 26.3 | **22.4** | 12.0 / 10.4 |
| 4 | 28.7 | **24.6** | 14.8 / 11.1 |

- The old down kernel was bound by shared-memory reads at T >= 2.
- k_hc_down2 alone: 6.8 µs (T = 1) and 9.9 µs (T = 4).

**KLD**, verify windows of 3, hot set 512: **0.008931** (96.67%; sw74 0.008910).

**Teacher-forced decode**, P2 conditions, 6 windows of 128 tokens (the old kernels ran first):

| Arm | old hc | hc v2 | |
|---|---|---|---|
| 32K plain | 98.2 107.0 104.3 104.2 104.3 99.7, mean 102.9 | 105.3 109.3 107.7 106.4 106.4 99.4, mean **105.7** | +2.7% |
| 32K `--spec 1` | mean 109.3 (verify 11.92 ms) | mean **111.8** (verify 11.62 ms) | +2.3% |
| 245K `--spec 1` | mean 85.1 (verify 16.04 ms) | mean **86.3** (verify 15.82 ms) | +1.4% |

The verify rounds got 0.22-0.30 ms faster, as `test_hc_decode` predicts (about 2.8 µs x 97
mixes).
