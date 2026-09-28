# sw74: decode launch fusions, all against none (2026-09-28)

Defaults on, against `FLASHRT_LINEAR_MULTI=0 FLASHRT_FUSE_EPI=0`. The "off" arm ran first.

The fusions, at most about 250 launches fewer per decode step:
- **`linear_multi` v2** (a block of 4 warps per row; sw73). One launch each for:
  - GDN alpha + beta, with the gates as the epilogue (3 → 1, 36 layers);
  - the router + shared-expert gate (2 → 1, 48 layers);
  - the indexer q + k (2 → 1, 12 layers).
- **`FLASHRT_FUSE_EPI`:**
  - The q and k L2 norm runs inside the decode conv kernel (36 launches).
  - SwiGLU writes the shared-expert down projection's q8_1 activations
    (`gemv::swiglu_q8_1`; 48).
  - The GDN gated RMS norm writes `ssm_out`'s q8_1 activations
    (`gemv::gated_rms_norm_q8_1`; 33 layers, not the Q3R ones).
  - `linear_shared` quantizes x once for the mat-vecs of one input that take q8_1
    (GDN qkv + gate, QSA q/k/v, shared gate + up; about 40, depending on the layer's types).

**KLD**, verify windows of 3, hot set 512 (`fast-win3-hot512.log`): **0.008910** (96.80%).
sw73 gave 0.008951 and sw68 0.009167.

**Teacher-forced decode**, P2 conditions, from the saved states, 6 windows of 128 tokens:

| Arm | none | all fusions | |
|---|---|---|---|
| 32K plain | 97.5 102.4 101.8 99.9 100.9 96.2, mean 99.8 | 103.6 107.1 105.7 104.2 104.3 99.8, mean **104.1** | +4.3% |
| 32K `--spec 1` | mean 109.2 (verify 11.93 ms) | mean **111.3** (verify 11.65 ms) | +1.9% |
| 245K `--spec 1` | mean 83.6 (verify 16.36 ms) | mean **85.5** (verify 15.96 ms) | +2.3% |

The gain is larger than sw69's estimate (2-3%) on plain decode. Each window is faster in the
same position, so this is not window noise.
