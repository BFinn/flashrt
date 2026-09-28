# sw78: the layer-end hc combine folded into the decode hc kernels (2026-09-28): no measurable gain

What changed:
- **Staging:** `k_hc_down2` stages x + out * 2 sigmoid(inject / 4) (an fma, as `k_hc_combine`).
- **The residual:** `k_hc_up_mix2` stores the combined x for its columns after the dependency
  wait, so there is no race with the down kernel's reads.
- **The combine weights** pass through the partials buffer.
- 96 fewer launches per token. `FLASHRT_HC_COMB2=0` gives the combine its own kernel again.

**`test_hc_decode`** (`test_hc_decode.txt`): x, mixed and inject match the reference (x to
3e-8). Combine + mix, µs:

| T | own kernel | folded |
|---|---|---|
| 1 | 19.8 | 18.4 |
| 2 | 24.5 | 20.5 |
| 3 | 26.5 | 23.3 |
| 4 | 28.6 | 26.6 |

**KLD**, verify windows: 0.008931, identical: the same fp32 values reach the norm.

**Teacher-forced decode**, 6 windows each (own kernel first):

| Arm | own kernel | folded |
|---|---|---|
| 32K plain | mean 107.2 | mean 106.7 |
| 32K `--spec 1` | mean 112.9 (verify 11.52 ms) | mean 112.6 (verify 11.57 ms) |
| 245K `--spec 1` | mean 88.8 (verify 15.32 ms) | mean 87.4 (verify 15.60 ms) |

- **Within run-to-run spread:** the same configuration measured 106.4 in sw77 and 107.2 here.
- **Why the kernel savings disappear end to end:** in a MoE layer with CPU misses, the GPU
  waits for the host in `k_moe_combine_db`, so faster GPU work often only lengthens that wait.
  sw79 looks at the miss path itself.
- **Kept**, since it is exact and removes launches, but it is not counted as a gain.
