# Speed work 14: dp4a MoE hit kernels, indexer at depth, k_hc_down retune (2026-09-27)

## Changes

- **Expert cache in the arena's planar Q2_0 layout.** A slot is an arena blob copied as is: no
  staging buffer, no unrepack kernel, and swaps are a single H2D copy.
- **Cache-hit experts use two flashrt kernels** instead of ggml's grouped MMVQ plus two Q8_1
  quantize launches:
  - `k_moe_gate_up` quantizes x to int8 per 64 values (aligned with the Q2_0 weight blocks) and
    computes gate and up with dp4a. The codes come out of the planar bytes with one shift and
    mask per 4 elements. It then applies SwiGLU and quantizes the hidden block for the down
    kernel.
  - `k_moe_down` runs the down-projection the same way.
  - Only real hits are processed (a device-side hit count), where ggml's launch always covered
    10 entries.
- **Indexer at depth:** a coalesced warp-per-key scoring kernel (dim 128), and a 4-pass 8-bit radix
  select that finds the same threshold as the old 32-pass select.
- **`k_hc_down`:** 16 rows per block (one wave on 84 SMs) and float4 reads of the shared
  activations.

## Checks

- **`test_moe_hits`:** the hit kernels against a double-precision reference on random planar
  experts. Relative L2 is 0.8-1.4% (int8 rounding of x and of the hidden rows), and the pad entry
  is not written.
- **`test_cpu_pool`:** 20,000 runs with idle gaps (workers asleep in the futex), empty runs and
  barriers, 0 violations. The stress case came from Strata issue #29 (via the flashrt-46 session).
- **`fr_parity qsa` on `long2216`:** indexer selection unchanged at 99.9481% of cells identical.
  208 checks over tolerance, the known FP16 drift.
- **Fast-path KLD:** 0.008901, median 0.00117, same top-1 96.59%, PPL ratio 0.9999. The gate holds.

## Speed at 2K (3 runs, 256 tokens, adaptive cache)

**88.75 / 88.38 / 88.53 tok/s**, against 81.2 / 80.9 / 81.1 in sw12. Hit rate 91.28%.

| Kernel time per token (nsys, 64 tokens) | sw10 | sw14 |
|---|---:|---:|
| Hit experts: gate+up, down (plus the Q8_1 quantizes before) | 653 + 620 µs, plus quantizes | 596 + 375 µs |
| `k_hc_down` | 1,337 µs | 1,120 µs |
| `quantize_q8_1` launches per token | 398 | 302 |
| GPU kernel total | 10.66 ms | 10.27 ms |
