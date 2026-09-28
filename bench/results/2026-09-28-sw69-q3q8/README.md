# sw69: the three leads after sw68 (2026-09-28)

## 1. 3-bit dense layers: Q3_K multiplied as Q8_0, exactly

`tools/bench_mmq` (`bench_mmq.txt`) measures ggml's MMQ on the dense shapes at T = 16,384:

| Type | 10,240 x 2,560 | 6,144 x 2,560 | 12,288 x 2,560 | 2,560 x 6,144 |
|---|---|---|---|---|
| Q3_K | 120.9 TOPS | 116.6 | 120.6 | 106.4 |
| Q8_0 | 166.1 | 159.9 | 167.3 | 132.3 |

Other types for scale: IQ4_XS 164-171, Q4_K 150-157, Q5_K 143-149.

**Why the conversion is lossless.** A Q3_K weight is d * sc * (q - 4), with sc in -32..31 and
q - 4 in -4..3. The product (q - 4) * sc can reach +128, which int8 cannot hold. The model has
it: 4.1% of the 72.6M scale groups have sc = -32. Its negation lies in -128..124, so Q8_0 holds
every Q3_K weight exactly with scale -d.

`q3r::q3k_to_q8_0` converts on the GPU. Prefill now multiplies Q3_K matrices (Q3R ones
unpacked first) as Q8_0 in scratch; `FLASHRT_Q3_Q8=0` switches it off.
- `test_gemm`: 0 of 1,638,400 values differ; the product differs from Q3_K's by 2.7e-8
  (summation order).
- q8 KV, automatic chunks, one run each:

| Prompt | Q3_K MMQ | as Q8_0 | greedy tokens |
|---|---|---|---|
| 32,768 | 5,600.1 tok/s | **5,758.8** (+2.8%) | identical |
| 65,536 | 5,638.0 tok/s | **5,804.6** (+3.0%) | identical |

- KLD (logits from chunks, fp16 KV): 0.008577 (96.79%).

A kernel of flashrt's own for Q3R would at best match this: the int8 expansion (q - 4) * sc per
16-value group has to happen somewhere, as MMQ does.

## 2. GDN: a warp per state column is slower; the chunked (WY) form is the remaining lever

llama.cpp's fused CUDA op gives each state column a warp (4 rows per lane, 6,144 warps). A
flashrt version of that design (`k_gdn_delta_warp`, `FLASHRT_GDN_KERNEL=warp`, `gdn-warp/`), 32K
prefill under nsys:

| GDN kernel | time (72 calls) | prefill |
|---|---|---|
| column kernel (default) | 0.483 s | 5,589.9 tok/s |
| warp per column | 1.581 s | 4,707.7 tok/s |

The shuffle reductions per token outweigh the occupancy. flashrt's kernel is already about 3x
faster than that design.

**Estimate for the chunked form** (chunks of 64, as llama.cpp's graph version and FLA use):
- **FLOPs:** about 2.6x the recurrence (86K against 33K MAC per token and head), but in
  tensor-core-sized products.
- **Parallel parts:** within a chunk (K K^T, the triangular solve, W and U) chunks are
  independent.
- **Sequential part:** the state carry-over across chunks is sequential per head, but splits by
  value column (each of S's columns evolves on its own). So about 192 CTAs would each run 256
  small steps per 16K chunk.
- **Estimate** (not measured): 0.97 s → roughly 0.3-0.4 s at 64K, about 5% of prefill. It is a
  large kernel project with numerics to validate (the solve in fp32).

## 3. Decode's small mat-vecs (from the sw63 profile, 32K plain, 10.0 ms of kernels per token)

| Group | calls per token | µs per call | µs per token |
|---|---|---|---|
| router, BF16 2,560 → 512 (+ indexer q) | 60 | 4.8 | 286 (near bandwidth) |
| `ssm_alpha` / `ssm_beta`, BF16 2,560 → 48 | 72 | 2.0 | 144 |
| shared-expert gate scalar, BF16 2,560 → 1 | 48 | 1.7 | 83 |
| `quantize_q8_1` (activations for MMVQ) | ~250 | 1.1 | ~280 |
| shared expert gate/up, 640 rows | 99 | 2.9 | 286 |
| `k_hc_combine`, GDN conv, gates, norms | ~300 | 1-3 | ~500 |

Fusing the pairs that share an input would remove about 150-200 launches: alpha+beta, router +
shared-expert gate, indexer q+k, shared gate+up, and repeated quantization of one input. That is
about 0.2-0.3 ms per token, **~2-3% of decode**, at the cost of concatenated weights and
split outputs.
