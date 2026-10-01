# sw4: register-resident GDN delta rule, rank-count top-k routing, float4 grouped RMS norm (2026-09-27)

**The changes** (from `2026-09-27-sw8-affinity/README.md`, which summarises sw3 to sw8):
- **The GDN delta rule keeps its state in registers** (`k_gdn_delta_reg<128>`). It went from
  1,929 µs per token in sw2's profile to 208 µs here.
- **`k_route` takes its top-10 with a rank count.**
- **The grouped RMS norm reads float4.**

## Setup

`sw4.sh` runs four things on one build (the commit is not recorded):
1. **The KLD gate on the reference path:** `fr_kld` against the FP16-KV llama.cpp base
   (`$BENCH/kld/kl8k-f16.bin`): 2 chunks of 8,192 tokens, batches of 64. This is the P1
   protocol of `2026-09-27-p1-kld`, without `--fast`.
2. **Speed:** 3 `fr_bench` runs on the first 2,048 tokens of
   `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, then 128 greedy tokens on the fast path. The
   expert cache is static: 8,059 slots (32.8%) filled from the prompt's routing counts.
3. **A decode profile:** `nsys profile --capture-range=cudaProfilerApi --trace=cuda,osrt` over
   64 decode tokens at 2K, reduced with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`.
   CUDA graphs did not exist yet (they came in sw17), so every kernel is its own launch.
4. **Two GDN parity outputs,** `k3_gdn_ar65.txt` and `k3_wiki64.txt`. `sw4.sh` does not
   produce them, and their command lines are not recorded. Their content is `fr_parity`'s GDN
   check on the `ar65` and `wiki64` dumps of `2026-09-27-p1-parity`.

## Results

### KLD, reference path (`kl8k-flashrt.log`)

| KL mean | Median | p99 | p99.9 | Max | Same top-1 | PPL flashrt / base (ratio) |
|---:|---:|---:|---:|---:|---:|---|
| 0.008901 | 0.001206 | 0.115371 | 0.278409 | 0.532406 | 96.740% | 2.5253 / 2.5238 (1.00058) |

The P1 gate is KLD ≤ 0.03, so the gate holds. P1's reference-path KLD was 0.008947.

### Speed (`db_2k_r*.txt`)

| Run | Decode tok/s | Hit rate | Misses | Host ms per token: waiting for routing / running misses |
|---:|---:|---:|---:|---|
| 1 | 74.27 | 88.14% | 7,288 | 10.32 / 2.08 |
| 2 | 74.60 | 89.24% | 6,614 | 10.44 / 1.91 |
| 3 | 71.49 | 88.20% | 7,250 | 10.32 / 2.53 |

sw3 on the same arm reached 62.2 / 63.5 / 63.7 tok/s.

### GDN parity (tolerance 2e-3 relative L2)

| Dump | Checks | Over tolerance | Worst |
|---|---:|---:|---|
| `ar65` (token by token) | 4,680 | 0 | 8.866e-4 (`linear_attn_out-20`) |
| `wiki64` (64-token batch, then 4 decode steps) | 2,024 | 99 | 2.981e-2 (`linear_attn_out-26`) |

- **On `ar65`, the register-resident kernel passes every check.** P1's GDN parity on the same dump
  had a worst case of 9.8e-4.
- **The `wiki64` misses:** 34 of the 99 are on the 64-token batched prompt, and 65 on the four
  decode steps after it. The P1 parity README attributes batch differences of 1e-2 to 6e-2 to
  llama.cpp's batched path (chunked delta rule, MMQ). Whether that also explains the misses on
  the decode steps after the batch is not recorded.

### Decode profile (64 tokens at 2K)

The token count is confirmed by `k_moe_combine_db`: 3,072 instances, which is 64 tokens × 48
layers. "Per token" below is total time / 64. The ggml type ids in the kernel names are
11 = Q3_K, 13 = Q5_K, 42 = Q2_0.

| Kernel | Share | µs per token | Calls per token | µs per call |
|---|---:|---:|---:|---:|
| `mul_mat_vec_f` BF16 (256 threads) | 14.8% | 1,778.5 | 386 | 4.61 |
| `mul_mat_vec_q` Q3_K | 12.2% | 1,466.2 | 90 | 16.29 |
| `mul_mat_vec_f` BF16 (160 threads) | 11.9% | 1,432.3 | 97 | 14.77 |
| `k_moe_combine_db` (GPU waiting for the CPU misses) | 9.0% | 1,081.5 | 48 | 22.53 |
| `mul_mat_vec_q_moe` Q2_0, two variants (cache-hit experts) | 5.5% + 5.2% | 656.4 + 624.5 | 48 + 48 | 13.67 / 13.01 |
| `k_route` | 5.0% | 607.5 | 48 | 12.66 |
| `mul_mat_vec_q` Q5_K (includes the LM head) | 5.0% | 603.9 | 16 | 37.74 |
| `quantize_q8_1` | 3.7% | 448.0 | 398 | 1.13 |
| `k_attn_part<12>` | 1.9% | 232.3 | 12 | 19.36 |
| `k_gdn_delta_reg<128>` | 1.7% | 207.8 | 36 | 5.77 |

- **GPU kernel time is 770.07 ms over 64 tokens, or 12.0 ms per token.**
- **Launches:** the API summary has 81,856 `cudaLaunchKernelExC` and 53,624 `cudaLaunchKernel`
  calls, about 2,117 per token (computed: (81,856 + 53,624) / 64).
- **The two BF16 `mul_mat_vec_f` kernels are the largest cost:** 3.2 ms per token. They are
  mostly the hyper-connection mixes (the router is BF16 too). sw5 fused the hc up-projection,
  and sw7 the hc down-projection.

## Files

- `sw4.sh`: the script.
- `kl8k-flashrt.log`: the reference-path KLD.
- `db_2k_r{1,2,3}.txt`: speed runs.
- `decode_2k_cuda_gpu_kern_sum.csv`, `decode_2k_cuda_api_sum.csv`: the nsys summaries. The
  `.nsys-rep` is not in git.
- `k3_gdn_ar65.txt`, `k3_wiki64.txt`: GDN parity.
