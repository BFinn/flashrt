# sw76: decode kernel profile at 32K, plain and `--spec 1`, after sw71-sw75 (2026-09-28)

**Why.** To see where a decode token's GPU time went after sw71-sw75: the chunked GDN on tensor
cores, `linear_multi`, the decode launch fusions and the v2 hc kernels. The runs use P2's
conditions: temperature 1.0, top-k 20, top-p 0.95, from the saved 32K state. This is a profile,
not an A/B. Its speeds were measured under nsys.

## Command line (`sw76.sh`)

For each arm `a` in `plain` and `spec1`:

```
nsys profile -f true -o p_$a --trace=cuda --cuda-graph-trace=node \
  $FLASHRT/build/fr_bench $MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 32768 --gen 128 --windows 1 \
  --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin \
  [spec1: --mtp $MODELS/mtp-Flash-Next-Q8_0-noembd.gguf --spec 1 --draft-vocab $BENCH/mtp-vocab/ranks.txt] \
  --temp 1.0 --top-k 20 --top-p 0.95 --seed 1 --teacher
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o p_$a p_$a.nsys-rep
```

- **Each arm is one window of 128 teacher-forced tokens.** Both arms route the same tokens: the
  printed tokens are identical.
- **`--cuda-graph-trace=node` shows each kernel inside the CUDA graphs.** Without it, a graph
  shows as one launch.
- **The flashrt-cuda skill uses this script as its profiling template.**
- **Build:** the commit is not recorded.

## The runs (`p_plain.txt`, `p_spec1.txt`, under nsys)

| Arm | Decode tok/s | Expert slots | Hit rate | Swaps | Speculation |
|---|---:|---:|---:|---:|---|
| plain | 87.89 | 8,907 (36.2%) | 94.34% | 277 | |
| `--spec 1` | 109.12 | 8,096 (32.9%) | 93.10% | 245 | 83 rounds, 1.542 tokens per round, the draft accepted in 54.2% of rounds; per round: draft 1.04 ms, verify 12.85 ms, commit 0.22 ms |

- **61.1% of layers had no CPU miss** in the plain run.
- **The miss server's log puts a single-miss layer at 97.6 µs.**

## Top kernels, plain (128 tokens; per token = total / 128)

Total GPU kernel time is 1,326.89 ms, or 10,366 µs per token. The instance counts run slightly
above whole multiples of 128 (for example 129 `k_argmax` calls), so some calls per token are
fractional. ggml type ids: 13 = Q5_K, 23 = IQ4_XS, 12 = Q4_K.

| Kernel | Share | µs per token | Calls per token | µs per call (avg / median) |
|---|---:|---:|---:|---|
| `k_moe_combine_db` (waiting for the CPU misses) | 16.1% | 1,667.4 | 48 | 34.74 / 7.71 |
| `k_hc_up_mix2<1>` | 12.9% | 1,340.5 | 97.8 | 13.71 / 14.56 |
| `k_hc_down2<1>` | 7.1% | 732.5 | 96.8 | 7.57 / 7.55 |
| `q3r::k_matvec` | 7.1% | 731.6 | 49.4 | 14.81 |
| `k_moe_gate_up` (cache-hit experts) | 5.9% | 613.4 | 48 | 12.78 |
| `mul_mat_vec_q` Q5_K (includes the LM head) | 5.8% | 605.7 | 16.1 | 37.56 |
| `k_attn_part<12, KvQ8Hot>` | 4.2% | 432.2 | 12.1 | 35.74 |
| `mul_mat_vec_q` IQ4_XS | 3.9% | 405.5 | 39.3 | 10.32 |
| `k_moe_down` | 3.7% | 385.2 | 48 | 8.03 |
| `k_route` | 3.7% | 382.5 | 48 | 7.97 / 7.97 |
| `k_bf16_multi<1>` | 3.5% | 366.4 | 96.4 | 3.80 |
| `mul_mat_vec_q` Q4_K | 3.2% | 327.2 | 38.3 | 8.54 |
| `k_hot_copy` | 2.1% | 216.2 | 12.1 | 17.88 |
| `k_gdn_delta_reg<128>` | 2.0% | 211.5 | 36.3 | 5.83 |
| `k_idx_select` | 1.7% | 177.8 | 12.1 | 14.70 |
| `k_argmax` | 0.8% | 78.9 | 1.0 | 78.30 |

## Top kernels, `--spec 1` (83 verify rounds; per round = total / 83)

Total GPU kernel time is 1,131.01 ms: 13,627 µs per round, or 8,836 µs per generated token over
128 tokens. A template argument of 2 (`k_hc_up_mix2<2>`, ggml's `mul_mat_vec_q<type, 2>`) marks
the kernel's two-token form. The plain run uses the same kernels with 1.

| Kernel | Share | µs per round | Calls per round | µs per call |
|---|---:|---:|---:|---:|
| `k_hc_up_mix2<2>` | 13.3% | 1,806.2 | 97 | 18.62 |
| `k_moe_combine_db` | 13.2% | 1,792.7 | 48 | 37.35 |
| `k_moe_gate_up_g` | 7.2% | 980.8 | 48 | 20.43 |
| `k_hc_down2<2>` | 6.5% | 885.0 | 96 | 9.22 |
| `k_attn_part<12, KvQ8Hot>` | 6.2% | 842.4 | 13.2 | 64.03 |
| `q3r::k_matvec` | 6.1% | 830.9 | 49.6 | 16.75 |
| `k_moe_down_g` | 4.5% | 612.5 | 48 | 12.76 |
| `mul_mat_vec_q` Q5_K, 2 tokens | 4.5% | 607.1 | 16 | 37.94 |
| `mul_mat_vec_q` IQ4_XS, 2 tokens | 4.4% | 598.5 | 56 | 10.69 |
| `mul_mat_vec_q` Q5_K, 1 token | 3.6% | 490.8 | 1.2 | 411.50 |
| `k_route` | 2.8% | 380.3 | 48 | 7.92 |
| `k_gdn_delta_reg<128>` | 2.8% | 379.2 | 52.9 | 7.17 |

## What it led to

- **sw77, the doorbell skip.** sw77's README cites this profile: "at 32K, 61% of layers have no
  CPU miss (sw76: combine median 7.7 µs, `k_route` 8.0 µs)". Tokens without misses now skip the
  doorbell round trip (`docs/engine.md`: `--spec 1` at 32K +1.8%).
- **sw79, what a miss costs inside decode.** The ~97 µs per single-miss layer here, against
  32 µs for one miss in `bench_moe_cpu`, prompted sw79's measurement.
- **The hc mix is the largest GPU-side cost** (`k_hc_up_mix2` + `k_hc_down2`: 2,073 µs per
  token in plain, computed from the table). sw78 tried folding the layer-end combine into it, with
  no measurable gain.

## Files

- `sw76.sh`: the script.
- `p_plain.txt`, `p_spec1.txt`: `fr_bench` and nsys output.
- `p_plain_cuda_gpu_kern_sum.csv`, `p_spec1_cuda_gpu_kern_sum.csv`: the kernel summaries. The
  `.nsys-rep` files are not in git.
