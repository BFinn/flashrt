# sw80: 64K prefill kernel profile (2026-09-28)

**Why.** The script's comment reads: "64K prefill kernel profile after sw69-sw72 (q8 KV,
automatic chunks), for the next prefill work". sw69 multiplied Q3_K as Q8_0, and sw71-sw72 put
the chunked GDN on fp16 tensor cores.

This is a profile, not an A/B. Its prefill speed was measured under nsys.

## Command line (`sw80.sh`)

```
nsys profile -f true -o p64k --trace=cuda \
  $FLASHRT/build/fr_bench $MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 65536 --gen 1 --prefill-chunk auto --kv q8
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o p64k p64k.nsys-rep
```

The script does not pass `--cuda-graph-trace=node`. The prefill kernels still appear one by one
in the summary. The build's commit is not recorded.

## The run (`p64k.txt`)

- **Chunks:** 16,384 tokens, chosen automatically (estimated 8,103 MiB). The chunk buffers took
  7,406 MiB.
- **Prefill:** 65,536 tokens in 10.9 s, **5,989.6 tok/s**, with the experts streamed to the
  GPU. The first chunk took 2.83 s (5,782 tok/s).
- **Decode:** the one decode token is not a measurement.

## Top kernels (`p64k_cuda_gpu_kern_sum.csv`)

Total GPU kernel time is 10,533.29 ms, about 160.7 µs per prompt token (computed: total /
65,536). The prompt ran as 4 chunks of 16,384, so a kernel with 192 instances runs once per
layer per chunk (4 × 48). ggml type ids: 8 = Q8_0 (the Q3_K layers are multiplied as Q8_0),
23 = IQ4_XS, 12 = Q4_K.

| Kernel | Share | Total ms | Instances | µs per call |
|---|---:|---:|---:|---:|
| `moe_q2::k_moe_q2<2, 32, 2>` (routed experts, gate/up) | 11.4% | 1,198.45 | 192 | 6,241.90 |
| `k_attn_tc<12, 1, 4>` (attention) | 9.0% | 947.01 | 6,144 | 154.14 |
| `mul_mat_q` Q8_0 | 8.5% | 900.38 | 368 | 2,446.68 |
| `k_hc_combine_norm4` | 8.1% | 855.89 | 384 | 2,228.87 |
| `moe_q2::k_moe_q2_down` | 7.4% | 782.40 | 192 | 4,075.02 |
| `mul_mat_q` IQ4_XS | 6.0% | 629.65 | 224 | 2,810.92 |
| `k_gated_mean_x<bf16>` | 5.0% | 529.49 | 384 | 1,378.88 |
| CUTLASS BF16 GEMM 64x256 | 4.4% | 466.96 | 720 | 648.55 |
| CUTLASS BF16 GEMM with ReLU 64x64 | 3.5% | 373.57 | 384 | 972.83 |
| `mul_mat_q` Q4_K | 3.4% | 361.96 | 152 | 2,381.35 |
| `k_gdn_chunk_prep<128>` | 2.9% | 300.23 | 4,608 | 65.15 |
| `k_stream_combine<bf16>` | 2.5% | 268.54 | 192 | 1,398.64 |
| `k_gdn_conv_par` | 2.2% | 234.59 | 144 | 1,629.07 |
| `quantize_mmq_q8_1` | 2.2% | 233.65 | 928 | 251.78 |
| `gemm::k_to_bf16` | 2.2% | 232.53 | 1,156 | 201.15 |
| `k_gdn_chunk_state<128, 32, 8, 3>` | 2.1% | 217.58 | 4,608 | 47.22 |
| `k_route_topk` | 2.0% | 206.52 | 192 | 1,075.61 |
| `k_gated_rms_norm` | 1.9% | 204.83 | 144 | 1,422.41 |
| `mul_mat_q` Q2_0 | 1.9% | 201.70 | 232 | 869.40 |
| `k_idx_select` | 1.7% | 176.73 | 5,952 | 29.69 |
| `k_idx_scores_tc` | 1.5% | 157.16 | 5,952 | 26.40 |
| `k_l2_norm` | 0.9% | 93.62 | 144 | 650.13 |

**Grouped shares,** computed by summing the CSV rows by kernel name:
- all `mul_mat_q` kernels, with their stream-k fixups: 22.6%;
- all `moe_q2::` kernels: 19.5% (gate/up and down alone: 18.8%);
- `k_attn_tc`: 9.0%;
- the GDN kernels (`k_gdn_*`, `k_l2_norm`, `k_gated_rms_norm`): 10.0%;
- the indexer (`k_idx_*`): 3.2%.

The docs' split for the hyper-connection kernels (~21%) is not recomputed here.

## What it established

`docs/sweet-spots.md` ("Prefill is bound by arithmetic and bandwidth in roughly equal parts
(sw80)"), `docs/engine.md` and the flashrt-cuda skill split the prefill as follows:

| Part | Share | Limit (from `docs/sweet-spots.md`) |
|---|---|---|
| Dense MMQ (ggml) | ~22% | 120-167 TOPS; beating it needs a new int8 GEMM |
| hc | ~21% | at memory bandwidth |
| Routed experts (moe_q2) | ~19% | gate/up at ~175 TOPS against a ~283 TOPS ceiling |
| GDN (chunked, fp16 tensor cores) | ~9% | prep is DRAM-bound; state is mma-bound and imbalanced |
| Attention | ~9% | L2-bound gathers |

The next prefill rounds started from this profile:
- **sw81:** `k_route_topk` took 1.08 ms per 16K-token chunk. A warp per token made it 26 times
  faster. sw81 also moved the GDN q/k L2 norm (`k_l2_norm`, 0.094 s here) into the conv kernel.
- **sw82:** `k_to_bf16` took 0.23 s at 64K. BF16 products of one input now share one conversion.
- **sw83:** moe_q2 with the activation scale and magic sum packed.

## Files

- `sw80.sh`: the script.
- `p64k.txt`: `fr_bench` and nsys output.
- `p64k_cuda_gpu_kern_sum.csv`: the kernel summary. The `.nsys-rep` is not in git.
