# sw120: decode kernel profile for P-5, plain and `--spec 2` at 32K and 245K (2026-09-30)

**Why.** P-5 of `docs/improvement-plan.md` lists four targets: a multi-CTA `k_idx_select`, a
parallel hot-set CLOCK, and Q3R and LM-head bandwidth, "a few percent each at depth". Before
choosing among them, sw120 profiled decode in four cells: plain and `--spec 2`, at 32K and 245K. All four use q8 host KV with a hot set of 4,096
blocks, the saved states and teacher forcing. This is a profile, not an A/B. Its speeds were
measured under nsys.

## Command line (`sw120.sh`)

For `ctx` in 32 and 245 (`--n-prompt` 32768 or 245760), and each arm in `plain` and `spec2`:

```
nsys profile -f true -o p_${a}_$ctx --trace=cuda --cuda-graph-trace=node \
  $FLASHRT/build/fr_bench $MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt $NP --gen 128 --windows 1 \
  --kv-hot 4096 --load-state $BENCH/state-${ctx}k-q8-mtp.bin \
  [spec2: --mtp $MODELS/mtp-Flash-Next-Q8_0-noembd.gguf --spec 2 --draft-vocab $BENCH/mtp-vocab/ranks.txt] \
  --teacher
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o p_${a}_$ctx p_${a}_$ctx.nsys-rep
```

- **Sampling is greedy** (no `--temp`). The drafts are the head's argmax.
- **Plain and `--spec 2` print the same tokens at each depth,** as teacher forcing intends.
- **Build:** not recorded in this folder. sw121's README names 0a581c5, the commit that added
  this script, as the build before its change. Whether this profile ran on exactly that build is
  not recorded.

## The runs (under nsys)

| Cell | Decode tok/s | Expert slots | Hit rate | Swaps | Speculation (per round: draft / verify / commit) |
|---|---:|---:|---:|---:|---|
| plain, 32K | 101.17 | 8,907 (36.2%) | 95.20% | 1,837 | |
| plain, 245K | 81.69 | 8,663 (35.2%) | 91.83% | 2,873 | |
| `--spec 2`, 32K | 97.09 | 8,094 (32.9%) | 93.94% | 2,074 | 75 rounds, 1.707 tokens per round; 2.07 / 14.99 / 0.48 ms |
| `--spec 2`, 245K | 74.42 (130 tokens) | 7,824 (31.8%) | 90.25% | 2,633 | 78 rounds, 1.667 tokens per round; 2.47 / 19.44 / 0.45 ms |

Cache settings, all cells: decayed LFU, admit 1.00, margin 1.20, swap budget 64, seed scale 0.030.

## Plain decode: top kernels (128 tokens; per token = total / 128)

Total GPU kernel time is 1,213.34 ms at 32K (9,479 µs per token) and 1,493.42 ms at 245K
(11,667 µs per token). Instance counts run slightly above whole multiples of 128 (129 `k_argmax`
calls), so some calls per token are fractional. ggml type ids: 13 = Q5_K, 23 = IQ4_XS,
12 = Q4_K.

| Kernel | Calls per token | 32K: µs per token (share) | 245K: µs per token (share) | µs per call, 32K / 245K |
|---|---:|---:|---:|---|
| `k_hc_up_mix2<1>` | 97.8 | 1,509.9 (15.9%) | 1,509.0 (12.9%) | 15.45 / 15.44 |
| `k_idx_select` | 12.1 | 177.0 (1.9%) | **1,244.9 (10.7%)** | 14.64 / **102.94** |
| `k_moe_combine_db` (waiting for the CPU misses) | 48 | 666.5 (7.0%) | 1,104.2 (9.5%) | 13.89 / 23.00 (median 1.82 / 2.27) |
| `k_hc_down2<1>` | 96.8 | 750.3 (7.9%) | 750.1 (6.4%) | 7.75 / 7.75 |
| `q3r::k_matvec` | 49.4 | 732.7 (7.7%) | 730.7 (6.3%) | 14.84 / 14.80 |
| `k_moe_gate_up` | 48 | 617.2 (6.5%) | 602.8 (5.2%) | 12.86 / 12.56 |
| `mul_mat_vec_q` Q5_K (includes the LM head) | 16.1 | 606.8 (6.4%) | 606.6 (5.2%) | 37.63 / 37.62 |
| `k_idx_scores128` | 12.1 | 82.5 (0.9%) | 452.8 (3.9%) | 6.82 / 37.44 |
| `k_attn_part<12, KvQ8Hot>` | 12.1 | 432.6 (4.6%) | 430.9 (3.7%) | 35.77 / 35.63 |
| `mul_mat_vec_q` IQ4_XS | 39.3 | 408.3 (4.3%) | 408.9 (3.5%) | 10.39 / 10.40 |
| `k_moe_down` | 48 | 393.0 (4.1%) | 378.1 (3.2%) | 8.19 / 7.88 |
| `k_hot_copy` | 12.1 | 214.2 (2.3%) | 376.3 (3.2%) | 17.71 / 31.12 |
| `k_bf16_multi<1>` | 96.4 | 367.6 (3.9%) | 366.7 (3.1%) | 3.81 / 3.81 |
| `k_route` | 48 | 352.5 (3.7%) | 360.4 (3.1%) | 7.34 / 7.51 |
| `mul_mat_vec_q` Q4_K | 38.3 | 328.1 (3.5%) | 325.4 (2.8%) | 8.57 / 8.50 |
| `k_hot_select` | 12.1 | 113.9 (1.2%) | 270.3 (2.3%) | 9.42 / 22.35 |
| `k_gdn_delta_reg<128>` | 36.3 | 210.5 (2.2%) | 210.3 (1.8%) | 5.80 / 5.80 |
| `k_argmax` | 1.0 | 79.4 (0.8%) | 79.6 (0.7%) | 78.82 / 79.01 |

- **What grows with depth:** the indexer (`k_idx_select`, `k_idx_scores128`), the hot set
  (`k_hot_copy`, `k_hot_select`) and the miss wait. Everything else costs the same at both depths.
- **The hc mix (`k_hc_up_mix2` + `k_hc_down2`) is the largest fixed cost:** 2.26 ms per token at
  both depths (computed from the table), about 24% of the GPU kernel time of a 32K token.

## `--spec 2` decode: top kernels (per verify round = total / rounds)

Total GPU kernel time is 1,252.66 ms over 75 rounds at 32K (16,702 µs per round) and
1,673.59 ms over 78 rounds at 245K (21,456 µs per round). Template argument 3 is the three-token
verify window.

| Kernel | Calls per round | 32K: µs per round (share) | 245K: µs per round (share) | µs per call, 32K / 245K |
|---|---:|---:|---:|---|
| `k_moe_combine_db` | 48 | 2,526.4 (15.1%) | 3,782.8 (17.6%) | 52.63 / 78.81 |
| `k_hc_up_mix2<3>` | 97 | 1,965.3 (11.8%) | 1,988.8 (9.3%) | 20.26 / 20.50 |
| `k_attn_part<12, KvQ8Hot>` | 14.2 | 1,170.5 (7.0%) | 2,068.7 (9.6%) | 82.59 / 146.02 |
| `k_idx_select` | 14.2 | 212.9 (1.3%) | 1,627.8 (7.6%) | 15.02 / 114.91 |
| `k_moe_gate_up_g` | 48 | 1,234.4 (7.4%) | 1,198.8 (5.6%) | 25.72 / 24.98 |
| `mul_mat_vec_q` Q5_K, 1 token | 2.2 | 978.5 (5.9%) | 979.6 (4.6%) | 442.08 / 444.26 |
| `q3r::k_matvec` | 49.6 | 958.4 (5.7%) | 961.2 (4.5%) | 19.30 / 19.37 |
| `k_hc_down2<3>` | 96 | 912.6 (5.5%) | 925.9 (4.3%) | 9.51 / 9.64 |
| `k_idx_scores128` | 14.2 | 142.9 (0.9%) | 886.3 (4.1%) | 10.08 / 62.57 |
| `k_moe_down_g` | 48 | 781.2 (4.7%) | 760.4 (3.5%) | 16.27 / 15.84 |

## What it led to (P-5, sw121-sw128; `docs/engine.md`)

- **sw121, the selection on an 8-CTA cluster.** `k_idx_select` was 1,245 µs per token at 245K,
  103 µs per call, 10.7% of GPU time (this profile). Its README cites these numbers. The cluster
  version takes 25 µs per call: 245K plain +9.9%, `--spec 2` +3.7%, 32K +2.9%.
- **sw122, the parallel CLOCK.** `k_hot_select` was 22.4 µs per call at 245K. In parallel it
  takes 4.4 µs: 245K plain +1.9%.
- **sw123 and sw125, the hc mix: both rejected.** sw125's budget uses this profile: a one-token
  mix is `k_hc_down2` 7.8 µs plus `k_hc_up_mix2` 15.3 µs, 97 per token, about 16% of a 32K
  token. Even a perfect mix would gain about 3%. sw123 tried an L2 prefetch, and sw125 loading
  the weights during the miss wait. `docs/engine.md` records the hc mix track as closed.
- **sw126, the argmax on a cluster.** `k_argmax` took 79 µs per token here. The cluster version
  takes 5.2 µs.
- **sw126, fp16 pooled indexer keys.** `k_idx_scores128` read fp32 keys at bandwidth, 37 µs per
  call at 245K. With fp16 keys it takes 21 µs.
- **Still open:** `docs/engine.md` (Known issues) cites sw120 and sw122 for `k_hot_copy`: about
  31 µs per call at 245K, one call per QSA layer, about 0.37 ms per token.

## Files

- `sw120.sh`: the script.
- `p_plain_32.txt`, `p_plain_245.txt`, `p_spec2_32.txt`, `p_spec2_245.txt`: `fr_bench` and nsys
  output.
- `p_*_cuda_gpu_kern_sum.csv`: the kernel summaries. The `.nsys-rep` files are not in git.
