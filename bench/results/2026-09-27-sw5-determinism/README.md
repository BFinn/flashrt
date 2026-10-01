# sw5: deterministic indexer selection, fused decode hc up-projection (2026-09-27)

**The changes** (from `2026-09-27-sw8-affinity/README.md`):
- **The QSA indexer writes its selected cells in a fixed order.** Before, it filled the cell list
  through atomic slots. The order of attention's sum then changed from run to run, and so did the
  output.
- **The decode hyper-connection up-projection is one kernel,** `k_hc_up_mix`. It does the scale
  and SiLU, the BF16 mat-vec and the gated mean.

## Setup

`sw5.sh`, one build (the commit is not recorded):
1. **Parity:** `fr_parity ... hc` on `$BENCH/parity/wiki64.frd`, and `fr_parity ... qsa` on
   `$BENCH/parity/long2216.frd` (2,216 token-by-token steps).
2. **Speed:** 3 `fr_bench` runs on the first 2,048 tokens of
   `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, then 128 greedy tokens on the fast path. The expert
   cache is static: 8,059 slots (32.8%) from the prompt's routing counts.
3. **One `fr_bench --reference` run,** to compare its tokens with the fast path's.
4. **An nsys decode profile:** 64 tokens at 2K, `--capture-range=cudaProfilerApi --trace=cuda,osrt`,
   with no CUDA graphs yet.

## Results

### Speed and determinism (`db_2k_r*.txt`, `ref_2k.txt`)

| Run | Decode tok/s | Hit rate | Hits / misses |
|---|---:|---:|---|
| 1 | 77.75 | 86.18% | 52,949 / 8,491 |
| 2 | 78.06 | 86.18% | 52,949 / 8,491 |
| 3 | 77.95 | 86.18% | 52,949 / 8,491 |
| Reference path | 36.93 | | |

- **The three fast runs are identical:** the same tokens, hit and miss counts, and share of
  layers per miss count. Until sw4 these varied from run to run. `docs/engine.md` cites this
  run for "Deterministic indexer selection" ("sw5: runs bit-reproducible since").
- **sw4 ran at 74.3 / 74.6 / 71.5 tok/s on the same arm.**
- **The fast path's tokens differ from the reference path's from the fifth printed token**
  (243693 against 240991). The sw8 README explains why: a GPU cache hit (Q8_1 activations in
  ggml's MMVQ) is not bit-identical to a CPU miss, so near-ties flip. The check is the KLD, not
  token identity.

### Parity (tolerance 2e-3 relative L2)

| Check | Checks | Over tolerance | Worst |
|---|---:|---:|---|
| hc mix and combine, `wiki64` (`par_hc_wiki64.txt`) | 1,664 | 0 | 1.258e-3 (`hc_mixed-0 (ffn)`, on the 64-token batch) |
| QSA attention, `long2216` (`par_qsa_long.txt`) | 26,592 | 208 | 4.997e-3 (layer 27, step 2078) |
| Indexer selection, `long2216` | 1,980 (layer, step) pairs | | 99.9481% of cells identical, worst pair 99.8048% |

- **On the four decode steps, the fused hc kernel is within 5.9e-8 to 1.5e-7 of llama.cpp.**
- **The QSA results match sw2's:** 209 over, worst 5.0e-3, selection 99.948%. The misses fit
  llama.cpp's FP16 flash-attention accumulation (sw2 and P1 READMEs).

### Decode profile (64 tokens at 2K; per token = total / 64)

| Kernel | Share | µs per token | Calls per token | µs per call |
|---|---:|---:|---:|---:|
| `mul_mat_vec_f` BF16 (256 threads) | 15.6% | 1,780.3 | 386 | 4.61 |
| `mul_mat_vec_q` Q3_K | 12.9% | 1,469.7 | 90 | 16.33 |
| `k_moe_combine_db` (waiting for CPU misses) | 10.9% | 1,244.6 | 48 | 25.93 |
| `k_hc_up_mix` (new) | 7.9% | 895.5 | 97 | 9.23 |
| `mul_mat_vec_q_moe` Q2_0, two variants | 5.7% + 5.5% | 650.9 + 622.0 | 48 + 48 | 13.56 / 12.96 |
| `k_route` | 5.3% | 607.2 | 48 | 12.65 |
| `mul_mat_vec_q` Q5_K | 5.3% | 604.0 | 16 | 37.75 |
| `quantize_q8_1` | 4.0% | 460.6 | 398 | 1.16 |

- **GPU kernel time is 729.77 ms over 64 tokens,** 11.4 ms per token (sw4: 12.0 ms).
- **What `k_hc_up_mix` replaced** (computed from the two CSVs): in sw4, the 160-thread BF16
  `mul_mat_vec_f` (1,432.3 µs per token), `k_gated_mean` (182.0) and `k_scale_silu` (92.9) were
  1,707 µs per token, 97 calls each. Here none of them appears, and `k_hc_up_mix` takes 895.5 µs.
- **Launches:** about 1,923 per token (computed: (75,648 + 47,416) launch calls / 64), against
  about 2,117 in sw4.

## Files

- `sw5.sh`: the script.
- `db_2k_r{1,2,3}.txt`: fast-path runs. `ref_2k.txt`: the reference-path run.
- `par_hc_wiki64.txt`, `par_qsa_long.txt`: parity.
- `decode_2k_cuda_gpu_kern_sum.csv`, `decode_2k_cuda_api_sum.csv`: the nsys summaries.
