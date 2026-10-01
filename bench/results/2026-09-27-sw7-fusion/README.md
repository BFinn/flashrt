# sw7: fused hc down, PLE read overlapped with layer 0, GPU argmax (2026-09-27). A regression, and a crash

**The changes** (from `2026-09-27-sw8-affinity/README.md`):
- **One fused kernel for the decode hyper-connection down side** (`k_hc_down`): norm, down and
  inject, with per-stream partials.
- **The PLE rows are read on a helper thread while layer 0 runs.**
- **The argmax runs on the GPU** (`k_argmax`).

## Setup

`sw7.sh`, one build (the commit is not recorded):
1. **hc parity:** `fr_parity ... hc` on `$BENCH/parity/wiki64.frd`.
2. **Fast-path KLD:** `fr_kld ... --ctx 8192 --chunks 2 --batch 64 --fast`.
3. **Speed:** 3 `fr_bench` runs on the first 2,048 tokens of
   `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, then 128 greedy tokens on the fast path. The expert
   cache is static: 8,059 slots (32.8%).
4. **An nsys decode profile:** 64 tokens at 2K, no CUDA graphs yet.

## Results

### Speed (`db_2k_r*.txt`): a regression

| Run | Decode tok/s | Hit rate | Host ms per token: waiting for routing / running misses |
|---|---:|---:|---|
| 1 | 67.36 | 88.97% | 12.09 / 2.02 |
| 2 | 65.21 | 88.97% | 12.01 / 2.58 |
| 3 | 64.68 | 88.97% | 12.61 / 2.10 |

- **sw6 ran at 77.9-79.2 tok/s on the same arm.** The three runs agree with each other (same
  tokens, same hit rate). Their tokens and hit rate differ from sw6's.
- **The cause, found in sw8: thread affinity.** RowReader threads were created while the main
  thread was pinned to the pool's first CPU. They inherited the pin and shared that CPU with the
  spinning miss server, so the PLE read took 4 ms instead of 0.4 ms. sw8 unpinned the helper
  threads and reached 82.9-83.3 tok/s.

### Decode profile (64 tokens at 2K; per token = total / 64)

| Kernel | Share | µs per token | Calls per token | µs per call |
|---|---:|---:|---:|---:|
| `mul_mat_vec_q` Q3_K | 12.9% | 1,469.4 | 90 | 16.33 |
| `k_moe_combine_db` (waiting for CPU misses) | 12.3% | 1,403.0 | 48 | 29.23 |
| `k_hc_down` (new) | 11.7% | 1,335.7 | 97 | 13.77 |
| `k_hc_up_mix` | 8.0% | 909.5 | 97 | 9.38 |
| `mul_mat_vec_q_moe` Q2_0, two variants | 5.8% + 5.4% | 657.3 + 618.2 | 48 + 48 | 13.69 / 12.88 |
| `mul_mat_vec_q` Q5_K | 5.3% | 604.9 | 16 | 37.81 |
| `mul_mat_vec_f` BF16 (256 threads) | 5.3% | 598.9 | 193 | 3.10 |
| `k_argmax` (new) | 0.6% | 78.2 | 1 | 78.19 |

- **GPU kernel time is 729.70 ms over 64 tokens, 11.4 ms per token** (sw6: 11.6 ms). The
  regression is on the host side, not in GPU time.
- **What `k_hc_down` replaced** (computed from the sw6 and sw7 CSVs):
  - the 256-thread BF16 `mul_mat_vec_f` fell from 386 to 193 calls per token, and from 1,781.5
    to 598.9 µs per token;
  - `k_grouped_rms_norm_v4` (166 µs per token in sw6) no longer appears.
- **`k_hc_down` runs at 13.8 µs per call.** The sw8 README lists a faster `k_hc_down` as next
  work, and sw14 retuned it.
- **`k_argmax`'s 78 µs per token was fixed much later:** sw126's argmax on an 8-CTA cluster takes
  5.2 µs (`docs/engine.md`).

### hc parity (`par_hc_wiki64.txt`)

1,664 checks, 0 over 2e-3. The worst is 1.258e-3 (`hc_mixed-0 (ffn)`, on the 64-token batch).
The decode steps are at 1e-7.

### Fast-path KLD: crashed

- **`kl8k-fast-crashed.log`** (written as `kl8k-fast.log` by the script, renamed here): "CUDA
  error: unspecified launch failure" at a ggml kernel launch. `timeout` reports a core dump.
- **`kld_dbg.log`:** a rerun over 1 chunk only (4,095 scored tokens), which completed:

  | KL mean | Median | p99 | Same top-1 | PPL ratio | Hit rate |
  |---:|---:|---:|---:|---:|---:|
  | 0.011322 | 0.001610 | 0.135650 | 95.922% | 0.99854 | 82.36% |

  Its command line is not in `sw7.sh`. The sw8 README says the crash did not reproduce with
  `CUDA_LAUNCH_BLOCKING=1`, citing this log. One chunk is not the gate protocol, and sw7's
  build has no full two-chunk fast-path KLD. sw8's build passed: 0.008672.
- **The crash's cause is not proven.** sw12 matched this abort and one in sw11 to Xid 43. They fit
  a false doorbell timeout from unsigned timer arithmetic. The timeout was fixed to report
  instead of trapping, and no abort has occurred since sw11 (`docs/engine.md`, Known issues).

## Files

- `sw7.sh`: the script.
- `db_2k_r{1,2,3}.txt`: speed runs.
- `decode_2k_cuda_gpu_kern_sum.csv`: the nsys kernel summary. No API summary was kept.
- `par_hc_wiki64.txt`: hc parity.
- `kl8k-fast-crashed.log`: the crashed KLD run.
- `kld_dbg.log`: the one-chunk rerun.
