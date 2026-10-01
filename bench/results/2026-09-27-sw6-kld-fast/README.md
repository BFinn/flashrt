# sw6: the first fast-path KLD; `k_route` with prefetched slot lookups (2026-09-27)

**The change** (from `2026-09-27-sw8-affinity/README.md`): `k_route` prefetches its
expert-cache slot lookups and takes its top-k over a filtered set of candidates.

**The new check:** this is the first KLD run on the fast decode path (`fr_kld --fast`). In this
mode, each 8,192-token chunk's first half is prefilled in batches. The scored half then runs one
token at a time through the fast path: expert cache, GPU routing and doorbells. Until sw6, the
KLD gate had scored only the reference forward.

## Setup

`sw6.sh`, one build (the commit is not recorded):
1. **KLD:** `fr_kld ... kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast` against the
   FP16-KV llama.cpp base in `$BENCH/kld`.
2. **Speed:** 3 `fr_bench` runs on the first 2,048 tokens of
   `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, then 128 greedy tokens on the fast path. The expert
   cache is static: 8,059 slots (32.8%).
3. **An nsys decode profile:** 64 tokens at 2K, no CUDA graphs yet.

## Results

### Fast-path KLD (`kl8k-fast.log`)

| KL mean | Median | p99 | p99.9 | Max | Same top-1 | PPL flashrt / base (ratio) | Expert cache hit rate |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.008550 | 0.001146 | 0.098448 | 0.268701 | 0.400162 | 96.777% | 2.5291 / 2.5238 (1.00206) | 65.60% |

- **The fast path passes the P1 gate (≤ 0.03)** and sits inside llama.cpp's own path-to-path
  noise. llama.cpp's batched path against its base gives 0.008589 (`2026-09-27-p1-kld`).
- **The static cache does not carry over to new text.** The KLD run's cache has 7,782 slots
  (31.7%), filled from chunk 0's prompt, and hits only 65.6% on the scored text. `fr_bench`, whose
  decode continues the prompt's own text, hits 86-89%. This motivated the adaptive cache of sw11
  and sw12, which reaches 89.3% on the same test.

### Speed (`db_2k_r*.txt`)

| Run | Decode tok/s | Hit rate | Hits / misses |
|---|---:|---:|---|
| 1 | 77.91 | 86.18% | 52,949 / 8,491 |
| 2 | 79.09 | 86.18% | 52,949 / 8,491 |
| 3 | 79.23 | 86.18% | 52,949 / 8,491 |

- **The new `k_route` leaves the outputs unchanged.** The tokens and the hit and miss counts are
  identical to sw5's.
- **Speed:** sw5 ran at 77.75 / 78.06 / 77.95 tok/s.

### Decode profile (64 tokens at 2K; per token = total / 64)

| Kernel | Share | µs per token | Calls per token | µs per call |
|---|---:|---:|---:|---:|
| `mul_mat_vec_f` BF16 (256 threads) | 15.4% | 1,781.5 | 386 | 4.62 |
| `k_moe_combine_db` (waiting for CPU misses) | 14.4% | 1,671.3 | 48 | 34.82 |
| `mul_mat_vec_q` Q3_K | 12.7% | 1,467.7 | 90 | 16.31 |
| `k_hc_up_mix` | 7.7% | 895.3 | 97 | 9.23 |
| `mul_mat_vec_q_moe` Q2_0, two variants | 5.6% + 5.4% | 650.4 + 622.3 | 48 + 48 | 13.55 / 12.97 |
| `mul_mat_vec_q` Q5_K | 5.2% | 603.7 | 16 | 37.73 |
| `quantize_q8_1` | 4.0% | 459.7 | 398 | 1.15 |
| `k_route` | 3.3% | 383.7 | 48 | 7.99 |

- **`k_route` takes 7.99 µs per call,** down from 12.65 in sw5: 607 → 384 µs per token.
- **GPU kernel time is 742.30 ms over 64 tokens, 11.6 ms per token.** The miss wait in
  `k_moe_combine_db` is longer than in sw5's profile (1,244.6 µs per token). That kernel's time
  is the GPU waiting for the host's misses, not GPU work.

## Files

- `sw6.sh`: the script.
- `kl8k-fast.log`: the fast-path KLD.
- `db_2k_r{1,2,3}.txt`: speed runs.
- `decode_2k_cuda_gpu_kern_sum.csv`, `decode_2k_cuda_api_sum.csv`: the nsys summaries.
