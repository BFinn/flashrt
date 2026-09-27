# Speed work 3-8: decode at 2K, 42.75 → 83.3 tok/s (2026-09-27)

This folder summarises steps sw3 to sw8. Each step has its own folder (`2026-09-27-sw3-doorbell` ...
`2026-09-27-sw8-affinity`) with its script and raw outputs.

**Arm:** `fr_bench` with a 2,048-token wikitext prompt (`p0c-20260927/wiki.prompt_ids.txt`) and 128
greedy tokens on the fast path. The expert cache is filled from the prompt's routing counts
(8,059 slots, 32.8% of experts), with 8 CPU workers, the doorbell miss server on CPU 0 and the
enqueue thread on CPU 8. Every row is 3 runs.

| Step | Change | tok/s (3 runs) | Correctness |
|---|---|---|---|
| sw2 | Split-K flash-decode attention | 55.4 / 61.2 / 55.6 | QSA parity unchanged |
| sw3 | Doorbells: per-layer mailboxes in mapped memory, a miss-server thread, and a GPU-side wait in the combine kernel; the whole token is enqueued at once | 62.2 / 63.5 / 63.7 (host sync per layer on the same build: 60.6 / 60.5) | tokens identical |
| sw4 | Register-resident GDN delta rule (1,929 → 208 µs per token), rank-count top-k in `k_route`, float4 grouped RMS norm | 74.3 / 74.6 / 71.5 | KLD 0.008901 (reference path, `kl8k-flashrt.log`) |
| sw5 | Deterministic indexer selection order (no atomic slots); fused decode hc up-projection (scale + SiLU, BF16 mat-vec, gated mean) | 77.8 / 78.1 / 78.0 | runs now bit-identical; hc parity 0 of 1,664 over |
| sw6 | `k_route`: prefetched slot lookups, candidate-filtered top-k | 77.9 / 79.1 / 79.2 | **fast-path KLD 0.008550** (`fr_kld --fast`) |
| sw7 | Fused hc down (norm + down + inject, per-stream partials); PLE rows read on a helper thread while layer 0 runs; GPU argmax | 67.4 / 65.2 / 64.7 (regression) | fast KLD run crashed, see below |
| sw8 | RowReader and PLE fetch threads unpinned | **82.9 / 83.3 / 83.3** | **fast-path KLD 0.008672**, no crash |

## Correctness

The reference is an FP16-KV llama.cpp base on 2 × 8K wikitext chunks, the P1 gate protocol.
`fr_kld --fast` scores the decode path: each chunk's first half is prefilled in batches, then the
scored half runs one token at a time through the fast path (expert cache, GPU routing,
doorbells).

| Path | KLD mean | Median | p99 | Same top-1 | PPL ratio |
|---|---:|---:|---:|---:|---:|
| Reference path, P1 (`2026-09-27-p1-kld`) | 0.008947 | 0.00118 | 0.113 | 96.78% | 1.0011 |
| Fast decode path, sw6 | 0.008550 | 0.00115 | 0.098 | 96.78% | 1.0021 |
| Fast decode path, sw8 | 0.008672 | 0.00115 | 0.118 | 96.47% | 0.9991 |
| llama.cpp against itself (`-ub 512`) | 0.008589 | 0.00119 | 0.103 | 96.70% | 1.0023 |

- **The gate holds** (KLD ≤ 0.03). The fast path sits inside llama.cpp's own path-to-path noise.
- **Greedy tokens differ from the reference path after a few tokens.** A GPU cache hit (Q8_1
  activations in ggml MMVQ) is not bit-identical to a CPU miss (flashrt's Q8 AVX-512 path), so
  near-ties flip. This is why the KLD above is the check, not token identity.
- **Runs are deterministic since sw5.** The indexer used to fill its cell list through atomic
  slots, so the order in which attention summed the cells varied from run to run.

## What moved the time (nsys, 64 tokens at 2K)

| | sw2 | sw8 |
|---|---:|---:|
| GPU kernel time per token | 13.1 ms | 11.5 ms |
| Kernel launches per token | 2,116 | 1,730 |
| Wall minus GPU time per token | 3.3-4.9 ms (host sync per layer) | 1.56 ms of gaps (nsys) |
| `k_gdn_delta` | 1,929 µs | 208 µs |
| BF16 MMVF | 3,207 µs | 599 µs, plus `k_hc_down` 1,336 and `k_hc_up_mix` 910 |
| GPU waiting for CPU misses (`k_moe_combine_db`) | host sync | 1.1-1.4 ms |

## Lessons

- **Pool wake-up cost.** One miss took 94 µs while the workers slept between layers (a 300 µs
  spin window), against 32 µs standalone. With a 2 ms spin window it takes 43 µs, and each extra
  miss adds about 30 µs, which is DRAM speed.
- **Thread affinity is inherited.** RowReader threads created while the main thread was pinned
  to the pool's first CPU ended up sharing that CPU with the spinning miss server. The PLE read
  took 4 ms instead of 0.4 ms (sw7). Helper threads now unpin themselves.
- **The sw7 crash** ("unspecified launch failure" in the fast KLD run, before the first
  progress line) did not reproduce after the affinity fix: not with `CUDA_LAUNCH_BLOCKING=1`
  (`kld_dbg.log` in sw7) and not in sw8. The cause is unknown. Watch for it.
- **A static expert cache does not carry over to new text.** Filled from chunk 0's prompt, the
  cache hit 65.6% on the KLD run's decode, against 86-89% in `fr_bench`, where the decode
  continues the prompt's own text. An adaptive cache is needed.

## Next

- **Q3_K MMVQ:** about 475 MB per token at about 320 GB/s takes 1.47 ms. Other types reach
  745-780 GB/s.
- **Faster `k_hc_down`:** 13.8 µs per call.
- **An adaptive expert cache.**
- **The P1 gate depths:** 32K and 250K.
