# KV compression 1: Q8_0 QSA KV cache (2026-09-28)

`--kv q8` stores the QSA K/V cache as llama.cpp's Q8_0: blocks of 32 int8 values with an fp16
scale d = amax / 127 each, quantized from the F32 values as llama.cpp does. That is 272 bytes
per cell per KV head, against 512 for fp16.
- **Writes:** K is normed and roped in place, then both K and V go through a small quantize
  kernel.
- **Reads:** the attention partials dequantize as they read. The kernel is templated on the KV
  format.
- **Indexer:** its pooled keys stay F32, so selection works on unchanged data.
- **State files:** they record the format. An fp16 state loads into a q8 cache by converting on
  the GPU. That is how the depth runs below were done (speed only).

## Correctness

Fast-path KLD with q8 KV for the whole run (prefill and decode), against the FP16-KV llama.cpp
base (`kl8k-fast-q8.log`):

| Arm | KLD mean | Median | p99 | Same top-1 | PPL ratio |
|---|---:|---:|---:|---:|---:|
| flashrt, fp16 KV (sw17) | 0.008744 | 0.00114 | 0.111 | 96.73% | 0.9989 |
| **flashrt, q8 KV** | **0.009220** | 0.00121 | 0.119 | 96.64% | 0.9972 |
| llama.cpp, q8_0 KV, `-ub 16` (P1 noise floor) | 0.007764 | 0.00116 | 0.093 | 97.07% | 1.0037 |

The P1 gate (≤ 0.03) and the P4 quality-option budget (≤ 0.02) both hold.

## Speed (3 runs at 2K; 3 windows of 128 from saved states at depth)

| Arm | fp16 KV (sw17) | q8 KV | Expert slots, fp16 → q8 | Hit rate with q8 |
|---|---|---|---|---|
| 2K, 256 tokens | 98.70 / 99.88 / 100.76 | 99.92 / 100.18 / 99.41 | | |
| 32K | 94.88 / 96.18 / 99.22 (fresh prefill) | 95.22 / 97.07 / 94.75 | 7,481 → 7,783 | 89-91% |
| 245,760 | 61.27 / 62.01 / 64.23 | **70.22 / 63.47 / 71.46** | 3,499 → 5,538 | 62-72% |

- **At 245K the freed 2.8 GB becomes 2,000 more expert slots** and +10-15% decode.
- **At short contexts q8 changes nothing,** because the KV there is small.
- **The next lever is host-resident KV with a GPU hot set.** The indexer reads 2,052 cells per
  layer per token, and the P0 traces measured 6-15% misses at a 4K-block hot set.
