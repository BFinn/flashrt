# P1 correctness gate: long-context KL divergence (2026-09-27)

The gate measurement: llama-perplexity's KL-divergence protocol on wikitext-2 test (raw), 8,192-token
chunks, 2 chunks, the second half of each chunk scored (8,190 tokens), against an **FP16-KV
llama.cpp reference** (the dev tree, `-ub 16`, experts on the CPU; `kl8k-ref.log`, PPL
2.5239 ± 0.0509). flashrt's logits come from `tools/fr_kld` on the same tokens: the reference
forward (`arch/qwen4exp/forward_ref`) in 64-token batches, FP16-rounded KV, and the QSA indexer
active beyond 2,051 cells.

| Arm | KL mean | KL median | KL p99 | Same top-1 | PPL ratio vs base |
|---|---:|---:|---:|---:|---:|
| **flashrt** (`kl8k-flashrt.log`) | **0.008947** | **0.001180** | 0.1131 | 96.78% | 1.0011 |
| llama.cpp, batched path: `-ub 512`, FP16 KV (`kl8k-ub512.log`) | 0.008589 ± 0.000258 | 0.001193 | 0.1033 | 96.70% | 1.0023 |
| llama.cpp, deployed config: `-ub 16`, Q8_0 KV (`kl8k-q8.log`) | 0.007764 ± 0.000212 | 0.001159 | 0.0934 | 97.07% | 1.0037 |

- **flashrt is inside llama.cpp's own path-to-path noise:** the same median, a mean within 4% of
  llama.cpp's batched path, and the smallest perplexity drift of the three.
- **The P1 gate (KLD vs llama.cpp ≤ 0.03) passes** with a 3.4× margin.
- **Short, high-entropy tests do not work as a gate.** The 65-token test in
  `2026-09-27-p1-parity` gave llama.cpp against itself a mean KLD of 0.109.

## QSA indexer (long contexts)

`qsa_long.txt`: 2,216 token-by-token steps (a wikitext prompt), every QSA layer.
- **Selection:** flashrt's selected cells match llama.cpp's `indexer_top_k` for 99.948% of cells
  over 1,980 sparse (layer, step) pairs. The worst pair is 99.80%: one block of 513, which is the
  near-tie and exact-zero relu tie case.
- **Attention outputs:** 234 of 26,592 checks are above 2e-3, worst 5.0e-3.
  - 142 of those are before selection starts (step < 2,050), in the same layers as before
    (for example layer 27).
  - The error rises smoothly with context and does not jump at the threshold. This is the
    known FP16 flash-attention and requantization drift, not the indexer.
