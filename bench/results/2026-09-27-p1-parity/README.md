# P1: forward-pass parity against llama.cpp (2026-09-27)

flashrt's correctness-first GPU forward (`arch/qwen4exp/blocks.cu`) is checked against the
llama.cpp dev tree with `tools/fr_parity`. The reference intermediates come from
`tools/ref_dump`: the dev tree with an eval callback, FP16 KV cache, and experts on the CPU.

**Dumps** (token ids in the `.meta` files):
- `wiki64`: 64 wikitext tokens as one batch, then 4 greedy decode steps.
- `ar9`, `ar65`: 1 prompt token, then 8 or 64 greedy decode steps. This is token by token,
  so llama.cpp uses its decode kernels throughout.
- `batch65`: the 65 `ar65` tokens processed as one batch, with logits at every position.

**Metric:** relative L2 of flashrt's block output against llama.cpp's, when the block is fed
llama.cpp's exact inputs. Tolerance 2e-3.

## Per block

| Block | Dump | Checks | Over tolerance | Worst |
|---|---|---:|---:|---|
| Hyper-connection mix and combine (every layer, both sides, head) | wiki64 | 1,664 | 0 | 1.3e-3 on the prompt batch; ~1e-7 on decode steps |
| GDN mixer (conv, L2 norm, gated delta rule, gated norm) | ar65 | 4,680 | 0 | 9.8e-4 |
| QSA mixer (dense attention while the context fits the selection width) | ar65 | 780 | 2 | 2.8e-3 (layer 27, late steps) |
| MoE (router, top-10, CPU experts, gated shared expert) | ar9 | 864 | 0 | 1.3e-3; routing 4,320 / 4,320 identical |
| PLE (n-gram rows + layer) and head | ar9 | 27 | 0 | 2.3e-7; `ple_embd` and logits bit-identical |

- **Mat-vecs:** every projection (Q/K/V/O on llama.cpp's inputs) is bit-identical to
  llama.cpp's. The vendored MMVQ kernels are the same code.
- **Batch vs decode:** on the batched prompt (`wiki64`), GDN and QSA differ by 1e-2 to 6e-2,
  and 29 of 32,010 routed experts differ. That is llama.cpp's batched path (chunked delta
  rule, MMQ instead of MMVQ, batched flash attention, a different BF16 router matmul), not
  a flashrt defect. On token-by-token dumps the same blocks pass.
- **The two QSA misses:** they grow with context, which fits llama.cpp's FP16 flash-attention
  accumulation.

## End to end (`full_ar65.txt`)

flashrt's own embedding, KV caches, GDN and PLE states, MoE and head, over 65 steps:
- greedy tokens identical at **55 of 65** steps;
- KL(llama.cpp || flashrt) per step: **median 0.0022, mean 0.060, max 0.73**.

The per-layer error on step -1 (a single token, `full_ar9.txt`):
- Layers 0-2 match to 1e-7.
- The first QSA layer adds 3.6e-4. flashrt's gated attention output matches `attn_gated`
  to 6e-8. The output projection re-quantizes its input to 8 bits, and that tiny
  difference crosses one rounding boundary.
- The error grows about 10% per layer through the quantized matmuls, to 7e-3 by layer 35.
- Near-tied routing then flips one expert in five layers, and the error reaches 5e-2 by
  layer 47.

## The noise floor: llama.cpp against itself

The same 65 tokens, llama.cpp token by token (`ar65`) against llama.cpp batched (`batch65`):
- greedy tokens identical at **55 of 65** positions;
- KL per position: **median 0.0034, mean 0.109, max 1.30**.

So on this test, flashrt is closer to llama.cpp's decode path than llama.cpp's batched path
is.
- This model is very sensitive to arithmetic order: 2-bit weights, 8-bit activation
  re-quantization at every matmul, and near-tied top-10 routing.
- Short contexts are high-entropy, so a few positions dominate the mean.

**Consequence for the P1 gate** ("KLD vs llama.cpp ≤ 0.03" in `docs/design.md`): as worded,
llama.cpp itself fails it on this test. The gate needs to be measured:
- on a long-context evaluation like the one behind the 0.008 figures (wikitext, 8K
  context, second half of each chunk scored), which needs the QSA indexer beyond 2,051 cells;
- relative to llama.cpp's own path-to-path noise floor on the same evaluation.
