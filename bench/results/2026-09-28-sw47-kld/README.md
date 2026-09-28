# sw47: column GDN delta rule, and the KLD gate for the prefill kernels (2026-09-28)

`k_gdn_delta_col` (arch/qwen4exp/blocks.cu) takes GDN calls of 16 or more tokens:
- four lanes own a column of the 128 x 128 state (32 rows each), so a token's two column sums
  are two shuffles each, and there is no block barrier per token;
- the tokens' q, k, v, gate and beta arrive in tiles of 16 through shared memory (`cp.async`,
  double buffered).

Decode keeps `k_gdn_delta_reg`. `FLASHRT_GDN_COL=0` turns it off.

**Speed:** 32,768 tokens, q8 KV, chunks of 8,192, tensor-core attention on, one run each.

| GDN kernel | prefill |
|---|---|
| block kernel (`FLASHRT_GDN_COL=0`) | 2,631.3 tok/s |
| column kernel v1: next token's q/k in registers (`n32k_q8_col1-v1.txt`) | 2,504.1 tok/s (slower: waits on DRAM every token) |
| column kernel, tiles through shared memory | **2,806.8 tok/s** (+6.7%) |

**KLD gate:** `fr_kld --ctx 8192 --chunks 2`, with both kernels on. Every scored logit comes
from a prefill chunk, except in the `--fast` row.

| Run | KLD mean | same top-1 | before (old kernels) |
|---|---|---|---|
| fp16 KV, `--prefill-chunk 1024` | 0.008834 | 96.56% | 0.008492 (sw37) |
| (same, with column kernel v1) | 0.008692 | 96.89% | |
| q8 KV, `--prefill-chunk 1024` | 0.008442 | 96.92% | |
| host KV + hot set 512 (VRAM mirror), `--prefill-chunk 1024` | 0.008442 | 96.92% | 0.008813 (sw44) |
| `--fast --prefill-chunk 2048 --kv-hot 512` (chunked prefill, then decode) | 0.008723 | 96.95% | 0.008598 (sw44) |

All values are at the noise floor: reference path 0.0089, llama.cpp against itself 0.0078-0.0086.
The two fp16 rows differ only in the GDN's FMA order, which shows the run-to-run spread of
small numeric changes (about ±0.0002).
