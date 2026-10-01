# sw13: decode at 245,760 tokens with fp16 KV and the adaptive cache, before the depth fixes (2026-09-27)

**Why:** the first decode measurement at the P1 gate's long depth. The gate is ≥ 47 tok/s at
250K (`docs/design.md`). sw12 had made two things possible: storing the QSA KV cache as fp16 (half
the VRAM, same numerics) let a 245K context fit, and the adaptive cache had just landed.

## Setup

- **Tool:** `fr_bench`, one run (`sw13.sh`).
- **Prompt:** the first 245,760 tokens of `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, a fresh
  prefill on the reference path. Then 3 decode windows of 128 greedy tokens on the fast path.
- **KV and cache:** fp16 KV, adaptive decayed-LFU cache with swap budget 8.
- **Build:** sw12's (fp16 KV and the adaptive cache), before sw14's depth fixes. The exact commit
  is not recorded.

## Results (`adapt_245k.txt`)

| | Window 1 | Window 2 | Window 3 |
|---|---:|---:|---:|
| Decode tok/s | 41.90 | 44.87 | 44.42 |
| Hit rate | 50.94% | 61.08% | 59.29% |
| Swaps (as printed after each window) | 399 | 1,386 | 2,333 |

- **Prefill:** 245,760 tokens in 2,198.6 s (111.8 tok/s, reference path).
- **Expert cache:** 3,460 slots (14.1% of experts), from 5,586 MiB free. The fp16 KV leaves this
  little VRAM at this depth.
- **Over the 3 windows:** 57.10% hits (105,253 hits, 79,067 misses). The host spent 14.89 ms per
  token waiting for routing and 7.29 ms running misses.
- **Misses per layer:** only 3.6% of layers had no miss. The most common count was 4 (16.3% of
  layers), and 1.5% had 10.

## What it established

- **The P1 gate at 250K was not met by this build:** 41.9-44.9 tok/s, against ≥ 47.
- **sw15 re-measured after sw14's depth fixes** (dp4a MoE hit kernels, coalesced indexer
  scoring, 8-bit radix select) and reached 55.8 / 52.0 / 62.3 tok/s from a fresh prefill.
  `2026-09-28-sw15-pcie-245k/README.md` shows this run as its "before" row.
- **The gate row in `docs/design.md` comes from sw17** (CUDA graphs, saved state):
  61.3 / 62.0 / 64.2.
- **At this depth the expert cache is the limit.** About 14% of experts fit, and the hit rate
  is 51-61%. `docs/design.md` notes the same for P1 and names KV compression and offload (P4) as
  the next lever. q8 host KV with a GPU hot set followed in sw18 to sw21.

## Files

- `sw13.sh`: the script.
- `adapt_245k.txt`: `fr_bench` output: prefill, the three windows, the miss histogram and the
  first tokens.
