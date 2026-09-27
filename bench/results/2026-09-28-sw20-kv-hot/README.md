# KV compression 2: host-resident q8 KV with a GPU hot set; indexer select at depth (2026-09-28)

`--kv-hot BLOCKS` (q8 only) keeps each QSA layer's full KV cache in pinned, mapped host memory.
The GPU holds a hot set of BLOCKS 4-cell blocks (4,096 blocks = 16K cells, 17.8 MB per layer)
with a block→slot table.
- **Writes** go to the host store, and through to the slot when the block is resident.
- **v1 (sw19):** attention read missed cells zero-copy from the host, and a one-CTA kernel
  promoted missed blocks with CLOCK after attention.
- **v2 (sw20):**
  - `k_hot_select` (one CTA) marks the selected resident blocks, pins them for the step, and
    assigns CLOCK victims to up to 1,024 missed blocks.
  - `k_hot_copy` (one CTA per block) copies them host→slot before attention.
  - Attention then reads GPU slots, with zero-copy as the fallback for overflow.
- **Numerics:** values are identical in both stores, so where a block sits never changes a
  result. Everything is device-side and runs inside the CUDA graphs.

## Correctness

- **2K with a 256-block hot set** (smaller than the 513 selected blocks: constant misses and
  promotions): tokens identical to plain q8 KV (`hot256_2k.txt`).
- **Fast-path KLD, q8 KV with a 512-block hot set:** 0.009014, median 0.00126, same top-1 96.90%
  (`kl8k-fast-hot512.log`). Plain q8 KV: 0.009220. They differ only because the free VRAM, and so
  the expert cache, differs.

## Speed (from the saved fp16 states, converted to q8 on load; 3 windows of 128 unless noted)

| Arm | 245,760 | 32K | Expert slots at 245K |
|---|---|---|---|
| fp16 KV (sw17) | 61.3 / 62.0 / 64.2 | 94.9 / 96.2 / 99.2 | 3,499 |
| q8 KV (sw18) | 70.2 / 63.5 / 71.5 | 95.2 / 97.1 / 94.8 | 5,538 |
| q8 + hot set 4096, v1 | 63.6 / 64.0 / 63.4 | 85.1 / 90.5 / 92.9 | 7,744 |
| q8 + hot set 4096, v2 | 63.8 / 65.0 / 65.3 | 91.5 / 93.5 / 94.3 | 7,742 |
| v2 + segmented select, 6 windows | 66.3 / 69.4 / 70.5 / 71.8 / 73.7 / 76.4 | | 7,742 |
| same, swap budget 32, 6 windows | 67.5 / 72.8 / 72.9 / 75.6 / 74.8 / 75.8 | | 7,742 |

## What the 245K profile shows (nsys with `--cuda-graph-trace=node`, 64 tokens)

| µs per token | q8 | q8 + hot set v2 | + segmented select |
|---|---:|---:|---:|
| Wait on CPU misses (`k_moe_combine_db`) | 4,737 | 3,577 | 4,097 (other window) |
| `k_idx_select` | 1,217 | 1,214 | **847** |
| `k_idx_scores128` (at bandwidth) | 446 | 445 | 445 |
| Attention partials | < 272 | 428 | 429 |
| Hot copy + select | | 347 + (< 322) | 359 + 203 |
| GPU kernel total | 14,339 | 13,966 | 14,130 |

- **The hot set is a small net win on GPU time.** It cuts the miss wait by about 1.2 ms for
  about 0.8 ms of upkeep.
- **At depth the expert cache is warming up during the whole 3-window measurement.** Its prior is
  the routing counts of the entire 245K prompt, which is mostly old text. Over 6 windows the hit
  rate climbs from 67% to 80% and decode from 66 to 76 tok/s. A larger swap budget and a prior
  weighted toward recent prompt text (next, sw21) attack this.
- **`k_idx_select` (one CTA per layer at 61K blocks) is the largest indexer cost.** The segmented
  output pass took 1.21 → 0.85 ms per token. The 4 histogram passes remain. `fr_parity qsa` on
  `long2216` still shows 99.9481% of cells identical.
