# sw21-22: cache warm-up at depth, swap budget, host-resident embedding (2026-09-28)

**Changes:**
- **Prefill routing counts halve every 4,096 tokens,** so the cache prior favours the prompt's
  recent text.
- **The token embedding lives in pinned mapped host memory,** since decode reads one row per
  token. That gives 260 MB of VRAM to the expert cache.
- **`k_idx_select` histogram atomics are warp-aggregated.**
- **The swap budget was tried at 8 and 32.**

## Checks

- **Selection** unchanged (`fr_parity qsa` on `long2216`: 99.9481% identical).
- **Fast-path KLD** with q8 KV + hot set 512, the host embedding and budget 32: 0.008728, median
  0.00122, same top-1 96.95%.

## Results

| Arm | Runs | tok/s |
|---|---|---|
| 2K, 512 tokens, budget 8 (sw21) | 2 | 102.27 / 103.25 |
| 2K, 512 tokens, budget 32 (sw21) | 2 | 103.99 / 102.81 |
| 2K, 256 tokens, budget 32 + host embedding (sw22) | 3 | 100.77 / 101.97 / 101.86 |
| 32K from the fp16 state, q8 + hot 4096, budget 8 (sw20) | 3 windows | 91.5 / 93.5 / 94.3 |
| 32K, same, budget 32 (sw22) | 3 windows | 94.2 / 86.7 / 83.9 (hits 91 → 80%) |
| 245K **fresh prefill** with the decayed prior, q8 + hot 4096, budget 32 (sw21) | 6 windows | 68.7 / 70.5 / 75.9 / 67.3 / 80.2 / 76.3 |
| 245K from that state, budget 32 + host embedding (sw22) | 6 windows | 71.1 / 67.0 / 72.3 / 63.6 / 72.4 / 73.6 |
| 245K from the fp16 state, old prior, budget 8 (sw20) | 6 windows | 66.3 / 69.4 / 70.5 / 71.8 / 73.7 / 76.4 |
| 245K from the fp16 state, old prior, budget 32 (sw20) | 6 windows | 67.5 / 72.8 / 72.9 / 75.6 / 74.8 / 75.8 |

- **The decayed prior raises the first window's hit rate at 245K,** from 67-69% to 75-79%.
- **Budget 32 shortens warm-up at depth, but churns the cache at 32K.** The hit rate fell from
  about 90% to 80% by the third window. The default stays at 8 (`--swap-budget` to change it).
- **At 245K, window-to-window variance from the generated text (68-88% hit rate) is as large as
  the effects being tuned.** Comparisons there need more windows or more runs than 3.
