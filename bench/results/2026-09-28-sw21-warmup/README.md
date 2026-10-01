# sw21: cache warm-up at depth: a decayed prefill prior and swap budget 8 against 32 (2026-09-28)

**Why.** sw20 found that at 245K the expert cache warms up during the whole measurement. Its
prior was the routing counts of the entire 245K prompt, which is mostly old text, and over 6
windows the hit rate climbed from 67% to 80% (`2026-09-28-sw20-kv-hot/README.md`). sw21 tries
two fixes:
- **Prefill routing counts halve every 4,096 tokens,** so the prior favours the prompt's recent
  text.
- **A larger swap budget:** 32 uploads in flight instead of 8.

sw21 and sw22 are summarised together in `2026-09-28-sw22-defaults/README.md`.

## Setup

`sw21.sh`, one build (the commit is not recorded):
1. **2K, budget 8 against 32:** 2 runs each, interleaved. The first 2,048 tokens of
   `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, then 512 greedy tokens. fp16 KV, 8,084 slots
   (32.9%), CUDA graphs on.
2. **245K, fresh prefill:** 245,760 tokens with the decayed prior, q8 host-resident KV with a hot
   set of 4,096 blocks per layer (`--kv-hot 4096`), budget 32, 6 windows of 128 greedy tokens.
   The run saved its state to `$BENCH/state-245k-q8.bin` (`--save-state`). That snapshot was
   later reused for speed runs at 245K.

## Results

### 2K, 512 tokens

| Budget | Run | Decode tok/s | Hit rate | Misses | Swaps |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 102.27 | 91.94% | 19,809 | 1,617 |
| 8 | 2 | 103.25 | 91.94% | 19,809 | 1,617 |
| 32 | 1 | 103.99 | 92.72% | 17,884 | 1,658 |
| 32 | 2 | 102.81 | 92.37% | 18,763 | 1,745 |

- **At 2K the budget makes no clear difference:** 102.3-103.3 tok/s against 102.8-104.0.
- **The two budget-8 runs are identical** in hits, misses and swaps. The two budget-32 runs are
  not. Why is not recorded. `docs/engine.md` later found that committing uploads whenever a query
  found them done made runs vary (sw106), and now commits them at a fixed step.

### 245,760 tokens, fresh prefill, decayed prior, q8 KV + hot set 4,096, budget 32 (`hot_b32_245k_fresh.txt`)

| Window | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 68.72 | 70.46 | 75.85 | 67.25 | 80.19 | 76.27 |
| Hit rate | 75.08% | 79.80% | 85.30% | 70.70% | 86.66% | 81.94% |

- **Prefill:** 245,760 tokens in 2,211.3 s (111.1 tok/s, reference path).
- **Expert cache:** 7,691 slots (31.3% of experts), from 11,164 MiB free. sw13 had 3,460 slots
  with fp16 KV on the GPU.
- **Over the 6 windows:** 79.91% hits (294,587 hits, 74,053 misses, all on the CPU). The host
  spent 9.22 ms per token waiting for routing and 3.80 ms running misses.

## What it established

- **The decayed prior stays.** `docs/engine.md`'s decision table, under "Prefill routing counts
  halve every 4,096 tokens", reads: "sw21: first-window hit rate at 245K 67-69% → 75-79%".
- **The window-to-window variance is large.** The hit rate moves by 16 points between windows 4
  and 5 (70.70% → 86.66%). The sw22 README concludes that at 245K this variance is as large as the effects being
  tuned. `docs/engine.md` now asks for at least 6 windows at depth.
- **The swap budget went to 32 by default with this run's commit, and back to 8 in sw22** (the
  next commit): 32 churned the cache at 32K, where hits fell from about 90% to 80% by the third
  window. The engine now starts 64 uploads per step and commits them at the next step
  (`docs/engine.md`).
- **Cited in:**
  - `docs/engine.md`, the 245K results row "q8 + hot set 4096, fresh prefill, decayed prior,
    6 windows";
  - `docs/sweet-spots.md`, the KV row (sw18, sw21);
  - `docs/design.md`, "P4 KV work so far": decode over 6 windows, 66-80 tok/s, with the host
    hot set.

## Files

- `sw21.sh`: the script.
- `b8_2k_r{1,2}.txt`, `b32_2k_r{1,2}.txt`: the 2K runs.
- `hot_b32_245k_fresh.txt`: the 245K run. The 6.5 GB state file it wrote is not in git.
