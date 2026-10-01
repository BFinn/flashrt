# sw9: decode at 32K depth with the static expert cache (2026-09-27)

**Why:** the first decode measurement at one of the P1 gate depths. The P1 gate is ≥ 57 tok/s at
32K (`docs/design.md`). The 2K runs of sw3 to sw8 do not show what depth does to attention, the
indexer and the expert cache.

## Setup

- **Tool:** `fr_bench`, one run.
- **Prompt:** the first 32,768 tokens of `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, prefilled once
  on the reference path. Then 3 decode windows of 128 greedy tokens (`--windows 3`) on the fast
  path.
- **Expert cache:** static, filled from the prompt's routing counts. The adaptive cache came in
  sw11.
- **Script:** `sw9.sh`.
- **Build:** after sw8. The exact commit is not recorded.

## Results (`db_32k.txt`)

| | Window 1 | Window 2 | Window 3 |
|---|---:|---:|---:|
| Decode tok/s | 78.90 | 77.80 | 70.69 |
| Hit rate | 85.62% | 85.06% | 74.97% |

- **Prefill:** 32,768 tokens in 271.3 s (120.8 tok/s, reference path).
- **Expert cache:** 6,911 slots (28.1% of experts), from 10,136 MiB free. At 2K the cache had
  8,059 slots.
- **Over the 3 windows:** 81.88% hits (150,925 hits, 33,395 misses). The host spent 9.38 ms per
  token waiting for routing and 3.15 ms running misses.
- **More layers have many misses than at 2K:** 28.4% of layers had no miss, and 1.5% had 7.

## What it established

- **The P1 gate at 32K is met** (70.7-78.9 against ≥ 57). The gate row in `docs/design.md` is
  measured later, with CUDA graphs (sw17: 94.9 / 96.2 / 99.2).
- **The static cache degrades as the text moves on.** By the third window the hit rate falls to
  75% and decode to 70.7 tok/s.
- **sw12 uses this run as its static baseline at 32K.** The adaptive cache holds 78.5 / 77.9 /
  79.5 tok/s (86.8 / 87.8 / 89.7% hits) on the same windows (`2026-09-27-sw12-adaptive`).

## Files

- `sw9.sh`: the script.
- `db_32k.txt`: `fr_bench` output: prefill, the three windows, the miss histogram and the first
  tokens.
