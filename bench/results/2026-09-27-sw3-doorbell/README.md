# sw3: doorbells instead of a host sync per layer (2026-09-27)

**The change.** Before sw3, the host synchronised with the GPU once per MoE layer: it waited for
routing, ran the CPU misses, then launched the rest of the layer. sw3 replaces this with
doorbells:
- per-layer mailboxes in mapped memory;
- a miss-server thread on the host;
- a GPU-side wait in the combine kernel (`k_moe_combine_db`).

The whole token is then enqueued at once. `fr_bench --no-doorbell` keeps the old host sync per
layer in the same build, so the two arms below are an A/B on one binary. The design is described
in `2026-09-27-sw8-affinity/README.md`, which summarises sw3 to sw8.

## Setup

- **Tool:** `fr_bench`, decoding on the fast path.
- **Prompt:** the first 2,048 tokens of `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, prefilled on
  the reference path. Then 128 greedy decode tokens.
- **Expert cache:** 8,059 slots (32.8% of experts), static, filled from the prompt's routing
  counts.
- **Runs:** 3 with doorbells, then 2 with `--no-doorbell` (`sw3.sh`).
- **Build:** the commit is not recorded.

## Results

| Arm | Run | Decode tok/s | Hit rate | Misses | Host ms per token: waiting for routing / running misses |
|---|---:|---:|---:|---:|---|
| Doorbells | 1 | 62.23 | 85.74% | 8,760 | 12.39 / 2.60 |
| Doorbells | 2 | 63.49 | 87.13% | 7,908 | 12.44 / 2.26 |
| Doorbells | 3 | 63.70 | 88.58% | 7,018 | 12.51 / 2.10 |
| Host sync per layer | 1 | 60.62 | 86.55% | 8,262 | 9.59 / 2.62 |
| Host sync per layer | 2 | 60.52 | 89.18% | 6,646 | 9.77 / 2.44 |

- **Doorbells are faster by about 3 tok/s at 2K** (62.2-63.7 against 60.5-60.6). `docs/engine.md`
  cites this run for its doorbell decision ("sw3: 60.5 → 63.5 tok/s at 2K").
- **Tokens:** the first 24 tokens printed in each log are the same in all five runs. The sw8
  README records sw3's tokens as identical.
- **The hit rate varies from run to run** (85.7-89.2%), although the prompt and the cache fill
  are the same. Runs were not bit-reproducible until sw5. Before sw5, the indexer filled its cell
  list through atomic slots, so the order of attention's sum changed between runs.
- **Host miss time per layer** (the logs' "misses per layer" tables): in doorbell run 1, a layer
  with 1 miss took 47.5 µs, 2 misses 74.5 µs and 3 misses 105.5 µs. That is about 30 µs per
  extra miss. In the doorbell runs, 30-37% of layers had no miss.

## Files

- `sw3.sh`: the script.
- `db_2k_r{1,2,3}.txt`: doorbell runs.
- `sync_2k_r{1,2}.txt`: host sync per layer (`--no-doorbell`).
