# sw11: the adaptive expert cache, first runs (2026-09-27)

**The change:** the expert cache becomes adaptive. The policy is decayed LFU with hysteresis, the
rule `tools/cache_sim.py` calls `dlfu`:
- every access adds 1 to a (layer, expert) count;
- every 4 tokens, all counts are multiplied by 0.7;
- a missed expert is admitted if its count is ≥ 2 and ≥ 1.5 times the weakest resident's;
- at most 8 uploads are in flight (the "swap budget").

The host updates the counts and uploads while the GPU runs the next token. `fr_bench
--static-cache` keeps the old fixed cache. The policy and its mechanics are described in
`2026-09-27-sw12-adaptive/README.md`, which covers sw11 and sw12 together.

## Setup

`sw11.sh`, one build (the commit is not recorded):
1. **2 runs with `--static-cache`.**
2. **2 runs adaptive.** The first also writes a routing trace (`--trace`).
3. **The fast-path KLD with the adaptive cache:** `fr_kld ... --ctx 8192 --chunks 2 --batch 64
   --fast`.

Every `fr_bench` run uses the first 2,048 tokens of `$BENCH/p0c-20260927/wiki.prompt_ids.txt`,
then 256 greedy tokens on the fast path, with 8,059 slots (32.8%).

## Results

### 2K, 256 tokens

| Arm | Run | Decode tok/s | Hit rate | Hits / misses | Swaps |
|---|---|---:|---:|---|---:|
| Static | 1 | 80.08 | 86.17% | 105,887 / 16,993 | |
| Static | 2 | crashed | | | |
| Adaptive (with trace) | 1 | 81.05 | 89.08% | 109,463 / 13,417 | 1,030 |
| Adaptive | 2 | 80.83 | 89.08% | 109,463 / 13,417 | 1,030 |

- **The adaptive cache cuts misses by 21%** (16,993 → 13,417) and gains about 1 tok/s at 2K.
- **The two adaptive runs are identical** in tokens, hits, misses and swaps. Recording the
  trace does not change the run.
- **Static run 2 aborted** with "CUDA error: unspecified launch failure" (`static_2k_r2.txt`).
  This is the second such abort after sw7's KLD run. sw12 matched both to Xid 43 and changed the
  doorbell timeout to report instead of trapping. `docs/engine.md` (Known issues) records that
  none has occurred since sw11, and that the root cause is not proven.
- **sw12 repeated the 2K comparison with 3 runs per arm:** static 78.7 / 80.0 / 80.5, adaptive
  81.2 / 80.9 / 81.1.

### Fast-path KLD with the adaptive cache (`kl8k-fast-adaptive.log`)

| KL mean | Median | p99 | p99.9 | Max | Same top-1 | PPL flashrt / base (ratio) | Hit rate | Swaps |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.008686 | 0.001087 | 0.108772 | 0.281755 | 0.651845 | 96.618% | 2.5267 / 2.5238 (1.00112) | 89.32% | 28,797 |

- **On new text, the cache now follows the text:** 89.3% hits, against 65.6% with the static
  cache in sw6's fast-path KLD. The cache is filled from chunk 0's prompt in both runs, with
  7,782 slots.
- **The P1 gate (≤ 0.03) holds.**

## Where the docs use it

`docs/engine.md` cites "sw11-12" for the adaptive decayed-LFU cache: "a static cache from the
prompt's routing falls to 66% hits on new text; the adaptive one holds about 89%". The engine's
current parameters differ from sw11's: admit 1, margin 1.2, counts seeded at 0.03 times the
prompt's, and 64 uploads per step committed at the next step (`docs/engine.md`).

## Files

- `sw11.sh`: the script.
- `static_2k_r1.txt`, `static_2k_r2.txt` (the abort), `adapt_2k_r1.txt`, `adapt_2k_r2.txt`:
  `fr_bench` output.
- `kl8k-fast-adaptive.log`: the fast-path KLD.
- `trace_2k.i16`: the routing trace of adaptive run 1, int16 [256 tokens][48 layers][10 experts]
  (122,880 values, expert ids 0-511). Which later analysis read it is not recorded.
