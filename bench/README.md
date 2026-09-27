# bench

| Script | Drives | Notes |
|---|---|---|
| `depthbench.py` | a llama-server it starts itself | One growing conversation at 1K / 32K / 131K / 245K tokens (prefix reuse); 384 generated tokens at temperature 0 with `ignore_eos`; samples VRAM |
| `strata_depthbench.py` | a line-protocol engine (Strata's `GEN`) | Same prompts and token ids as depthbench (tokenized by a running llama-server); `--sampling`, `--set`, `--n-depths` |
| `summarize.py` | the `SUMMARY` lines of either | Mean ± sd per arm and depth; runs ended early by a sampled EOS are excluded |

flashrt's engine gets its own driver once it generates (phase 1), using the same prompts.

## Baselines on the target box (RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600)

Decode tok/s at 1K / 32K / 134K / 250K, 384 generated tokens, measured 2026-09-26/27:

| Engine | 1K | 32K | 134K | 250K | Prefill at 32K / 134K / 250K |
|---|---:|---:|---:|---:|---|
| llama.cpp, stock | 33.4 | 28.8 | 20.2 | 14.5 | 1133 / 758 / 478 |
| llama.cpp, patched (expert cache, sparse QSA, pooled indexer keys), no MTP | 40.0 | 38.2 | 35.8 | 31.9 | 1117 / 725 / 444 |
| Strata, tuned, MTP, greedy | 79.1 | 83.3 | 76.8 | 74.9 | 1171 / 1098 / 989 |

These are the numbers flashrt's phase gates (docs/design.md) are set against.
