# bench

| Script | Drives | Notes |
|---|---|---|
| `depthbench.py` | a llama-server it starts itself | One growing conversation at 1K / 32K / 131K / 245K tokens (prefix reuse); 384 generated tokens at temperature 0 with `ignore_eos`; samples VRAM |
| `strata_depthbench.py` | a line-protocol engine (Strata's `GEN`) | Same prompts and token ids as depthbench (tokenized by a running llama-server); `--sampling`, `--set`, `--n-depths` |
| `summarize.py` | the `SUMMARY` lines of either | Mean ± sd per arm and depth; runs ended early by a sampled EOS are excluded |
| `engine_smoke.py` | `flashrt-engine` over its JSON-lines protocol | Requests, prefix reuse, cancellation |
| `server_smoke.py` | a running `flashrt-server` | 11 end-to-end checks of the OpenAI and Anthropic APIs (tools, reasoning, streaming, stops, prefix reuse, disconnects) |
| `kcmp2.py` | two nsys kernel summaries | Kernel time per token against per verify round |
| `flashrt_depthbench.py` | `flashrt-engine` over its JSON-lines protocol | Window 9's protocol for flashrt: the same token ids and summary as `strata_depthbench.py` (sw87, sw91) |
| `scrub.py` | result files | Replaces machine paths and names with the runbook's placeholders, from the untracked `.scrub.local` |
| `mtp_vocab.py` | a GGUF tokenizer (llama.cpp's gguf-py) and corpora | The frequency ranking behind `--draft-vocab` |
| `p0/` | the first measurement windows | Bandwidth probes, routing traces, llama.cpp sweeps (phase 0) |
| `quant/` | calibration for model-side quantization research | See `docs/research/dynamic-quant.md` |

flashrt itself is measured with `tools/fr_bench` (decode and prefill at a depth, from saved
states or a fresh prefill; `--teacher` for paired A/B runs) and `tools/fr_kld` (the KLD gate).
Each run's script, logs and README are in `results/<date>-<topic>/`, and `docs/engine.md` has
the runbook. A same-protocol run of flashrt against the baselines below (greedy, 384 tokens,
through the engine) is in `results/2026-09-29-sw91-depthbench`.

## Baselines on the target box (RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600)

Decode tok/s at 1K / 32K / 134K / 250K, 384 generated tokens, measured 2026-09-26/27:

| Engine | 1K | 32K | 134K | 250K | Prefill at 32K / 134K / 250K |
|---|---:|---:|---:|---:|---|
| llama.cpp, stock | 33.4 | 28.8 | 20.2 | 14.5 | 1133 / 758 / 478 |
| llama.cpp, patched (expert cache, sparse QSA, pooled indexer keys), no MTP | 40.0 | 38.2 | 35.8 | 31.9 | 1117 / 725 / 444 |
| Strata, tuned, MTP, greedy | 79.1 | 83.3 | 76.8 | 74.9 | 1171 / 1098 / 989 |

These are the numbers flashrt's phase gates (docs/design.md) are set against.

The Strata builds measured here (0.1.4 above; 0.1.6 in `results/2026-09-27-p0c` and
`results/2026-09-27-w9-validation`) ran with `--expert-cache`. Strata 0.1.6 logs a warning that its
GPU hit path with the cache on is not correct: its tokens diverge from a cache-off run. Its
timings are real; its draft acceptance, and so its MTP speed, come from outputs that differ from
the model's. Prefill figures for all engines are per new token: Strata and llama.cpp reused part
of each deeper prompt, and flashrt did not (`docs/improvement-plan.md`, R-1).
