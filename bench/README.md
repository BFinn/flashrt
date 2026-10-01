# bench

| Script | Drives | Notes |
|---|---|---|
| `depthbench.py` | a llama-server it starts itself | One growing conversation at 1K / 32K / 131K / 245K tokens (prefix reuse); 384 generated tokens at temperature 0 with `ignore_eos`; samples VRAM |
| `strata_depthbench.py` | a line-protocol engine (Strata's `GEN`) | Same prompts and token ids as depthbench (tokenized by a running llama-server); `--sampling`, `--set`, `--n-depths` |
| `summarize.py` | the `SUMMARY` lines of either | Mean ± sd per arm and depth; runs ended early by a sampled EOS are excluded |
| `engine_smoke.py` | `flashrt-engine` over its JSON-lines protocol | Requests, prefix reuse, cancellation |
| `server_smoke.py` | a running `flashrt-server` | 16 end-to-end checks of the OpenAI and Anthropic APIs (tools, reasoning, streaming, stops, prefix reuse, disconnects, sampling limits, `/metrics`) |
| `kcmp2.py` | two nsys kernel summaries | Kernel time per token against per verify round |
| `flashrt_depthbench.py` | `flashrt-engine` over its JSON-lines protocol | Window 9's protocol for flashrt: the same token ids and summary as `strata_depthbench.py` (sw87, sw91) |
| `depthsum.py` | the `SUMMARY` lines of `flashrt_depthbench.py` | Mean ± sd per arm and depth of decode tok/s, prompt seconds, prefill tok/s of the new tokens, reused tokens, hit rate and draft acceptance; GPU clock and temperature ranges |
| `agent_trace.py` | a running `flashrt-server` | An agentic coding session: multi-turn, with read-only `read_file` / `list_dir` tools over a source tree (sw97) |
| `gsm8k_eval.py` | any OpenAI-compatible server | GSM8K, greedy, thinking off; `--compare` gives both accuracies and McNemar's paired test (sw116) |
| `run.sh` | the whole tree, in plain bash | One-command reproduction on another machine: `preflight`, `build`, `test`, `decode`, `window9`, `kld`, `server` or `all` (sw129) |
| `reference/` | inputs, not a script | Window 9's token ids, the drafter's vocabulary ranking, the llama.cpp reference patch, the KLD base script and the Strata pin (`reference/README.md`) |
| `scrub.py` | result files | Replaces machine paths and names with the runbook's placeholders, from the untracked `.scrub.local` |
| `mtp_vocab.py` | a GGUF tokenizer (llama.cpp's gguf-py) and corpora | The frequency ranking behind `--draft-vocab` |
| `p0/` | the first measurement windows | Bandwidth probes, routing traces, llama.cpp sweeps (phase 0) |
| `quant/` | calibration for model-side quantization research | See `docs/research/dynamic-quant.md` |

flashrt itself is measured with `tools/fr_bench` (decode and prefill at a depth, from saved
states or a fresh prefill; `--teacher` for paired A/B runs) and `tools/fr_kld` (the KLD gate).
Each run's script, logs and README are in `results/<date>-<topic>/`, and `docs/engine.md` has
the runbook. Same-protocol runs of flashrt against the reference engines (window 9: 384 tokens
per depth, through the engine) start at `results/2026-09-29-sw87-depthbench`; the latest is
`results/2026-10-01-sw128-p5-window9` (n = 5 per arm).

## Baselines on the target box (RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600)

Decode tok/s at 1K / 32K / 134K / 250K, 384 generated tokens, measured 2026-09-26/27 (before
`results/` existed; no folder holds these runs):

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
the model's. Prefill figures for all engines are per new token, since each engine reuses part of
each deeper prompt: Strata and llama.cpp from the start, flashrt since its host checkpoints (sw95,
sw96; R-1 in `docs/improvement-plan.md`), 32,768 tokens at 134K and 134,004 at 250K (sw128).
