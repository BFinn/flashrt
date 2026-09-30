# flashrt

An inference runtime for a large Mixture-of-Experts model on one consumer GPU, with most of
the experts in host RAM. It is written for one model on one class of box, and measured
against the best existing engines on that box.

- **Model:** Qwen3.8-Flash-Next (`qwen4exp`: 48 layers of Gated DeltaNet and sparse attention,
  a 512-expert top-10 MoE, hyper-connections, a multi-token-prediction head). The weights are
  the ISTA-DASLab GSQ-RCO Q2_0 GGUF, about 62 GB.
- **Box:** RTX 5080 16 GB (sm_120), Ryzen 9 7900X (AVX-512), 64 GB DDR5-3600, PCIe Gen5.
- **Status (2026-09-29):** decode, speculative decoding with the model's MTP head, chunked
  prefill, an engine process and an OpenAI/Anthropic-compatible server work end to end.
  - The phase gates P1-P3 in [docs/design.md](docs/design.md) are met; P4 is done in part.
  - This is research code: one architecture and one quantization, built and tested on one
    machine.

## Results

**Same prompts as the reference engines** (`bench/results/2026-09-30-sw110-depthbench`,
against `bench/results/2026-09-27-w9-validation`):
- the same token ids (a synthetic prompt followed by an instruction, at 1K / 32K / 134K / 250K
  tokens);
- the depths in one sequence, each prompt reusing the previous one's shared prefix, 384 generated
  tokens per depth, a fresh engine per run;
- decode tok/s, mean ± sd over n runs.

| Engine | n | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|---:|
| llama.cpp (expert cache, sparse attention; no MTP), greedy | 4 | 37.4 ± 0.6 | 37.2 ± 1.1 | 32.8 ± 1.6 | 30.7 ± 1.1 |
| **greedy** | | | | | |
| Strata 0.1.6, MTP (see the caveat below) | 4 | 87.0 ± 0.7 | 96.0 ± 2.2 | 85.0 ± 2.7 | 80.4 ± 3.6 |
| flashrt, MTP (2 drafts per round) | 5 | 131.9 ± 2.8 | 107.1 ± 2.1 | 99.1 ± 1.0 | 87.0 ± 0.8 |
| flashrt, no MTP | 5 | 102.2 ± 1.1 | 94.8 ± 1.1 | 87.5 ± 0.3 | 82.8 ± 0.3 |
| **temperature 1.0** (top-p 0.95, top-k 20) | | | | | |
| Strata 0.1.6, MTP (n=3 at 250K) | 4 | 80.5 ± 4.1 | 79.1 ± 1.3 | 73.9 ± 1.8 | 69.9 ± 7.2 |
| flashrt, MTP, sampled drafts | 5 | 119.5 ± 10.7 | 108.3 ± 1.6 | 99.8 ± 4.3 | 91.4 ± 8.1 |

- **Prefix reuse.** Every prompt ends with the same instruction, so a deeper prompt shares only its
  context with the previous one. flashrt reuses 32,768 tokens at 134K and 134,004 at 250K, from
  host checkpoints taken during the previous prefill. Strata reused 32,768 and 131,072, and
  llama.cpp about 30,700 and 132,000. The step to 32K prefills cold in flashrt: the engine takes
  the checkpoint before a prompt's tail only after it has seen a prompt keep the text and change
  the tail.
- **Strata's build warns that its cache path is not correct.** With `--expert-cache` on, its log
  says the GPU hit path "is NOT CORRECT" and that its tokens diverge from a cache-off run. Its
  timings are real. Its draft acceptance, and so its MTP speed, come from outputs that differ from
  the model's.

What the table shows:
- **Against llama.cpp:** 2.5-2.7x with neither engine drafting, 2.8-3.5x with flashrt's MTP head.
- **Against Strata, greedy:** ahead at every depth, by 52%, 12%, 17% and 8%. Without its draft
  head flashrt is ahead at 1K, 134K and 250K, and level at 32K (94.8 against 96.0).
- **Against Strata, temperature 1.0:** ahead by 31-48% at every depth.
- **What moved it: the expert cache's warm-up.** The cache is filled from the prompt's routing, and
  this answer routes elsewhere. The policy now starts from scaled-down prompt counts, admits
  sooner, and starts up to 64 uploads per step, committed deterministically (sw99-sw109). Hit rates
  rose from 66-77% to 80-89%, and decode 8-34% over sw96. Prefix reuse (sw95, sw96) changed the
  prompt's time, not decode.
- **Tuned on this protocol:** the cache's settings were found here (sw89, sw100, sw104), and
  checked teacher-forced on wikitext too, where they gain 4-5% (sw104, sw109).
- **Time to the first token at 250K:** 24.0 s with the draft head (sw102) and 22.3 s without it,
  for the 116,708 new tokens. Strata takes 123 s for its 119,640 new tokens.
- **Prefill per new token,** from each engine's own log, at 32K / 134K / 250K:
  - flashrt without the draft head: 5,665 / 5,753 / 5,233 tok/s;
  - flashrt with the head: 5,397 / 5,421 / 4,852 (its prompt pass in chunk calls with grouped
    expert GEMMs, sw102);
  - Strata: 1,173 / 1,090 / 969;
  - llama.cpp: 1,108 / 724 / 443.

**An agentic coding session** (`bench/agent_trace.py`; `bench/results/2026-09-29-sw97-agent`,
`2026-09-30-sw109-commit-lag`): 12 turns through the server, with the model reading this
repository through tool calls, up to 77K tokens of context, with the MTP head. Every turn reuses
the whole previous conversation, and time to the first token is 1.0-2.6 s per turn. Decode is
about 121 tok/s at 82% expert-cache hits (sw109; 108 tok/s at 76% with the cache's earlier
settings). Short turns written right after a tool result still hit least (57% on the first file).

**Continuing natural text** (`tools/fr_bench`, wikitext prompts from saved states, 6 windows of
128 tokens; `2026-09-28-sw85-sampled-drafts`, one run per arm):
- at temperature 1.0 with sampled drafts, 143 tok/s at 32K and 97 at 245K;
- plain greedy decode, about 107 at 32K.

The prompt predicts the generation well there: the expert cache hits 93-95% at 32K and 83-89% at
245K. For context only, Strata on wikitext measured 74.7 / 62.2 tok/s at temperature 1.0 and
90.1 / 103.0 greedy at 32K / 245K (`2026-09-27-p0c`, 2 runs). That was a different harness (a
growing conversation through its server, 384 tokens), so it is not a like-for-like comparison.
A same-harness wikitext run is part of phase 2 of the plan.

**Evidence** for every number is in `bench/results/`. [docs/sweet-spots.md](docs/sweet-spots.md)
maps each result to its folder.

**Quality.** Every output-affecting change is gated on KL divergence against an FP16-KV
llama.cpp reference (2 × 8K wikitext chunks):
- flashrt measures 0.0082-0.0092 throughout;
- llama.cpp against itself on the same protocol measures 0.0078-0.0086.

Speculative sampling is exact in distribution (`tests/test_spec_sample.cpp`, and a
distribution test on the model).

## How it works

Batch-1 decode of an offloaded MoE is a memory-traffic problem. flashrt is built around that
([docs/design.md](docs/design.md), [docs/engine.md](docs/engine.md)).

- **One CUDA graph per decode step or verify window.** Kernels written for the model's shapes
  are fused where it paid: hyper-connection mixes, multi-output mat-vecs, epilogues that write
  the next mat-vec's quantized input.
- **A VRAM expert cache** (decayed LFU, adaptive). Its misses are computed on the CPU with an
  AVX-512 Q2_0 kernel, while the GPU computes the hits. A doorbell protocol through mapped host
  memory avoids a host sync per layer.
- **Speculative decoding with the model's own MTP head.** Drafts are sampled from the head's
  distribution and verified by speculative sampling. The head's experts are requantized to
  Q2_0, and its LM head is trimmed to the frequent tokens.
- **Chunked prefill** with each layer's experts streamed to the GPU while the previous layer
  computes. Its kernels:
  - int8 tensor-core expert kernels on the planar Q2_0 layout;
  - tensor-core sparse attention and indexer;
  - Gated DeltaNet in chunked (WY) form on fp16 tensor cores;
  - ggml's MMQ for the dense layers.
- **An engine process** (JSON lines) with prefix reuse and recurrent-state checkpoints, behind
  a Rust server with OpenAI and Anthropic APIs. The server has its own tokenizer, byte-exact
  with llama.cpp's, and its own chat-template renderer.

## Build

Linux, CUDA 12.8+ (for sm_120), GCC 12+, CMake 3.28+, Ninja, Rust 1.82+.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DFLASHRT_NATIVE=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build
cargo build --release --manifest-path server/Cargo.toml
```

`FLASHRT_NATIVE=ON` tunes the CPU code for the build machine (`-march=native`), as every
measurement here was built. Leave it off for binaries that must run on other CPUs.

## Run

```bash
M=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=mtp-Flash-Next-Q8_0-noembd.gguf           # the MTP draft head (embedding and LM head borrowed)

# decode speed at a depth, with speculation
build/fr_bench $M --ids prompt_ids.txt --n-prompt 32768 --prefill-chunk auto --kv q8 \
    --gen 128 --mtp $D --spec 1 --temp 1.0 --top-k 20 --top-p 0.95

# the server (it starts the engine)
server/target/release/flashrt-server --model $M --port 8090 --engine build/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 1
```

- **Prompt ids:** `prompt_ids.txt` holds the prompt as token ids from the model's tokenizer,
  whitespace-separated (one per line works).
- **Draft vocabulary:** `--draft-vocab RANKS` trims the draft head's vocabulary.
  `bench/mtp_vocab.py` builds the ranking.
- **Timings:** responses carry a `timings` object, as llama.cpp's server does (on the last chunk
  of an OpenAI stream): `prompt_n` prefilled and `cache_n` reused prompt tokens, `prompt_ms`,
  `predicted_n`, `predicted_ms`, the rates, `draft_n` / `draft_n_accepted` with the MTP head, and
  `expert_cache` (the decode's routed experts found in VRAM, `hits`, and not, `misses`).
- **More:** [docs/engine.md](docs/engine.md) has the full runbook: benchmarks, the KLD
  harness, the engine protocol and profiling.

## Tests

```bash
(cd build && ctest --output-on-failure)                    # 14 tests; ~30 s on the target box
FLASHRT_TEST_MODEL=$M ctest --test-dir build -R 'gemm|moe_q2'   # the two that need the model
cargo test --release --manifest-path server/Cargo.toml     # server unit tests
```

The CUDA tests check each kernel against a CPU or double-precision reference. `test_gemv`
(against ggml's own kernels) builds with `-DFLASHRT_LLAMA_DIR=<llama.cpp tree>`.

End-to-end checks run against the model:
- `tools/fr_kld`: the KLD gate;
- `fr_bench --dist-test`: speculative sampling's distribution;
- `bench/server_smoke.py`: the server's APIs against a running server.

## Documentation

| Document | Contents |
|---|---|
| [docs/design.md](docs/design.md) | Architecture, engine protocol, phase gates and their status |
| [docs/engine.md](docs/engine.md) | The engine as built: the decode, speculation and prefill flows, why each piece is shaped as it is (with evidence), what was rejected, the runbook |
| [docs/sweet-spots.md](docs/sweet-spots.md) | The best configurations, where each tuning track stopped and why, untested paths ranked, every `FLASHRT_*` toggle |
| [docs/background.md](docs/background.md) | The initial measurements on the target box |
| [docs/interfaces.md](docs/interfaces.md) | How generic flashrt is, and the C++ seams |
| [docs/clean-room.md](docs/clean-room.md) | What may and may not be copied into this repository |
| [bench/README.md](bench/README.md) | Benchmark drivers and the baseline numbers |
| `bench/results/<date>-<topic>/` | Every measurement behind a claim, each with a README |

## Layout

```
core/            GGUF reader, expert arena (host RAM), CPU worker pool, JSON, platform helpers
kernels/cuda/    shared GPU kernels: sampling, int8 expert GEMMs (moe_q2), Q3R mat-vec,
                 wrappers over the vendored ggml mat-vec/MMQ kernels
quant/q2_0/      the Q2_0 format: reference and AVX-512 CPU kernels, the CPU expert path
arch/qwen4exp/   the model: weights, blocks (GDN, sparse attention, hyper-connections), the MoE
                 fast and streaming paths, the MTP head, the forward program
engine/          the engine process (JSON lines on stdin/stdout)
server/          OpenAI/Anthropic front end (Rust)
tools/           fr_bench, fr_kld, fr_parity, probes and microbenchmarks
tests/           kernel and component tests (ctest)
bench/           benchmark drivers, KLD and server smoke scripts, results
third_party/     vendored ggml (MIT)
docs/            design, engine, sweet spots, clean-room policy
```

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). `third_party/ggml` is ggml/llama.cpp
code under the MIT license, vendored unmodified. Contributions follow the
[clean-room policy](docs/clean-room.md).
