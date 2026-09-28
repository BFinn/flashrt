# flashrt

An inference runtime for a large Mixture-of-Experts model on one consumer GPU, with most of
the experts in host RAM. It is written for one model on one class of box, and measured
against the best existing engines on that box.

- **Model:** Qwen3.8-Flash-Next (`qwen4exp`: 48 layers of Gated DeltaNet and sparse attention,
  a 512-expert top-10 MoE, hyper-connections, a multi-token-prediction head). The weights are
  the ISTA-DASLab GSQ-RCO Q2_0 GGUF, about 62 GB.
- **Box:** RTX 5080 16 GB (sm_120), Ryzen 9 7900X (AVX-512), 64 GB DDR5-3600, PCIe Gen5.
- **Status (2026-09-28):** decode, speculative decoding with the model's MTP head, chunked
  prefill, an engine process and an OpenAI/Anthropic-compatible server work end to end.
  - The phase gates P1-P3 in [docs/design.md](docs/design.md) are met; P4 is done in part.
  - This is research code: one architecture and one quantization, built and tested on one
    machine.

## Results

Decode and prefill in tok/s on the target box. Prompts are wikitext, the context is the given
depth, and the KV cache is q8.

| Engine | Decode at 32K | Decode at 245-250K | Prefill at 32K | Prefill at 245-250K |
|---|---:|---:|---:|---:|
| llama.cpp, stock (2026-09) | 28.8 | 14.5 | 1,133 | 478 |
| llama.cpp, patched (expert cache, sparse attention) | 38.2 | 31.9 | 1,117 | 444 |
| Strata, tuned, MTP, greedy | 83.3 | 74.9 | 1,171 | 989 |
| **flashrt**, plain decode | ~107 | 78.5 | **6,216** | **5,309** |
| **flashrt**, MTP `--spec 1`, sampled drafts | **143.1** | **97.1** | | |

**Caveats:**
- **Different protocols.** The baselines (`bench/README.md`) are greedy, 384 tokens per depth,
  through each engine's server. flashrt's rows come from `tools/fr_bench`: temperature 1.0,
  top-k 20, top-p 0.95, means of 6 windows of 128 tokens, from saved states. A same-protocol
  run is still to do.
- **Measurement dates.** flashrt's 245K plain decode and prefill figures predate the last
  rounds of work (sw31, sw61).
- **Evidence** for each number is in [docs/sweet-spots.md](docs/sweet-spots.md), which points
  to the `bench/results/` folder.

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
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build
cargo build --release --manifest-path server/Cargo.toml
```

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
