# flashrt

A fast, architecture-specialised inference runtime for large Mixture-of-Experts models on
a single consumer GPU, with the experts in host RAM.

- **Status:** phase 0 (measurement). Nothing generates tokens yet.
- **First target:** Qwen3.8-Flash-Next (GSQ-RCO Q2_0) on an RTX 5080 16 GB, a Ryzen 9 7900X
  and 64 GB DDR5.

The goal is to beat the best existing engines on this class of box by building each
architecture as explicit, fused, graph-captured code on top of a generic core. That core
covers memory tiers, the adaptive expert cache, CPU/GPU concurrency, speculation and
sampling. See [docs/design.md](docs/design.md).

## Layout

```
core/            memory tiers, expert cache, CPU pool + doorbells, scheduler, sampler, KV (C++20)
kernels/cuda/    shared GPU building blocks (CUDA C++)
kernels/cpu/     shared CPU building blocks (AVX-512)
quant/<type>/    kernel packs per quant format (CPU + GPU + repack + parity tests)
arch/<name>/     one forward program per architecture (qwen4exp first)
engine/          the engine process (JSON-lines protocol, see docs/design.md)
server/          OpenAI/Anthropic front end (Rust)
tools/           probes, routing-trace cache simulator, autotune, pack tools
bench/           depth benchmark, KL harness, the baseline numbers
docs/            design, clean-room policy
```

## Build (Linux, CUDA 12.8+ for sm_120)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build
cargo build --release --manifest-path server/Cargo.toml
```

## Phase 0 tools

| Tool | What it measures |
|---|---|
| `build/membw` | Host DRAM read bandwidth: N threads, AVX-512, 4 KiB vs 2 MiB pages |
| `build/h2dbw` | Host→device bandwidth: pinned vs registered memory, expert-sized vs large copies, zero-copy kernel reads, with or without concurrent CPU reads |
| `tools/cache_sim.py` | Expert-cache hit rate vs slots for LRU, decayed-LFU and Belady, over a routing trace |

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions follow the
[clean-room policy](docs/clean-room.md).
