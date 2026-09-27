# flashrt design

A runtime built for large MoE models that do not fit in VRAM, on one consumer GPU plus a
CPU with fast SIMD. Generic where genericity is free, specialised where the speed comes
from.

The first target is Qwen3.8-Flash-Next (qwen4exp), GSQ-RCO Q2_0, on an RTX 5080 16 GB with
a Ryzen 9 7900X and DDR5.

## Where the time goes (why the design looks like this)

Batch-1 decode on an offloaded MoE is a memory-traffic problem:

- **Dense weights.** Mixers, routers, the shared expert, hyper-connections and the head
  (about 3.5 GB) are read from VRAM once per verify window.
- **Routed experts.** About 0.66 GB per token. Hits come from the VRAM cache. Misses come
  out of host DRAM, computed on the CPU or copied over PCIe; both paths are bound by DRAM
  bandwidth.
- **Round structure.** Round time ≈ mixer + max(CPU misses, GPU hits) + draft + overhead,
  and the halves wait for each other at every layer. The **hit rate** is the main lever;
  **speculation** amortises the dense reads and syncs.
- **Prefill.** Every 2048-token chunk touches essentially every expert, so prefill is
  bound by host-to-device transfer. Bigger chunks are the lever.

Measured baselines on the target box are in `bench/README.md`.

## Layers

| Layer | Generic or specialised | Contents |
|---|---|---|
| `core/` | Generic, stable | Memory tiers (VRAM expert cache, pinned host arena, SSD), cache policies (decayed-LFU, profile prefill, slot borrowing for prefill buffers), CPU worker pool and GPU↔CPU doorbells (mapped memory, in-graph waits), verify-window scheduler, exact speculative sampling, KV cache and page tables, GGUF loading |
| `kernels/cuda`, `kernels/cpu` | Shared building blocks | Norms, rope, flash-decode, top-k, sampling, GEMV/GEMM primitives |
| `quant/<type>/` | Specialised per format | CPU kernel (e.g. AVX-512 VBMI+VNNI for Q2_0), GPU GEMV and grouped GEMM, repack tool, parity tests |
| `arch/<name>/` | Specialised per architecture | A hand-written forward program: which fused kernels run in which order, captured as one CUDA graph per window. Hot kernels are templated on the architecture's shapes, weight map, and parity tests against llama.cpp. |
| box profile | Configurable | VRAM reserve, PCIe miss share, worker count, prefill chunk size; written by `flashrt autotune` |
| `server/` (Rust) | Separate process | OpenAI and Anthropic APIs, chat templates, tool-call parsing, auth; talks to the engine over the protocol below |

**Rule:** no general graph IR and no dynamic scheduler in the hot path. An architecture is
explicit code. The add-on API gets designed when the second architecture arrives, not
before.

**Language:**
- C++20 for the engine and CPU kernels (AVX-512 intrinsics).
- CUDA C++ for GPU kernels.
- Rust for the front-end server.
- Python only for offline tools: packing, profiling, analysis.

## Engine protocol (`engine` ↔ `server`)

JSON lines on stdin and stdout, one object per line. The engine serves one sequence at a
time, and the server queues requests.

Engine to server, at start:

```json
{"ev":"ready","version":"0.1.0","arch":"qwen4exp","max_context":262144,"features":["stop","sampling"]}
```

Server to engine:

```json
{"op":"generate","id":"r1","prompt":[1,2,3],"max_new":512,
 "sampling":{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"seed":42},
 "stop_ids":[248044]}
{"op":"stop","id":"r1"}
{"op":"quit"}
```

Engine to server, while generating:

```json
{"ev":"token","id":"r1","tok":1234}
{"ev":"progress","id":"r1","prompt_done":8192,"prompt_total":65536}
{"ev":"done","id":"r1","generated":384,"prompt_tokens":65536,"reused":32768,
 "prompt_ms":61234.5,"decode_ms":4890.1,"finish":"length","drafts":{"proposed":330,"accepted":218}}
{"ev":"error","id":"r1","msg":"..."}
```

The engine keeps the last conversation's state and reuses the longest matching prompt
prefix. `reused` reports how many tokens were reused.

## Phases and gates

Gates are measured on the target box with `bench/`. Speeds are decode tok/s at 32K / 250K
depth, unless noted.

| Phase | Scope | Gate |
|---|---|---|
| P0 | Measure: DRAM and PCIe bandwidth probes, routing-trace simulator, KL harness | Measured bandwidths, a hit-rate vs slots curve, and Strata's time breakdown on this box |
| P1 | Dense path in one graph per window, doorbell CPU miss engine, AVX-512 Q2_0 kernel, decayed-LFU byte-sized cache; greedy, no MTP | ≥57 / ≥47; KLD vs llama.cpp ≤0.03 (measured as below) |
| P2 | Trimmed-vocab MTP drafter, multi-token CPU kernel, exact speculative sampling | ≥80 / ≥72 at temperature 1.0, top_p 0.95, top_k 20, plus a distribution test |
| P3 | Prefill with 8K chunks on borrowed slots, grouped int8 Q2_0 GEMM, tensor-core indexer, prefix cache | Prefill ≥2,000 at 32K and ≥1,700 at 250K |
| P4 | Q4 KV with Hadamard rotation, dynamic CPU/PCIe split, huge pages; then KLD-gated cache-conditional routing and expert deferral | ≥95 / ≥85 exact; ≥110 with quality options at KLD ≤0.02 |

**P1 status (2026-09-28): gates met, scope complete.**

| Gate | Target | Measured | Source |
|---|---|---|---|
| 32K decode | ≥ 57 tok/s | 94.9 / 96.2 / 99.2 | 3 windows, 1 prefill; `bench/results/2026-09-28-sw17-graphs` |
| 250K decode | ≥ 47 tok/s | 61.3 / 62.0 / 64.2 | 245,760 tokens, 3 windows, saved state; `bench/results/2026-09-28-sw17-graphs` (fresh prefill before graphs: 55.8 / 52.0 / 62.3, `2026-09-28-sw15-pcie-245k`) |
| KLD, fast decode path | ≤ 0.03 | 0.0087 | `fr_kld --fast`; `bench/results/2026-09-28-sw17-graphs` |

- **Scope items done:** one graph per window (two graphs per token, around the PLE read), the
  doorbell CPU miss engine, the AVX-512 Q2_0 kernel, and the decayed-LFU cache with a swap
  budget.
- **At 250K the expert cache is the limit:** the fp16 KV takes 6 GB, leaving 14% of the experts
  resident and a 55-58% hit rate. KV compression and offload (P4) is the next lever.

**How the KLD gate is measured** (set 2026-09-27, `bench/results/2026-09-27-p1-kld`):
- **Protocol:** llama-perplexity's KL-divergence protocol on wikitext-2 test, 8,192-token chunks,
  2 chunks, scoring the second half of each chunk.
- **Reference:** an FP16-KV llama.cpp reference (dev tree, `-ub 16`), compared with `tools/fr_kld`.
- **Noise floor:** llama.cpp against itself on the same protocol, 0.0086 (`-ub 512`) and
  0.0078 (Q8_0 KV). A flashrt result near those is at the noise floor.
- **Short-context tests don't gate:** on 65 tokens llama.cpp against itself gives 0.11.

The correctness-first reference forward measured 0.0089 (pass).
