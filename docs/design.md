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

The table is the intended layering. The code does not have it yet: most of what it assigns to
`core/` lives in `arch/qwen4exp/` today, and the engine instantiates the qwen4exp classes
directly. The last column says where each piece is now.

| Layer | Generic or specialised | Contents (intended) | Where it is today |
|---|---|---|---|
| `core/` | Generic, stable | Memory tiers (VRAM expert cache, pinned host arena, SSD), cache policies (decayed-LFU, profile prefill, slot borrowing for prefill buffers), CPU worker pool and GPU↔CPU doorbells (mapped memory, in-graph waits), verify-window scheduler, exact speculative sampling, KV cache and page tables, GGUF loading | `core/` has GGUF loading, the expert arena, the CPU pool, the row reader, JSON and platform helpers. The expert cache, its policy, the doorbells and the miss server are in `arch/qwen4exp/moe_fast.cu`; windows, checkpoints and the KV caches in `forward.cu`, `gdn.cu`, `qsa.cu` and `ple.cu`; speculative sampling in `kernels/cuda/sample.cu` |
| `kernels/cuda`, `kernels/cpu` | Shared building blocks | Norms, rope, flash-decode, top-k, sampling, GEMV/GEMM primitives | `kernels/cuda`: ggml wrappers (GEMV, MMQ), `moe_q2`, Q3R, sampling. Norms, rope, attention and the other mixer kernels are in `arch/qwen4exp/` (`hc.cu`, `gdn.cu`, `qsa.cu`, `ple.cu`) |
| `quant/<type>/` | Specialised per format | CPU kernel (e.g. AVX-512 VBMI+VNNI for Q2_0), GPU GEMV and grouped GEMM, repack tool, parity tests | `quant/q2_0/` has the CPU kernel and repack; the GPU expert kernels are in `kernels/cuda/moe_q2.cu` and `moe_fast.cu`. Q2_0 is the only pack |
| `arch/<name>/` | Specialised per architecture | A hand-written forward program: which fused kernels run in which order, captured as one CUDA graph per window. Hot kernels are templated on the architecture's shapes, weight map, and parity tests against llama.cpp. | `arch/qwen4exp/`, as intended, plus the pieces above |
| box profile | Configurable | VRAM reserve, PCIe miss share, worker count, prefill chunk size | Command-line flags and `FLASHRT_*` variables; there is no profile file and no autotune tool |
| `server/` (Rust) | Separate process | OpenAI and Anthropic APIs, chat templates, tool-call parsing, auth; talks to the engine over the protocol below | `server/`, as intended |

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
 "prompt_ms":61234.5,"decode_ms":4890.1,"finish":"length","drafts":{"proposed":330,"accepted":218},
 "cache":{"hits":141200,"misses":6340,"miss_ms":2810.4,"slots":8809}}
{"ev":"error","id":"r1","msg":"..."}
```

The engine keeps the last conversation's state and reuses the longest matching prompt
prefix. `reused` reports how many tokens were reused: the whole previous sequence when the prompt
extends it, else the latest recurrent-state checkpoint inside the shared prefix (one at the end of
the previous prompt, and up to `--ckpts` taken during earlier prefills at chunk ends and before
the prompt's last `--ckpt-tail` tokens). `cache` counts the decode's routed experts found in the
VRAM expert cache and those that missed it, the host's time computing the misses (`miss_ms`),
and the cache's slot count after the request (`slots`; both since 2026-10-01, optional to a
reader). The server sums these events at `/metrics`.

Requests are checked before anything runs; a bad one gets an `error` event and changes no state.
The limits: token ids within the vocabulary; `max_new` ≥ 1; `temperature` ≥ 0 (0 is greedy);
`top_k` 1..64; `top_p` in (0, 1]; `min_p` in [0, 1); `seed` a whole number in 0..2^53.
A request that fails while running gets an `error` event, and the engine resets to an empty
sequence, so the next request starts cold. After a failure the process cannot recover from (a
doorbell timeout, a sticky CUDA error), the `error` message ends with "(fatal: the engine
exits)", and the engine exits with status 3.

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

**P4 KV work so far (2026-09-28), all exact within the KLD gate:**
- **q8_0 KV** (`--kv q8`): fast-path KLD 0.0092. 245K decodes at 70.2 / 63.5 / 71.5 tok/s
  (`bench/results/2026-09-28-sw18-kv-q8`).
- **Host-resident q8 KV with a GPU hot set** (`--kv-hot 4096`): fast-path KLD 0.0087-0.0090.
  - At 245K it frees VRAM for about 7,700-7,900 expert slots, as many as at 32K.
  - Decode over 6 windows: 66-80 tok/s (`2026-09-28-sw20-kv-hot`, `2026-09-28-sw21-warmup`,
    `2026-09-28-sw22-defaults`).
  - The hit rate at depth is still set by cache warm-up and by the generated text.
- **Not done:** Q4 KV with a Hadamard rotation. Given the hot set, it matters less than P4 assumed.

**P2 status (2026-09-28): gates met.** Temperature 1.0, top-k 20, top-p 0.95, from saved states
with the MTP head's KV, 6 windows of 128 tokens each:

| Gate | Target | Plain decode | `--spec 1` (MTP, one draft) | Source |
|---|---|---|---|---|
| 32K decode | ≥ 80 tok/s | 95.9-99.3 (mean 97.9) | 107.8-119.6 (mean 114.2) | `bench/results/2026-09-28-sw31-p2-temp1` |
| 250K decode | ≥ 72 tok/s | 77.2-80.9 (mean 78.5) | 81.5-101.3 (mean 90.2) | 245,760 tokens; same folder |
| Distribution test | exact | | first token equal in 400/400 (2K) and 300/300 (245K), second in 82/82 and 59/59 | `--dist-test`; sw31, sw33 |
| KLD with verify windows and rewinds | ≤ 0.03 | | 0.0087 (windows of 4), 0.0092 (windows of 3, hot set) | sw25, sw33 |

- **Scope done:** the MTP head (Q4_0 experts, trimmed LM head), verify windows on the fast path
  with rewinds, window graphs, exact speculative sampling on a GPU sampler, the multi-token CPU
  miss kernel in use, and the engine process serving it all over the protocol.
- **Since sw85, drafts at temperature > 0 are sampled** from the head's distribution and verified
  by speculative sampling: 32K `--spec 1` 143.1 tok/s, 245K 97.1.
  - The output is exact in distribution (`test_spec_sample`; the distribution test within
    noise), but no longer equals plain sampling token for token.
  - `--argmax-drafts` restores the scheme measured in the table above.
- **At temperature 1.0 one draft per round was best with argmax drafts** (1.56-1.61 tokens per
  round). With sampled drafts a second draft is break-even or slightly ahead (sw85). A second
  draft's acceptance does not pay for the wider window's extra expert misses. Greedy decoding
  at 2K gains 20% with 1-2 drafts (123 tok/s against 102).
- **The window's union of missed experts is the cost of speculation here:** each extra verified
  token adds about 3 ms at 2K, mostly waiting for the CPU (`bench/results/2026-09-28-sw28-profile`).

**P3 status (2026-09-28): gates met.** Prefill in chunks, each layer's experts streamed to the GPU:

| Gate | Target | Measured | Source |
|---|---|---|---|
| Prefill at 32K | ≥ 2,000 tok/s | 2,135 (chunks of 4,096), 2,272 (8,192); 2,008 through the server with the cache rebuild | `bench/results/2026-09-28-sw39-p3`, `sw45-p3` |
| Prefill at 250K | ≥ 1,700 tok/s | 1,959 (q8 KV in VRAM), 1,940 (host KV with a VRAM mirror); 245,760 tokens, chunks of 8,192 | `sw42-p3`, `sw43-p3` |
| KLD of prefill logits | ≤ 0.03 | 0.0085 (chunks of 1,024 and 4,096), 0.0088 (host KV) | `sw37-p3`, `sw44-p3` |

- **Scope done:** big chunks on the expert cache's VRAM (the engine lends it and rebuilds the
  cache after), the grouped int8 Q2_0 GEMM (ggml's MMQ, launched by flashrt), and the prefix cache
  (the engine reuses the previous sequence or its checkpoint).
- **The tensor-core indexer** came with the prefill kernels, later on 2026-09-28. That work also
  added tensor-core attention, a GDN column kernel, BF16 hc activations and flashrt's own expert
  grouping, the chunk length chosen from free VRAM, and flashrt's own int8 expert kernels on the
  planar layout (`moe_q2`). Prefill now runs at 5,580 tok/s at 32K and 5,170-5,309 at 245K. KLD
  0.0084-0.0087. Evidence: `sw46`-`sw62`.
- **The reference path took 38 minutes for 245K;** chunked prefill took about 2 minutes at the
  gate and now takes 46-48 s.

**How the KLD gate is measured** (set 2026-09-27, `bench/results/2026-09-27-p1-kld`):
- **Protocol:** llama-perplexity's KL-divergence protocol on wikitext-2 test, 8,192-token chunks,
  2 chunks, scoring the second half of each chunk.
- **Reference:** an FP16-KV llama.cpp reference (dev tree, `-ub 16`), compared with `tools/fr_kld`.
- **Noise floor:** llama.cpp against itself on the same protocol, 0.0086 (`-ub 512`) and
  0.0078 (Q8_0 KV). A flashrt result near those is at the noise floor.
- **Short-context tests don't gate:** on 65 tokens llama.cpp against itself gives 0.11.

The correctness-first reference forward measured 0.0089 (pass).
