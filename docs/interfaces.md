# Interfaces: how generic flashrt is, and where

**Status: a design sketch, not implemented.** None of the generic types below (`ArchModule`,
`ModelSpec`, `QuantPack`, `MoeBlock`, `StateSpec`, the `Forward` interface) exists in the tree.
`qwen4exp::Forward` (`arch/qwen4exp/forward.hpp`) is the one model's concrete forward pass, not
this interface. The engine (`engine/session.cpp`) uses the qwen4exp classes directly, and the
offload machinery this page calls generic lives in `arch/qwen4exp/`. `CLAUDE.md`'s rule stands: the API is designed when a
second architecture arrives. The plan for getting there is in `docs/improvement-plan.md`
(G-2, G-3).

Shapes the current code assumes (a fast path needs them, or the code throws):

| Assumption | Where |
|---|---|
| Experts are Q2_0 (GGUF type 42), planar-repacked; the arena, cache slots, CPU kernel and GPU hit kernels assume it | `arch/qwen4exp/spec.cpp`, `quant/q2_0/`, `moe_fast.cu`, `kernels/cuda/moe_q2.cu` |
| `d_model % 512 == 0`, the expert FFN within 64 quant blocks, top-k ≤ 16, experts ≤ 1024 | `moe_fast.cu`, `moe_ref.cu` (routing) |
| Hyper-connections: the fused decode kernels need `hc == 4` (and rank 320 for the v2 up-mix) | `hc.cu` |
| Attention head_dim 256 and a GQA group within `kAttnMaxGroup`; the q8 hot set needs head_dim 256 | `qsa.cu` |
| GDN conv ≤ 8, key dim ≤ 1024 | `gdn.cu` |
| Verify windows of at most 8 tokens | `moe_fast.cu`, `blocks.hpp` |

The rest of this page is the intended design.

## The decision

flashrt is **model-agnostic at the runtime level and model-specific inside compiled-in
modules**:

| Level | Generic? | Why |
|---|---|---|
| Engine protocol (tokens in, tokens out) | Fully | Nothing model-specific crosses it; the front end never changes for a new model |
| Runtime services: memory tiers, expert store and cache, CPU pool + doorbells, window scheduler, speculation, sampling, state snapshots | Fully | This offload machinery is the same for every large MoE, and it is most of the engineering |
| **The MoE block** | Generic primitive | Routing → cache hits on GPU → misses on CPU or PCIe → combine is identical across MoE architectures; only the router details and expert FFN shape vary, and those are parameters |
| Mixers (attention variants, linear attention, SSM) | Kernel library, shared where shapes match | GQA, MLA, QSA/DSA, GDN each need their own kernels; a second model using GQA reuses the GQA kernels |
| **Architecture program** | Specific, compiled in | The order of kernels in a layer, fusions, shapes as template constants, weight names. This is where the speed comes from |
| Quant pack | Specific per format, behind one interface | Each format brings its CPU dot kernel, GPU GEMV and GEMM, and repack |

Architectures and quant packs are **compiled in and registered statically**, selected at
load from the GGUF's `general.architecture` and tensor types. There are no runtime `.so`
plugins until there is a reason: static registration keeps everything inlinable and
avoids an ABI to maintain.

What is deliberately **not** generic:
- No graph IR and no op-by-op scheduler. Each architecture writes its forward pass as
  plain code over primitives, captured once per window shape into a CUDA graph.
- Only one GPU vendor (CUDA).

## The seams (C++ interfaces)

Sketches only, written before any code; the names would change when the API is designed.

### 1. Model description, parsed from the GGUF

```cpp
struct MoeSpec {                 // per MoE layer; most models share one
    int n_expert, top_k, n_shared;
    int d_model, d_ff_expert, d_ff_shared;
    RouterKind router;           // softmax-topk | sigmoid-topk-norm | ...
    float routed_scale;
};
enum class MixerKind { GQA, MLA, QSA, GDN, Mamba2, ... };
struct LayerSpec { MixerKind mixer; int mixer_cfg; bool moe; int moe_cfg; };
struct ModelSpec {
    std::string arch;            // "qwen4exp"
    int n_layer, d_model, n_vocab, max_context;
    std::vector<LayerSpec> layers;
    std::vector<MoeSpec> moe;    // indexed by LayerSpec::moe_cfg
    // mixer configs, rope, norms, MTP presence ...
};
```

`ModelSpec` is plain data. The generic services size themselves from it: KV and state
buffers, expert store, cache slots.

### 2. Architecture module

```cpp
struct ArchModule {
    const char* name;                                        // matches general.architecture
    ModelSpec (*parse)(const Gguf&);                         // metadata -> spec
    void (*bind)(const Gguf&, WeightPlan&);                  // tensors -> tiers (VRAM dense, host experts, SSD)
    std::unique_ptr<Forward> (*make)(Runtime&, const ModelSpec&);
};
struct Forward {                                             // one per loaded model
    virtual void record_window(WindowCtx&, int n_tokens) = 0;    // emits the window's kernels; captured once per n_tokens
    virtual void commit(int n_accepted) = 0;                     // speculation: keep tokens 0..n_accepted-1
    virtual DraftSource* drafter() { return nullptr; }           // MTP head, if the model has one
    virtual ~Forward() = default;
};
```

`record_window` calls generic primitives, e.g.
`moe.block(ctx, layer, x, router_w)`, `kv.append(...)` or `state.checkpoint(...)`, plus
the architecture's own fused kernels.

### 3. The MoE primitive (generic)

```cpp
class MoeBlock {
public:
    // Routes x (n_tokens rows), serves hits from the VRAM cache, sends a tuned share of
    // misses over PCIe and the rest to the CPU pool via doorbells, and combines with the
    // shared expert. Everything is enqueued into the current window graph.
    void block(WindowCtx&, int layer, DevTensor x, const RouterWeights&, DevTensor out);
};
```

It owns the expert store (the host arena of repacked blobs keyed by `(layer, expert)`),
the cache policy (decayed-LFU, profile prefill, asynchronous swaps) and the miss split.
It uses the quant pack for the actual math.

### 4. Quant pack

```cpp
struct QuantPack {
    GgmlType type;                                           // e.g. Q2_0
    size_t (*blob_bytes)(const MoeSpec&);                    // one expert, repacked
    void (*repack)(const void* gguf_rows, void* blob, const MoeSpec&);
    // CPU: an expert FFN for up to N tokens routed to it, reading the blob once
    void (*cpu_expert)(const void* blob, const float* x, int n_tok, float* y, const MoeSpec&, CpuScratch&);
    // GPU: grouped expert FFN over cached slots (decode), and grouped GEMM (prefill)
    void (*gpu_hits)(const HitPlan&, DevTensor x, DevTensor y, cudaStream_t);
    void (*gpu_prefill)(const PrefillPlan&, DevTensor x, DevTensor y, cudaStream_t);
};
```

Dispatch goes through a function pointer once per layer, not per element, so the cost is
negligible. Kernels inside are templated on the format.

### 5. State and speculation (generic)

Architectures declare their state as a list of `StateSpec`s: a KV page table per
attention layer, or a recurrent state per GDN/SSM layer. The runtime allocates them,
checkpoints them before a verify window, and restores them on rejection. Attention KV
needs only truncation; recurrent state needs a checkpoint and replay, which the runtime
provides as a service.

The sampler and the accept loop are generic: exact speculative sampling for deterministic
drafts.

## What a second model costs

A new MoE architecture needs:
- an `ArchModule`: parse, bind, forward program;
- any mixer kernel not already in the library.

Everything else is reused: the expert store, cache, CPU pool, PCIe split, sampling,
speculation plumbing, server and protocol. A new quant format needs one `QuantPack`.

Supporting a new model family is estimated at weeks per family, not months (an estimate: no
second family has been built).
