// SPDX-License-Identifier: Apache-2.0
// The fast MoE path for decode: a VRAM expert cache (experts in the arena's planar Q2_0 layout,
// one slot each, uploaded as is), routing on the GPU, cache hits as two grouped int8 (dp4a)
// kernels (gate+up with SwiGLU, then down) and misses on the CPU pool, overlapped with the GPU.
#pragma once

#include "arch/qwen4exp/blocks.hpp"

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace flashrt::qwen4exp {

struct ExpertCache {
    int n_slots = 0;
    size_t slot_bytes = 0;            // one expert: gate | up | down, planar Q2_0 (quant/q2_0/q2_0.hpp)
    uint8_t* slots = nullptr;         // device, n_slots * slot_bytes
    int32_t* table_dev = nullptr;     // [n_layer][n_expert] -> slot, or -1
    std::vector<int32_t> table;       // host mirror of table_dev
    std::vector<int32_t> owner;       // slot -> layer * n_expert + expert, or -1
};
ExpertCache alloc_expert_cache(const Spec& s, int n_slots);
void free_expert_cache(ExpertCache& c);

// Uploads the experts of `order` ((layer, expert), best first) into the free slots until the
// cache is full.
void expert_cache_fill(const Spec& s, ExpertCache& cache, const ExpertArena& arena,
                       const std::vector<std::pair<int, int>>& order, cudaStream_t stream);

// Host-side buffers for moe_block_fast (pinned where the GPU copies to or from them).
struct MoeFastHost {
    const ExpertArena* arena = nullptr;
    CpuPool* pool = nullptr;
    int32_t* route_host = nullptr;    // pinned: [n_hits, n_miss, miss experts[k], (float) miss weights[k]]
    float* x_host = nullptr;          // pinned: the block input, for the CPU misses
    float* cpu_out = nullptr;         // pinned: the CPU misses' weighted sum
    std::vector<uint8_t> act_mem, scratch;
    cudaEvent_t routed = nullptr;
    // Doorbell mode (decode): every layer has a mailbox in mapped pinned memory. k_route writes
    // the routing and x there and raises `routed`; a miss-server thread runs the misses as each
    // layer's routing lands and raises `done`; the combine kernel waits for `done` on the GPU.
    // No host sync per layer, so a whole token is enqueued at once.
    bool doorbell = false;
    uint8_t* mbox = nullptr;          // host view, n_layer * mbox_stride
    uint8_t* mbox_dev = nullptr;      // device view
    size_t mbox_stride = 0;
    uint32_t seq = 0;                 // token sequence number, the flag value for this token
    struct MissServer* server = nullptr;
    // PCIe misses (enable_pcie_misses): the GPU reads some misses straight from the mapped arena
    const uint8_t* arena_dev = nullptr;
    size_t arena_stride = 0;
    float pcie_frac = 0.0f;
    int pcie_max = 0;
    // the experts each layer selected for the last token, [n_layer][top_k] (for cache policy
    // and traces); complete once forward() returns
    std::vector<int32_t> access;
    std::vector<int32_t> access_prev;   // the previous token's (ForwardRef copies it at token end)
    // statistics
    long hits = 0, misses = 0, gpu_misses = 0;   // misses: CPU-served; gpu_misses: read over PCIe
    double wait_s = 0, cpu_s = 0;     // host time waiting for the routing, and running the misses
    double cpu_by_nm[17] = {};        // miss time by the layer's miss count
    long layers_by_nm[17] = {};
};
MoeFastHost alloc_moe_fast_host(const Spec& s);
void free_moe_fast_host(MoeFastHost& h);   // also stops the miss server

// Lets the GPU serve floor(misses * frac) (at most max_per_layer) of each layer's misses by
// reading the experts straight from the arena in host memory (registered and mapped here), in
// parallel with the CPU serving the rest.
void enable_pcie_misses(MoeFastHost& h, const ExpertArena& arena, float frac, int max_per_layer);

// Switches h to doorbell mode: allocates the mailboxes and starts the miss server pinned to
// `cpu` (the pool's caller CPU; the pool must not be used by another thread meanwhile).
void start_doorbell(const Spec& s, MoeFastHost& h, int cpu);
// Brackets one decode token in doorbell mode: begin before enqueuing the layers, end after the
// stream has synchronised (throws if the miss server failed).
void doorbell_begin_token(MoeFastHost& h);
void doorbell_end_token(MoeFastHost& h, const Spec& s);

// Adaptive expert cache: decayed LFU with hysteresis and a per-token swap budget. Every
// access adds 1 to the (layer, expert) count; every decay_every tokens all counts are
// multiplied by decay. A missed expert is admitted when its count is >= admit and >= margin
// times the weakest resident's, up to `budget` uploads in flight.
struct CachePolicyConfig {
    float decay = 0.7f;
    int decay_every = 4;
    float admit = 2.0f;
    float margin = 1.5f;
    int budget = 32;
};
struct CacheManager;   // opaque; see moe_fast.cu

// Registers the arena with CUDA (pinned uploads) and seeds the counts with `prior` (e.g. the
// prompt's routing counts, [n_layer * n_expert]).
CacheManager* create_cache_manager(const Spec& s, ExpertCache& cache, const ExpertArena& arena, const CachePolicyConfig& cfg,
                                   const std::vector<uint32_t>& prior);
void destroy_cache_manager(CacheManager* m);
// Called once per decode token, after the token's kernels are enqueued on `stream` and before
// it synchronises: learns from the previous token's routing (h.access_prev), commits finished
// uploads, and schedules evictions (applied after this token) and uploads (started after it).
void cache_manager_step(CacheManager* m, const MoeFastHost& h, cudaStream_t stream);
struct CacheStats {
    long swaps = 0, committed = 0;
};
CacheStats cache_manager_stats(const CacheManager* m);

// The GPU experts of one token: yh[k] [n] = down_k(silu(gate_k x) * up_k x) for k < *hit_n
// (device int), expert k's planar blob at hit_ptr[k] (device array; a cache slot or mapped host
// memory); rows k >= *hit_n are not written. x [n] float; activations are quantized to int8
// per 64 values inside. scratch holds moe_hits_scratch_bytes(K, ff) bytes. Needs n % 512 == 0,
// ff % 64 == 0, ff <= 4096.
size_t moe_hits_scratch_bytes(int K, int ff);
void moe_hits(const uint8_t* const* hit_ptr, const int32_t* hit_n, int K, const float* x, int n, int ff, void* scratch, float* yh,
              cudaStream_t stream);

// One token: out [d_model] = routed experts (hits on the GPU, misses on the CPU) + gated shared
// expert. Same math as moe_block.
void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out);

}  // namespace flashrt::qwen4exp
