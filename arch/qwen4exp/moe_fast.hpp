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
    bool db_failed = false;           // a doorbell step failed (doorbell_failed)
    // PCIe misses (enable_pcie_misses): the GPU reads some misses straight from the mapped arena
    const uint8_t* arena_dev = nullptr;
    size_t arena_stride = 0;
    float pcie_frac = 0.0f;
    int pcie_max = 0;
    // Windows (speculative verify): up to max_window tokens per doorbell step; the mailboxes are
    // sized for it (set before start_doorbell). window_T: the current step's token count.
    int max_window = 1;
    int window_T = 1;
    // the experts each layer selected for the last step's tokens, [token][n_layer][top_k] (for
    // cache policy and traces); complete once forward() returns
    std::vector<int32_t> access;
    std::vector<int32_t> access_prev;   // the previous step's (ForwardRef copies it at step end)
    int access_prev_T = 0;              // tokens in access_prev
    // statistics
    long hits = 0, misses = 0, gpu_misses = 0;   // misses: CPU-served; gpu_misses: read over PCIe
    double wait_s = 0, cpu_s = 0;     // host time waiting for the routing, and running the misses
    double cpu_by_nm[17] = {};        // miss time by the layer's miss count
    long layers_by_nm[17] = {};
};
MoeFastHost alloc_moe_fast_host(const Spec& s, int max_window = 1);
void free_moe_fast_host(MoeFastHost& h);   // also stops the miss server

// Lets the GPU serve floor(misses * frac) (at most max_per_layer) of each layer's misses by
// reading the experts straight from the arena in host memory (registered and mapped here), in
// parallel with the CPU serving the rest.
void enable_pcie_misses(MoeFastHost& h, const ExpertArena& arena, float frac, int max_per_layer);

// Switches h to doorbell mode: allocates the mailboxes and starts the miss server pinned to
// `cpu` (the pool's caller CPU; the pool must not be used by another thread meanwhile).
void start_doorbell(const Spec& s, MoeFastHost& h, int cpu);
// Brackets one decode step (T tokens, T <= max_window) in doorbell mode: begin before enqueuing
// the layers, end after the stream has synchronised (throws if the miss server failed).
void doorbell_begin_token(MoeFastHost& h, int T = 1);
void doorbell_end_token(MoeFastHost& h, const Spec& s);
// True once a doorbell step failed (a timeout on either side, or the miss server stopped): the
// mailboxes and the miss server are out of step, and only a new process recovers.
bool doorbell_failed(const MoeFastHost& h);

// Adaptive expert cache: decayed LFU with hysteresis and a per-token swap budget. Every
// access adds 1 to the (layer, expert) count; every decay_every tokens all counts are
// multiplied by decay. A missed expert is admitted when its count is >= admit and >= margin
// times the weakest resident's, up to `budget` uploads in flight.
struct CachePolicyConfig {
    float decay = 0.7f;
    int decay_every = 4;
    float admit = 1.0f;    // 2 until sw100: 1 with margin 1.2 follows a generation that routes unlike its
    float margin = 1.2f;   // prompt sooner (window 9 +3.0%, wikitext +1.6%, teacher-forced; sw99, sw100)
    int budget = 64;       // uploads in flight: 64 since sw104 (8 until sw89, then 32)
    // The warm-up after a fill from a prompt's routing: the policy's counts start at seed_scale
    // times the prompt's, while the fill still follows them. At 1 an expert the answer needs could
    // not beat the weakest resident for ~70 tokens (sw99, sw104). With budget 64, teacher-forced:
    // window 9 +7.7%, wikitext +4.4%, window 9 with the MTP head +23.8% (sw104).
    float seed_scale = 0.03f;
};
struct CacheManager;   // opaque; see moe_fast.cu

// Registers the arena with CUDA (mapped, pinned: full-speed copies; idempotent) and returns its
// device pointer. About 1.5 s for the whole arena, so loaders call it up front: otherwise the
// first prefill chunk or the cache manager pays it.
const uint8_t* arena_register(const ExpertArena& arena);

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
// Windows (T tokens): x [T][n], hit_ptr [T][K], hit_n [T], yh [T][K][n].
size_t moe_hits_scratch_bytes(int K, int ff, int T = 1);
void moe_hits(const uint8_t* const* hit_ptr, const int32_t* hit_n, int K, const float* x, int n, int ff, void* scratch, float* yh,
              cudaStream_t stream, int T = 1);

// T tokens (T > 1: a speculative window, doorbell mode only): out [T][d_model] = routed experts
// (hits on the GPU, misses on the CPU, each missed expert read once for the window) + gated
// shared expert. Same math as moe_block.
void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out, int T = 1);

}  // namespace flashrt::qwen4exp
