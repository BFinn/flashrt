// SPDX-License-Identifier: Apache-2.0
// The fast MoE path for decode: a VRAM expert cache (experts in ggml's Q2_0 layout, one slot
// each), routing on the GPU, cache hits as two grouped mat-vecs (fused gate+up, then down) and
// misses on the CPU pool, overlapped with the GPU work.
#pragma once

#include "arch/qwen4exp/blocks.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace flashrt::qwen4exp {

struct ExpertCache {
    int n_slots = 0;
    size_t slot_bytes = 0;            // one expert: gate | up | down, ggml Q2_0 blocks
    uint8_t* slots = nullptr;         // device, n_slots * slot_bytes
    int32_t* table_dev = nullptr;     // [n_layer][n_expert] -> slot, or -1
    std::vector<int32_t> table;       // host mirror of table_dev
    std::vector<int32_t> owner;       // slot -> layer * n_expert + expert, or -1
    uint8_t* staging = nullptr;       // device staging for one planar blob
};
ExpertCache alloc_expert_cache(const Spec& s, int n_slots);
void free_expert_cache(ExpertCache& c);

// Uploads the experts of `order` ((layer, expert), best first) into the free slots until the
// cache is full, converting the arena's planar blobs to ggml layout on the GPU.
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
    // statistics
    long hits = 0, misses = 0;
    double wait_s = 0, cpu_s = 0;     // host time waiting for the routing, and running the misses
    double cpu_by_nm[17] = {};        // miss time by the layer's miss count
    long layers_by_nm[17] = {};
};
MoeFastHost alloc_moe_fast_host(const Spec& s);
void free_moe_fast_host(MoeFastHost& h);   // also stops the miss server

// Switches h to doorbell mode: allocates the mailboxes and starts the miss server pinned to
// `cpu` (the pool's caller CPU; the pool must not be used by another thread meanwhile).
void start_doorbell(const Spec& s, MoeFastHost& h, int cpu);
// Brackets one decode token in doorbell mode: begin before enqueuing the layers, end after the
// stream has synchronised (throws if the miss server failed).
void doorbell_begin_token(MoeFastHost& h);
void doorbell_end_token(MoeFastHost& h);

// One token: out [d_model] = routed experts (hits on the GPU, misses on the CPU) + gated shared
// expert. Same math as moe_block.
void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out);

}  // namespace flashrt::qwen4exp
