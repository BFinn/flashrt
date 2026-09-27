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
    // statistics
    long hits = 0, misses = 0;
};
MoeFastHost alloc_moe_fast_host(const Spec& s);
void free_moe_fast_host(MoeFastHost& h);

// One token: out [d_model] = routed experts (hits on the GPU, misses on the CPU) + gated shared
// expert. Same math as moe_block.
void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out);

}  // namespace flashrt::qwen4exp
