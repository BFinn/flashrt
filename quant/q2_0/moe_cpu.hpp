// SPDX-License-Identifier: Apache-2.0
// The CPU half of one MoE layer: every routed expert that missed the VRAM cache, computed by
// a CpuPool with each expert's rows split across the workers.
//
// Phase A takes (miss, 32-row chunk) items from a shared counter: gate and up rows, SwiGLU,
// and the Q8 quantization of that 32-value block of the hidden vector (activation blocks are
// independent, so no barrier is needed inside the phase). After one barrier, phase B takes
// 64-row chunks of the output: the down rows of every miss, weighted and summed. Output rows
// are disjoint between items, so there are no atomics.
#pragma once

#include "core/cpu_pool.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cstddef>
#include <cstdint>

namespace flashrt::q2_0 {

// A routed expert that missed the VRAM cache, with the window's tokens routed to it.
struct Miss {
    const uint8_t* blob;   // repacked expert (expert_bytes)
    int n_tok;             // 1..4
    int tok[4];            // token index within the window
    float w[4];            // routing weight per token
};

size_t moe_cpu_scratch_bytes(ExpertShape s, int max_miss, int n_workers);

// out[t * ldo + r] = sum over misses routed to token t of w * expert(x[t])[r], for every
// t < n_tok_window and r < d_model; tokens with no misses get zeros. scratch holds
// moe_cpu_scratch_bytes(s, n_miss, pool.size()) bytes.
void moe_cpu(CpuPool& pool, ExpertShape s, const Miss* miss, int n_miss, const Q8Act* x, int n_tok_window, float* out,
             int ldo, void* scratch);

}  // namespace flashrt::q2_0
