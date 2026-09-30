// SPDX-License-Identifier: Apache-2.0
// The MoE for prefill chunks: a chunk of a few thousand tokens routes to essentially every
// expert of a layer, so the layer's whole expert slice streams from the host arena to the GPU
// (one copy, 676 MB, while the previous layer computes) and runs as grouped int8 tensor-core
// products read straight from the arena's planar Q2_0 (kernels/cuda/moe_q2.h), then the
// routing-weighted sum and the gated shared expert. Same math as moe_block (the CPU reference
// path). FLASHRT_MOE_Q2MMA=0 converts the slice to ggml's Q2_0 layout and uses ggml's MMQ
// (gemm::moe) instead.
#pragma once

#include "arch/qwen4exp/blocks.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace flashrt::qwen4exp {

struct ExpertStream;

// max_tokens: the largest chunk; the buffers (two planar layer slices, the chunk's routing and
// expert activations; with the MMQ path, a converted slice too) are allocated here. Registers the arena with CUDA
// for full-speed copies (once per process).
ExpertStream* create_expert_stream(const Spec& s, const ExpertArena& arena, int max_tokens);
void destroy_expert_stream(ExpertStream* es);
size_t expert_stream_bytes(const ExpertStream* es);   // device memory held
// What create_expert_stream(s, arena, max_tokens) allocates.
size_t expert_stream_bytes_for(const Spec& s, const ExpertArena& arena, int max_tokens);

// Starts copying layer il's experts into the free slice buffer (on the stream's own copy
// stream); moe_block_stream does it for the next layer, so a chunk only waits for its first.
void expert_stream_prefetch(ExpertStream* es, int il);

// out [T][d_model] = routed experts + gated shared expert for x [T][d_model], T <= max_tokens.
// counts ([n_layer][n_expert], device), if given, gets each selection.
void moe_block_stream(const BlockCtx& c, int il, const float* x, int T, ExpertStream& es, float* out, uint32_t* counts,
                      uint32_t* tail_counts = nullptr, int tail_from = 1 << 30);

}  // namespace flashrt::qwen4exp
