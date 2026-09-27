// SPDX-License-Identifier: Apache-2.0
// qwen4exp forward blocks on the GPU, correctness first: plain kernels plus flashrt::gemv,
// one call per block, each checked against llama.cpp's intermediates (tools/fr_parity).
// Fusion and graph capture come after parity.
//
// Layouts (row-major, token-major): residual streams x[T][hc][d_model]; block activations
// [T][d_model]; inject weights [T][hc]. Device pointers throughout.
#pragma once

#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace flashrt::qwen4exp {

// Device scratch shared by the blocks of one forward pass.
struct BlockScratch {
    float* f32 = nullptr;     // general float scratch
    size_t f32_elems = 0;
    void* q8 = nullptr;       // Q8_1 activations for gemv
    size_t q8_bytes = 0;
};
BlockScratch alloc_block_scratch(const Spec& s, int max_tokens);
void free_block_scratch(BlockScratch& b);

struct BlockCtx {
    const Spec& s;
    const GpuWeights& w;
    BlockScratch& scratch;
    cudaStream_t stream;
};

// W x for T tokens (any T): gemv in chunks of up to 8 tokens. x [T][cols], y [T][rows].
void linear(const BlockCtx& c, const GpuTensor& W, const float* x, float* y, int T);

// Hyper-connection mix. which: 0 = before the mixer (hc_attn_*), 1 = before the MoE
// (hc_ffn_*), 2 = the head (output_hc_*, no inject). Writes mixed [T][d_model], inject
// [T][hc] (unless which == 2), and optionally xn [T][hc*d_model] (the grouped-norm output).
void hc_mix(const BlockCtx& c, int il, int which, const float* x, int T, float* mixed, float* inject, float* xn_out = nullptr);

// Hyper-connection combine: x[t][s][:] += out[t][:] * 2*sigmoid(inject[t][s] / hc).
void hc_combine(const BlockCtx& c, float* x, const float* out, const float* inject, int T);

}  // namespace flashrt::qwen4exp
