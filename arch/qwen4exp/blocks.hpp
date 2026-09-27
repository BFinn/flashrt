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

// GDN (gated delta net) recurrent state of one layer for one sequence: the delta-rule state
// S[head][j][i] (value row i fastest; flashrt's own layout) and the causal-conv history of
// the last (conv - 1) qkv inputs, [conv-1][channels]. Zero at the start of a sequence.
struct GdnState {
    float* S = nullptr;
    float* conv = nullptr;
};
GdnState alloc_gdn_state(const Spec& s);
void reset_gdn_state(const Spec& s, GdnState& st, cudaStream_t stream);
void free_gdn_state(GdnState& st);

// GDN mixer for T consecutive tokens (state advances token by token). x [T][d_model] is the
// hyper-connection mix; out [T][d_model]. o_inner, if given, gets the delta-rule output
// before the gated norm, [T][heads][state] (llama.cpp's "attn_output" in GDN layers).
void gdn_mixer(const BlockCtx& c, int il, const float* x, int T, GdnState& st, float* out, float* o_inner = nullptr);

// KV cache of one QSA layer for one sequence: post-norm, post-rope K and V, [cell][kv_head][dim].
// Cell index = position. Stored as F32 holding values rounded to F16, like the parity
// reference's F16 KV cache (a packed F16 / Q8 layout comes with the fused kernels).
struct QsaCache {
    float* K = nullptr;
    float* V = nullptr;
    int capacity = 0;
};
QsaCache alloc_qsa_cache(const Spec& s, int capacity);
void free_qsa_cache(QsaCache& kv);

// QSA mixer for T consecutive tokens at positions pos0 .. pos0+T-1: projections, q/k RMS
// norms, rope, KV append, attention and the sigmoid output gate, then the output projection.
// The indexer's top-k is not implemented yet: while the context fits the selection width
// (indexer.top_k + block - 1 cells) QSA selects every cell, i.e. it is dense causal attention,
// and longer contexts throw.
void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out);

}  // namespace flashrt::qwen4exp
