// SPDX-License-Identifier: Apache-2.0
// qwen4exp forward blocks on the GPU, correctness first: plain kernels plus flashrt::gemv,
// one call per block, each checked against llama.cpp's intermediates (tools/fr_parity).
// Fusion and graph capture come after parity.
//
// Layouts (row-major, token-major): residual streams x[T][hc][d_model]; block activations
// [T][d_model]; inject weights [T][hc]. Device pointers throughout.
#pragma once

#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/ple.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/row_reader.hpp"
#include "core/cpu_pool.hpp"
#include "core/expert_arena.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flashrt::qwen4exp {

// Device scratch shared by the blocks of one forward pass.
struct BlockScratch {
    float* f32 = nullptr;     // general float scratch
    size_t f32_elems = 0;
    void* q8 = nullptr;       // Q8_1 activations for gemv
    size_t q8_bytes = 0;
    // QSA indexer: block scores [T][blocks] and selected cells [T][cells], grown on demand
    float* idx_scores = nullptr;
    size_t idx_scores_elems = 0;
    int32_t* idx_cells = nullptr;
    int32_t* idx_counts = nullptr;
    size_t idx_cells_elems = 0;
    // attention partials (split-K flash decode), grown on demand
    float* attn_part = nullptr;
    size_t attn_part_elems = 0;
};
BlockScratch alloc_block_scratch(const Spec& s, int max_tokens);
void free_block_scratch(BlockScratch& b);

struct BlockCtx {
    const Spec& s;
    const GpuWeights& w;
    BlockScratch& scratch;
    cudaStream_t stream;
    // Graph mode (decode, one token): device int32 [token, position, doorbell seq]. Kernels read
    // the per-token values from here instead of launch arguments, so a captured CUDA graph can be
    // replayed every token; the QSA path always runs the indexer selection (dense below its width).
    const int32_t* dparams = nullptr;
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
// The indexer keeps one pooled key per complete block (mean of the block's raw keys, RMS-normed,
// roped at the block's first position; as llama.cpp's pooled-key cache) and a ring of the last
// `block` raw keys, from which the next block is pooled.
struct QsaCache {
    uint16_t* K = nullptr;   // [capacity][kv_heads][head_dim] fp16 bits (the values are fp16-rounded anyway)
    uint16_t* V = nullptr;
    int capacity = 0;
    float* idx_pooled = nullptr;   // [capacity / block][idx_dim]
    float* idx_ring = nullptr;     // [block][idx_dim], slot = position % block
};
QsaCache alloc_qsa_cache(const Spec& s, int capacity);
void free_qsa_cache(QsaCache& kv);

// QSA mixer for T consecutive tokens at positions pos0 .. pos0+T-1: projections, q/k RMS
// norms, rope, KV append, the indexer, attention with the sigmoid output gate, and the output
// projection. Token q attends to every cell while q + 1 <= width (indexer.top_k + block - 1);
// beyond that, to the cells of the top ceil(width / block) blocks by indexer score, where the
// incomplete tail block always counts as one (llama.cpp's whole-block selection). Each token
// sees only the blocks complete at its own position.
// gated_out, if given, gets the gated attention output before the output projection,
// [T][heads * dim] (llama.cpp's "attn_gated"). sel_out, if given, gets each token's selected
// cells (empty when the token attends to every cell).
// Grows the QSA scratch (indexer scores for max_nb blocks, cells, attention partials) for T
// tokens; graph capture calls it first so nothing is allocated while capturing.
void qsa_scratch_reserve(const Spec& s, BlockScratch& bs, int T, int max_nb, int n_splits = 0);
void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out,
               float* gated_out = nullptr, std::vector<std::vector<int32_t>>* sel_out = nullptr);

// The MoE block, correctness path: the GPU computes the router logits and the shared expert;
// the host computes the routing (softmax over all experts, top-k, weights renormalised) and
// every routed expert with the CPU miss path (q2_0::moe_cpu over the host arena). The VRAM
// expert cache (GPU hits) comes after parity.
struct MoeHost {
    const ExpertArena* arena = nullptr;
    CpuPool* pool = nullptr;
    std::vector<float> x, logits, out;         // host staging
    std::vector<uint8_t> act_mem, scratch;     // Q8 activations, moe_cpu scratch
    std::vector<uint32_t>* counts = nullptr;   // optional: routes per (layer * n_expert + expert)
};
struct MoeTrace {                              // routing, for parity checks
    std::vector<int32_t> topk;                 // [T][top_k]
    std::vector<float> probs;                  // [T][top_k], before renormalisation
};
// out [T][d_model] = routed experts + gated shared expert, for the FFN-side mix x [T][d_model].
void moe_block(const BlockCtx& c, int il, const float* x, int T, MoeHost& h, float* out, MoeTrace* trace = nullptr);

// Token embeddings: out [T][d_model] = dequantized rows of token_embd.
// Whether embed() can run in graph mode (the token id read on the device) for these weights.
bool embed_graph_capable(const GpuWeights& w);
void embed(const BlockCtx& c, const int32_t* tokens, int T, float* out);

// The n-gram (PLE) layer. Rows are read from the SSD (RowReader), dequantized on the GPU;
// the depthwise conv keeps (kernel - 1) * ngram tokens of history per channel.
struct PleHost {
    const Ple* ple = nullptr;
    RowReader* reader = nullptr;
    std::vector<uint32_t> rows;
    std::vector<uint8_t> raw;
    void* raw_dev = nullptr;
    size_t raw_dev_bytes = 0;
    uint8_t* raw_pinned = nullptr;    // ple_fetch's destination (pinned, so the upload is async)
    size_t raw_pinned_bytes = 0;
};
struct PleState {
    float* hist = nullptr;   // [(kernel-1)*ngram][hc*d_model], oldest first
};
PleState alloc_ple_state(const Spec& s, const Ple& p);
void reset_ple_state(const Spec& s, const Ple& p, PleState& st, cudaStream_t stream);
void free_ple_state(PleState& st);

// emb [T][d_model]: the concatenated n-gram rows for tokens seq[pos0 .. pos0+T-1] (seq holds
// the whole sequence so far, for the n-gram context).
void ple_embed(const BlockCtx& c, PleHost& h, const int32_t* seq, int64_t pos0, int T, float* emb);
// The same in two halves, so the SSD read can overlap GPU work: ple_fetch (host only, any
// thread) reads the rows into h.raw_pinned; ple_upload enqueues the copy and the dequantize.
// h.raw_pinned must not be refetched until the stream has passed the upload.
void ple_fetch(PleHost& h, const int32_t* seq, int64_t pos0, int T);
void ple_upload(const BlockCtx& c, PleHost& h, int T, float* emb);

// out_dev[0] = index of the largest of x[0 .. n) (the lowest index on ties), on the GPU.
void argmax_dev(cudaStream_t stream, const float* x, int n, int32_t* out_dev);
// x [T][hc][d_model] += gated value + conv(normalised gated value), per llama.cpp's build_ple.
void ple_block(const BlockCtx& c, int il, const Ple& p, const float* emb, float* x, int T, PleState& st);

// logits [T][n_vocab] = output.weight x norm [T][d_model]
void head_logits(const BlockCtx& c, const float* norm, int T, float* logits);

}  // namespace flashrt::qwen4exp
