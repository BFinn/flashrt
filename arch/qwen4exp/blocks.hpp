// SPDX-License-Identifier: Apache-2.0
// qwen4exp forward blocks on the GPU: norms, hyper-connections (hc.cu), GDN (gdn.cu), QSA
// attention and its indexer (qsa.cu), batch routing and the reference MoE (moe_ref.cu), PLE
// (ple.cu); scratch, linear layers and embeddings (blocks.cu). They began as plain kernels, one call per block,
// checked against llama.cpp's intermediates (tools/fr_parity). Most now also have fused decode
// and tensor-core prefill variants, checked the same way and by the KLD gate (tools/fr_kld).
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
#include <cstdio>
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
    // matrix-matrix products (linear() with many tokens): the gemm workspace and a Q3_K copy of a
    // Q3R matrix, grown on demand (never used inside a captured graph)
    void* gemm_ws = nullptr;
    size_t gemm_ws_bytes = 0;
    void* q3k_tmp = nullptr;
    size_t q3k_tmp_bytes = 0;
    void* q8_tmp = nullptr;   // a Q3_K matrix as Q8_0 (lossless), for the faster Q8_0 MMQ
    size_t q8_tmp_bytes = 0;
    void* gdn_ws = nullptr;   // the chunked delta rule's per-slab products (gdn_chunk_ws_bytes)
    size_t gdn_ws_bytes = 0;
    // Q8P hc down and up of one mix, dequantized to BF16 for hc_mix outside the fused decode
    // kernels (allocated with the scratch: a window may take that path inside a graph)
    void* hc_bf16 = nullptr;
    size_t hc_bf16_bytes = 0;
    int32_t* tok_dev = nullptr;   // token ids of a many-token embed()
    size_t tok_cap = 0;
    // A new value whenever a buffer a captured graph bakes in is (re)allocated: f32, q8 and hc_bf16
    // (alloc_block_scratch), the indexer lists and attention partials (qsa_scratch_reserve). A
    // graph captured at another version is stale.
    uint64_t version = 0;
};
BlockScratch alloc_block_scratch(const Spec& s, int max_tokens);
size_t block_scratch_bytes(const Spec& s, int max_tokens);   // what alloc_block_scratch takes
uint64_t new_scratch_version();   // for BlockScratch::version: unique in the process
void free_block_scratch(BlockScratch& b);

struct BlockCtx {
    const Spec& s;
    const GpuWeights& w;
    BlockScratch& scratch;
    cudaStream_t stream;
    // Graph mode (a decode step of T <= kMaxGraphTokens tokens): device int32 [token 0, position
    // of token 0, doorbell seq, token 1, token 2, ...]. Kernels read the per-step values from here
    // instead of launch arguments, so a captured CUDA graph can be replayed every step; the QSA
    // path always runs the indexer selection (dense below its width).
    const int32_t* dparams = nullptr;
};
constexpr int kMaxGraphTokens = 8;

// W x for T tokens (any T): gemv in chunks of up to 8 tokens, or from kGemmMinTokens on the
// matrix-matrix path (kernels/cuda/ggml_gemm.h). x [T][cols], y [T][rows].
constexpr int kGemmMinTokens = 16;
void linear(const BlockCtx& c, const GpuTensor& W, const float* x, float* y, int T);

// Up to four BF16 mat-vecs of one input in one launch (decode's small projections): output i is
// y_i[t * rows_i + r], with an optional epilogue (GDN gates: g = softplus(z + p0[r]) * p1[r], or
// sigmoid). Only for T <= gemv::kMaxTokens and BF16 matrices of one row length (a multiple of
// 256): linear_multi_ok says whether it applies; the caller falls back otherwise.
struct LinearOut {
    const GpuTensor* W = nullptr;
    float* y = nullptr;
    int epi = 0;   // 0: none, 1: GDN decay, 2: sigmoid
    const float* p0 = nullptr;
    const float* p1 = nullptr;
};
bool linear_multi_ok(const LinearOut* outs, int n, int T);
void linear_multi(const BlockCtx& c, const LinearOut* outs, int n, const float* x, int T);
// Decode epilogue fusions (sw74): a producer kernel writes the q8_1 activations of the next
// mat-vec directly (gemv::swiglu_q8_1, gemv::gated_rms_norm_q8_1). q8_act says whether W's decode
// mat-vec takes q8_1 activations (ggml MMVQ; not float, Q3R or Q8P).
bool q8_act(const GpuTensor& W, int T);
// linear() for n matrices of one input; the ones taking q8_1 activations share one quantization of
// x, and in prefill the BF16 products one conversion
void linear_shared(const BlockCtx& c, const GpuTensor* const* Ws, float* const* ys, int n, const float* x, int T);

// Hyper-connection mix. which: 0 = before the mixer (hc_attn_*), 1 = before the MoE
// (hc_ffn_*), 2 = the head (output_hc_*, no inject), 3 = an MTP block's head (blk.il.nextn.hc_head_*,
// no inject). Writes mixed [T][d_model], inject [T][hc] (for which < 2), and optionally xn
// [T][hc*d_model] (the grouped-norm output).
void hc_mix(const BlockCtx& c, int il, int which, const float* x, int T, float* mixed, float* inject, float* xn_out = nullptr);
// Test hook (tests/test_hc_decode): the decode hc mix on raw pointers, hc == 4, T <= 4; down as
// Q8P ([rank][4 n] int8, then fp16 scales per 32), inject and up BF16. part: [T][4][rank + 4].
// With comb_out ([T][n]) and comb_inject ([T][4]), x is first combined (x += out * 2 sigmoid(inject / 4)).
void hc_decode_raw(int T, float* x, const float* w_norm, const void* down_q8p, const void* inject_bf16, const void* up_bf16, float* xn,
                   float* part, float* mixed, float* inject, int n, int rank, float eps, cudaStream_t st, const float* comb_out = nullptr,
                   const float* comb_inject = nullptr);
// hc_combine(c, x, out, comb_inject, T) then hc_mix(c, il, which, x, ...), the combine fused into
// the mix's RMS norm in prefill (comb_inject may be inject: it is read first).
void hc_combine_mix(const BlockCtx& c, int il, int which, float* x, const float* out, const float* comb_inject, int T, float* mixed,
                    float* inject);

// y[row] = x[row] / rms(x[row]) * w[(row % groups) * n ..], for `rows` rows of n values.
void rms_norm_rows(const BlockCtx& c, const float* x, const float* w, float* y, int n, int groups, int rows);

// x[t][s][:] = emb[t][:] for every stream s (the residual streams' start).
void hc_init(const BlockCtx& c, const float* emb, float* x, int T);

// Top-k routing on the GPU (softmax over E <= 1024 logits per token, top k by probability, ties
// to the lower index, weights renormalised as moe_block does): ids, wts [T][k]; with counts
// ([E], device), each selection adds 1 there.
// With tail_counts, rows t >= tail_from also count there (the prompt's last tokens, for the cache's fill).
void moe_route_topk(cudaStream_t stream, const float* logits, int T, int E, int k, int32_t* ids, float* wts, uint32_t* counts,
                    uint32_t* tail_counts = nullptr, int tail_from = 1 << 30);

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

// Speculative windows: what one GDN layer keeps so that a call of T tokens can be rewound to its
// first n (gdn_rewind): the state and conv history from before the call, the call's conv inputs,
// and the delta rule's per-token inputs, from which the accepted tokens are replayed.
struct GdnWindow {
    int max_tokens = 0;
    float* S_bak = nullptr;      // [heads][state][state]
    float* conv_old = nullptr;   // [conv-1][channels]
    float* qkv = nullptr;        // [max_tokens][channels], the conv inputs
    float* conv = nullptr;       // [max_tokens][channels], normalised conv outputs
    float* g = nullptr;          // [max_tokens][heads]
    float* beta = nullptr;       // [max_tokens][heads]
};
GdnWindow alloc_gdn_window(const Spec& s, int max_tokens);
void free_gdn_window(GdnWindow& w);

// GDN mixer for T consecutive tokens (state advances token by token). x [T][d_model] is the
// hyper-connection mix; out [T][d_model]. o_inner, if given, gets the delta-rule output
// before the gated norm, [T][heads][state] (llama.cpp's "attn_output" in GDN layers). With win
// (T <= win->max_tokens), the call can be rewound.
void gdn_mixer(const BlockCtx& c, int il, const float* x, int T, GdnState& st, float* out, float* o_inner = nullptr,
               GdnWindow* win = nullptr);
// The delta rule of one GDN layer over T tokens (prefill), the state S [heads][128][128] in place.
// conv [T][channels]: the L2-normed q and k heads, then v; g, beta [T][heads] (g the log decay);
// o [T][heads][128]. chunked: the chunked (WY) form, with ws of gdn_chunk_ws_bytes(s); otherwise
// the column kernel (the recurrence token by token).
size_t gdn_chunk_ws_bytes(const Spec& s);
void gdn_delta_prefill(const Spec& s, float* S, const float* conv, const float* g, const float* beta, float* o, int T, bool chunked,
                       void* ws, cudaStream_t stream);
// The decode form of the same delta rule (T up to a verify window's length), the state in
// registers: S_in -> S_out (the same buffer in a plain call); with S_bak, the state before the
// call is saved there as well. Inputs as for gdn_delta_prefill; dk must be 128.
void gdn_delta_decode(const Spec& s, const float* S_in, float* S_out, float* S_bak, const float* conv, const float* g, const float* beta,
                      float* o, int T, cudaStream_t stream);
// Rewinds the last gdn_mixer call (T tokens, with win) to its first n tokens (n < T; n == T is a
// no-op): the state is replayed from the backup over n tokens, the conv history rebuilt.
void gdn_rewind(const BlockCtx& c, GdnState& st, const GdnWindow& win, int T, int n);

// KV cache of one QSA layer for one sequence: post-norm, post-rope K and V, [cell][kv_head][dim].
// Cell index = position. Stored as F32 holding values rounded to F16, like the parity
// reference's F16 KV cache (a packed F16 / Q8 layout comes with the fused kernels).
// The indexer keeps one pooled key per complete block (mean of the block's raw keys, RMS-normed,
// roped at the block's first position; as llama.cpp's pooled-key cache) and a ring of the last
// 2 * block raw keys (slot = position % (2 * block)), from which the next block is pooled. Two
// blocks' worth, so that re-running positions after a rejected speculative window (up to
// block + 2 tokens) never finds an older position's key overwritten.
struct QsaCache {
    // K, V [capacity][kv_heads][head_dim]: fp16 bits (llama.cpp's F16 cache), or with q8 int8 in
    // blocks of 32 with an fp16 scale each in Ks, Vs [capacity][kv_heads][head_dim / 32]
    // (llama.cpp's Q8_0 cache: d = amax / 127)
    void* K = nullptr;
    void* V = nullptr;
    uint16_t* Ks = nullptr;
    uint16_t* Vs = nullptr;
    bool q8 = false;
    // Hot-set mode (q8 only, hot_blocks > 0): the full cache lives in mapped host memory (h*, device
    // views; h*_host for the host side) and K/V/Ks/Vs above hold only hot_blocks blocks of
    // qsa_block cells, filled with CLOCK before each attention; slot_of_block [capacity / block]
    // maps a block to its slot or -1.
    int hot_blocks = 0;
    void *hK = nullptr, *hV = nullptr;
    uint16_t *hKs = nullptr, *hVs = nullptr;
    void *hK_host = nullptr, *hV_host = nullptr, *hKs_host = nullptr, *hVs_host = nullptr;
    int32_t* slot_of_block = nullptr;
    int32_t* block_of_slot = nullptr;
    uint8_t* refbit = nullptr;
    int32_t* clock_hand = nullptr;   // [hand, step]
    uint32_t* pinned = nullptr;      // per slot: the step that last selected it (never evicted in that step)
    int32_t* promo = nullptr;        // [count, (block, slot) ...] promotions of the current step
    int capacity = 0;
    // Prefill mirror (hot-set mode only, qsa_mirror_begin/end): a full-size VRAM copy, written
    // with the host store while prefill chunks run, so their attention reads VRAM.
    void *mK = nullptr, *mV = nullptr;
    uint16_t *mKs = nullptr, *mVs = nullptr;
    int mcap = 0;   // cells the mirror holds
    uint16_t* idx_pooled = nullptr;   // [capacity / block][idx_dim], fp16 (as the prefill scores rounded them; P-5)
    float* idx_ring = nullptr;     // [qsa_ring_slots][idx_dim], slot = position % qsa_ring_slots
};
inline int qsa_ring_slots(const Spec& s) { return 2 * s.qsa_block; }
QsaCache alloc_qsa_cache(const Spec& s, int capacity, bool q8 = false, int hot_blocks = 0);
// Starts a prefill mirror of `cells` cells (copying positions [0, pos) from the host store), or
// ends it; no-op without a hot set.
void qsa_mirror_begin(const Spec& s, QsaCache& kv, int pos, int cells, cudaStream_t stream);
void qsa_mirror_end(QsaCache& kv);
// Empties the hot set (every block misses until promoted again); no-op without one.
void reset_qsa_hot(const Spec& s, QsaCache& kv, cudaStream_t stream);
// fp16 rows -> Q8_0 rows (values [n_rows][dim], scales [n_rows][dim / 32]), on the GPU.
void qsa_h2q8_rows(const void* src_f16, void* dst_q8, void* dst_scales, long n_rows, int dim, cudaStream_t stream);
// The indexer's block selection for T tokens at positions pos0.. (qsa_mixer runs it; public for its
// test): from scores [T][ld] (one per 4-cell block up to the position), the cells to attend to in
// cells [T][ldc] and their number in counts [T] (-1: dense, the position is within `width`).
// cluster: CTAs per token, 8 or 1 (the same output); 0 chooses as qsa_mixer does.
void qsa_select(const float* scores, int ld, int32_t* cells, int32_t* counts, int ldc, int pos0, int T, int r, int nsel, int width,
                cudaStream_t stream, int cluster = 0);
// One QSA layer's cache for positions [0, pos) to (save) or from a state file: K and V (with
// scales when q8), pooled indexer keys, and the ring of the last qsa_block raw keys (stored at
// slot position % qsa_block). file_q8 is the file's KV format; an fp16 file loads into a q8
// cache (converted on the GPU), not the other way.
void qsa_state_io(FILE* f, const Spec& s, QsaCache& kv, int pos, bool save, bool file_q8, cudaStream_t stream);
// Bytes of one cell's K (or V) over all KV heads, values plus scales.
size_t qsa_cell_bytes(const Spec& s, bool q8);
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
// tokens (n_splits < 0: no partials, for the tensor-core prefill kernel); graph capture calls it
// first so nothing is allocated while capturing.
void qsa_scratch_reserve(const Spec& s, BlockScratch& bs, int T, int max_nb, int n_splits = 0);
void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out,
               float* gated_out = nullptr, std::vector<std::vector<int32_t>>* sel_out = nullptr);

// The MoE block, reference path (moe_ref.cu): the GPU computes the router logits and the shared
// expert; the host computes the routing (softmax over all experts, top-k, weights renormalised)
// and every routed expert with the CPU miss path (q2_0::moe_cpu over the host arena). It serves
// batches up to max_batch outside decode; decode and windows use the VRAM expert cache
// (moe_fast.hpp), prefill chunks stream the experts to the GPU (moe_stream.hpp).
struct MoeHost {
    const ExpertArena* arena = nullptr;
    CpuPool* pool = nullptr;
    std::vector<float> x, logits, out;         // host staging
    std::vector<uint8_t> act_mem, scratch;     // Q8 activations, moe_cpu scratch
    std::vector<uint32_t>* counts = nullptr;   // optional: routes per (layer * n_expert + expert)
    std::vector<uint32_t>* tail_counts = nullptr;   // optional: the same for rows t >= tail_from
    int tail_from = 1 << 30;
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
// Speculative windows for a PLE layer: the conv history before the call and the call's conv inputs.
struct PleWindow {
    int max_tokens = 0;
    float* hist_old = nullptr;   // as PleState::hist
    float* rows = nullptr;       // [max_tokens][hc*d_model]
};
PleWindow alloc_ple_window(const Spec& s, const Ple& p, int max_tokens);
void free_ple_window(PleWindow& w);
// x [T][hc][d_model] += gated value + conv(normalised gated value), per llama.cpp's build_ple.
// With win (T <= win->max_tokens), the call can be rewound by ple_rewind.
void ple_block(const BlockCtx& c, int il, const Ple& p, const float* emb, float* x, int T, PleState& st, PleWindow* win = nullptr);
void ple_rewind(const BlockCtx& c, int il, const Ple& p, PleState& st, const PleWindow& win, int T, int n);

// logits [T][n_vocab] = output.weight x norm [T][d_model]
void head_logits(const BlockCtx& c, const float* norm, int T, float* logits);

}  // namespace flashrt::qwen4exp
