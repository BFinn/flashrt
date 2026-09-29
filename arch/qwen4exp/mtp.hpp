// SPDX-License-Identifier: Apache-2.0
// qwen4exp MTP (NextN) draft head: one extra decoder block trained to predict the token after
// next. Semantics follow llama.cpp's qwen4exp graph_mtp (src/models/qwen4exp.cpp, MIT; upstream
// PR #28243):
//   - input at position p: the target's final hyper-connection streams h_{p-1} [hc][n] (before
//     the output mix) and the token x_p;
//   - per stream: eh_proj(concat(enorm(embed(x_p)), hnorm(h_{p-1}))) -> streams [hc][n];
//   - one block: hc mix, QSA attention (its own KV cache and indexer), hc combine, hc mix, a
//     512-expert top-k MoE with a shared expert, hc combine -> h_out, the h of the next draft step;
//   - the nextn.hc_head mix, then the target's LM head -> logits for position p + 1.
// llama.cpp runs the draft block's attention dense; flashrt uses its indexer (QSA), which is the
// same below the selection width.
//
// The weights come from the draft GGUF (the "-noembd" export: token embedding and LM head are
// the target's). Everything lives in VRAM, routed on the GPU. The experts are requantized from
// the GGUF's Q8_0 at load to expert_bits: 4 (Q4_0, half the VRAM), 2 (ggml Q2_0) or 8 (as is);
// only draft acceptance depends on it. With set_vocab, the head covers a subset of the vocabulary (the drafter's argmax
// can only pick those tokens), a gathered copy of those rows of the target's LM head.
#pragma once

#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "kernels/cuda/sample.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

// Q8_0 blocks, two per 64 values (68 bytes), to ggml Q2_0 blocks (18 bytes) on `threads` threads:
// the head's experts at --mtp-bits 2, converted at load.
void convert_q8_0_to_q2_0(const uint8_t* src, uint8_t* dst, size_t nblocks64, int threads);

class MtpHead {
public:
    // g: the draft GGUF; target, target_w: the target model (embedding, LM head, shapes). The KV
    // cache is sized and formatted like the target's (kv_q8, kv_hot_blocks: see ForwardRef).
    MtpHead(const Gguf& g, const Spec& target, const GpuWeights& target_w, cudaStream_t stream, int max_ctx, int max_batch,
            bool kv_q8 = false, int kv_hot_blocks = 0, int expert_bits = 4);
    ~MtpHead();
    MtpHead(const MtpHead&) = delete;
    MtpHead& operator=(const MtpHead&) = delete;

    // Runs positions pos0 .. pos0 + T - 1. h_prev [T][hc][n] (device) holds the streams of the
    // position before each (the target's, or this head's h_out() for a chained draft; h_out()
    // itself may be passed); tokens (host) the tokens at the positions. Writes the KV cache at
    // those positions (re-running a position overwrites it), h_out() [T][hc][n], and the logits
    // of rows [out_from, T) to logits_dev [T - out_from][vocab()].
    void forward(const float* h_prev, const int32_t* tokens, int T, int pos0, int out_from, float* logits_dev);
    const float* h_out() const { return x_; }

    // The drafts of one round, k of them (<= 8), in one host sync: the first from chain_logits()
    // (the logits of the last forward() row, which must be row `row` of h_out(), at position
    // pos - 1), then k - 1 chained steps at positions pos .., each replaying one captured graph
    // that feeds its own draft and streams to the next.
    std::vector<int32_t> draft_chain(int row, int pos, int k);
    // Sampled drafts (temperature > 0): each draft is drawn from the head's logits through the
    // sampler chain p, with its q kept on the device for sample::spec_verify (q_ids / q_p / q_n,
    // [k][kMaxTopK]), the drafts at drafts_dev(). Temperature <= 0: argmax drafts again.
    void set_draft_sampling(const sample::Params& p, uint64_t seed);
    bool sampled() const { return sampled_; }
    const int32_t* drafts_dev() const { return chain_drafts_; }
    const int32_t* q_ids() const { return q_ids_; }
    const float* q_p() const { return q_p_; }
    const int32_t* q_n() const { return q_n_; }
    float* chain_logits() { return chain_logits_; }

    // Restricts the head to these token ids (empty: the full vocabulary again).
    void set_vocab(const std::vector<int32_t>& ids);
    // Allocates the trimmed head for up to n tokens now, so later set_vocab calls reuse it (a
    // server sets a vocabulary per request and must not reallocate VRAM the expert cache took).
    void reserve_vocab(int n);
    int vocab() const { return vocab_ids_.empty() ? ts_.n_vocab : int(vocab_ids_.size()); }
    const std::vector<int32_t>& vocab_ids() const { return vocab_ids_; }   // empty: the full vocabulary
    // The drafted token of one logits row (GPU argmax, mapped back to a token id); with p_top, also
    // its probability under the head's softmax (over vocab()).
    int32_t argmax(const float* logits_row_dev, float* p_top = nullptr);

    void reset();   // a new sequence
    // During a chunked prefill ending at end_pos: a VRAM mirror of the head's KV (hot-set mode;
    // the target's chunks have theirs), so the head's pass over the prompt attends on tensor
    // cores instead of through the hot set, whose cost grows with depth (P-1). prefill_end()
    // frees it; both are no-ops without a hot set.
    // Also, for the prefill's length, a second buffer set for calls of up to chunk_rows() rows:
    // forward() with more than max_batch rows uses it, and runs the MoE as grouped expert GEMMs
    // over the whole call (every expert read once per call) instead of 8-row mat-vec slices.
    // chunk_input() holds a call's h_prev rows. FLASHRT_MTP_CHUNK=0: batches of max_batch as before.
    void prefill_begin(int pos, int end_pos);
    void prefill_end();
    int chunk_rows() const { return chunk_.cap; }
    float* chunk_input() { return chunk_.hin; }
    // The indexer ring and the streams the next catch-up starts from (h [hc][n]): what a
    // restored checkpoint of the target also needs from the head.
    void save_checkpoint(const float* h_dev);
    void reserve_checkpoint();   // allocate it now (save_checkpoint otherwise does on first use)
    void restore_checkpoint(float* h_dev);
    // The same to or from a caller's buffer of checkpoint_bytes() (device or pinned host memory).
    size_t checkpoint_bytes() const;
    void save_checkpoint_to(void* dst, const float* h_dev);
    void restore_checkpoint_from(const void* src, float* h_dev);
    // The head's state after a prefill of pos positions (its KV cache, and the target's streams
    // at pos - 1, h_carry_dev [hc][n]), in a file of its own; load returns pos.
    void save_state(const std::string& path, int pos, const float* h_carry_dev);
    int load_state(const std::string& path, float* h_carry_dev);
    size_t weight_bytes() const { return w_.device_bytes() + exp_bytes_ + head_bytes_; }
    int layer() const { return il_; }

private:
    void enqueue(const float* h_prev, const int32_t* tokens, int T, int pos0, int out_from, float* logits_dev, const int32_t* dp);
    void moe(const BlockCtx& c, const float* x, int T, float* out);
    // per-call buffers: `dec_` for decode, drafting and small batches (the draft chain's graphs
    // are captured on it), `chunk_` during a chunked prefill; the members below point at one set
    struct Bufs {
        float *x = nullptr, *emb = nullptr, *en = nullptr, *hn = nullptr, *cat = nullptr, *mixed = nullptr, *inject = nullptr,
              *blk = nullptr, *norm = nullptr, *wts = nullptr, *hid = nullptr, *yd = nullptr, *sg = nullptr, *su = nullptr,
              *sh = nullptr, *gate = nullptr, *logits_e = nullptr;
        int32_t* ids = nullptr;
        void *xq = nullptr, *hq = nullptr;
        // grouped MoE (chunk set only): gate and up rows [T][K][ff], the gemm workspace, h_prev rows
        float *hg = nullptr, *hu = nullptr, *hin = nullptr;
        void* ws = nullptr;
        size_t ws_bytes = 0;
        BlockScratch scratch;
        int cap = 0;
    };
    void alloc_bufs(Bufs& b, int T, bool chunk);
    void free_bufs(Bufs& b);
    void use_bufs(Bufs& b);
    Bufs dec_, chunk_;
    static constexpr int kChunkRows = 1024;   // rows per call in a chunked prefill (~450 MiB of buffers)
    BlockScratch* scr_ = nullptr;   // the current set's scratch
    void load_experts_q4(const Gguf& g, bool q2);   // Q4_0, or Q2_0 when q2

    Spec s_;                 // the target's spec with the draft layer appended
    const Spec& ts_;
    const GpuWeights& tw_;
    GpuWeights w_;
    int il_ = 0;
    int max_batch_;
    cudaStream_t stream_;
    QsaCache kv_;
    float *x_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *hn_ = nullptr, *cat_ = nullptr, *mixed_ = nullptr,
          *inject_ = nullptr, *blk_ = nullptr, *norm_ = nullptr;
    // MoE scratch: routing [T][K], Q8_1 activations, the experts' hidden and output rows
    int32_t* ids_ = nullptr;
    float* wts_ = nullptr;
    void* xq_ = nullptr;
    void* hq_ = nullptr;
    float *hid_ = nullptr, *yd_ = nullptr, *sg_ = nullptr, *su_ = nullptr, *sh_ = nullptr, *gate_ = nullptr, *logits_e_ = nullptr;
    float *hg_ = nullptr, *hu_ = nullptr;   // non-null in the chunk set: the grouped MoE
    void* mws_ = nullptr;
    size_t mws_bytes_ = 0;
    // experts requantized at load (gate, up, down), outside w_
    GpuTensor exps_[3];
    void* exp_dev_ = nullptr;
    size_t exp_bytes_ = 0;
    // trimmed head
    std::vector<int32_t> vocab_ids_;
    GpuTensor head_;
    size_t head_bytes_ = 0;
    int head_cap_ = 0;   // rows the head buffer holds
    int32_t *amax_dev_ = nullptr, *amax_host_ = nullptr;   // [index, token, p as float bits]
    // draft chain (graph mode): params [token, position, -, step], drafts, the step's input streams
    int32_t *chain_dp_ = nullptr, *chain_drafts_ = nullptr;
    float *h_in_ = nullptr, *chain_logits_ = nullptr;
    cudaGraphExec_t chain_graph_ = nullptr;
    cudaGraphExec_t chain_graph_s_ = nullptr;   // the sampled-draft chain
    bool sampled_ = false;
    sample::DraftCfg dcfg_host_{};
    sample::DraftCfg* dcfg_dev_ = nullptr;
    int32_t *q_ids_ = nullptr, *q_n_ = nullptr;
    float* q_p_ = nullptr;
    int32_t chain_init_[4] = {};   // host source of the chain's parameter reset (stable for the async copy)
    BlockScratch chain_scratch_;   // the buffers the chain graph was captured with
    float* ckpt_ = nullptr;         // [ring | h]
};

}  // namespace flashrt::qwen4exp
