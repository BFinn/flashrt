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

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

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
    float* chain_logits() { return chain_logits_; }

    // Restricts the head to these token ids (empty: the full vocabulary again).
    void set_vocab(const std::vector<int32_t>& ids);
    int vocab() const { return vocab_ids_.empty() ? ts_.n_vocab : int(vocab_ids_.size()); }
    // The drafted token of one logits row (GPU argmax, mapped back to a token id); with p_top, also
    // its probability under the head's softmax (over vocab()).
    int32_t argmax(const float* logits_row_dev, float* p_top = nullptr);

    void reset();   // a new sequence
    // The head's state after a prefill of pos positions (its KV cache, and the target's streams
    // at pos - 1, h_carry_dev [hc][n]), in a file of its own; load returns pos.
    void save_state(const std::string& path, int pos, const float* h_carry_dev);
    int load_state(const std::string& path, float* h_carry_dev);
    size_t weight_bytes() const { return w_.device_bytes() + exp_bytes_ + head_bytes_; }
    int layer() const { return il_; }

private:
    void enqueue(const float* h_prev, const int32_t* tokens, int T, int pos0, int out_from, float* logits_dev, const int32_t* dp);
    void moe(const BlockCtx& c, const float* x, int T, float* out);
    void load_experts_q4(const Gguf& g, bool q2);   // Q4_0, or Q2_0 when q2

    Spec s_;                 // the target's spec with the draft layer appended
    const Spec& ts_;
    const GpuWeights& tw_;
    GpuWeights w_;
    int il_ = 0;
    int max_batch_;
    cudaStream_t stream_;
    BlockScratch scratch_;
    QsaCache kv_;
    float *x_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *hn_ = nullptr, *cat_ = nullptr, *mixed_ = nullptr,
          *inject_ = nullptr, *blk_ = nullptr, *norm_ = nullptr;
    // MoE scratch: routing [T][K], Q8_1 activations, the experts' hidden and output rows
    int32_t* ids_ = nullptr;
    float* wts_ = nullptr;
    void* xq_ = nullptr;
    void* hq_ = nullptr;
    float *hid_ = nullptr, *yd_ = nullptr, *sg_ = nullptr, *su_ = nullptr, *sh_ = nullptr, *gate_ = nullptr, *logits_e_ = nullptr;
    // experts requantized at load (gate, up, down), outside w_
    GpuTensor exps_[3];
    void* exp_dev_ = nullptr;
    size_t exp_bytes_ = 0;
    // trimmed head
    std::vector<int32_t> vocab_ids_;
    GpuTensor head_;
    size_t head_bytes_ = 0;
    int32_t *amax_dev_ = nullptr, *amax_host_ = nullptr;   // [index, token, p as float bits]
    // draft chain (graph mode): params [token, position, -, step], drafts, the step's input streams
    int32_t *chain_dp_ = nullptr, *chain_drafts_ = nullptr;
    float *h_in_ = nullptr, *chain_logits_ = nullptr;
    cudaGraphExec_t chain_graph_ = nullptr;
    BlockScratch chain_scratch_;   // the buffers the chain graph was captured with
};

}  // namespace flashrt::qwen4exp
