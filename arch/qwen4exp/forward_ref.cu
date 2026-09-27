// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/forward_ref.hpp"

#include "core/gguf.hpp"

#include <future>
#include <set>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
float* dalloc(size_t elems) {
    float* p = nullptr;
    ck(cudaMalloc(&p, elems * 4), "cudaMalloc forward buffer");
    return p;
}
}  // namespace

ForwardRef::ForwardRef(const Gguf& g, const Spec& s, const GpuWeights& w, const ExpertArena& arena, CpuPool& pool,
                       int max_ctx, int max_batch)
    : s_(s), w_(w), max_batch_(max_batch) {
    ple_ = parse_ple(g);
    reader_ = std::make_unique<RowReader>(g.shards[ple_.table_shard], ple_.table_offset, ple_.row_bytes, 16);
    ple_host_.ple = &ple_;
    ple_host_.reader = reader_.get();
    moe_host_.arena = &arena;
    moe_host_.pool = &pool;
    counts_.assign(size_t(s.n_layer) * s.n_expert, 0);
    moe_host_.counts = &counts_;
    scratch_ = alloc_block_scratch(s, max_batch);
    ck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    gdn_.resize(s.n_layer);
    kv_.resize(s.n_layer);
    ple_state_.resize(s.n_layer);
    for (int il : s.gdn_layers) gdn_[il] = alloc_gdn_state(s);
    for (int il : s.qsa_layers) kv_[il] = alloc_qsa_cache(s, max_ctx);
    for (int il : s.ple_layers) ple_state_[il] = alloc_ple_state(s, ple_);
    const size_t n = s.d_model, hc = s.hc_count, B = max_batch;
    emb_ = dalloc(B * n);
    x_ = dalloc(B * hc * n);
    mixed_ = dalloc(B * n);
    inject_ = dalloc(B * hc);
    blk_ = dalloc(B * n);
    pemb_ = dalloc(B * n);
    norm_ = dalloc(B * n);
    ck(cudaMalloc(&argmax_dev_, 4), "cudaMalloc argmax");
    ck(cudaHostAlloc(&argmax_host_, 4, cudaHostAllocDefault), "cudaHostAlloc argmax");
}

ForwardRef::~ForwardRef() {
    for (int il : s_.gdn_layers) free_gdn_state(gdn_[il]);
    for (int il : s_.qsa_layers) free_qsa_cache(kv_[il]);
    for (int il : s_.ple_layers) free_ple_state(ple_state_[il]);
    free_block_scratch(scratch_);
    for (float* p : {emb_, x_, mixed_, inject_, blk_, pemb_, norm_}) cudaFree(p);
    if (ple_host_.raw_dev) cudaFree(ple_host_.raw_dev);
    if (ple_host_.raw_pinned) cudaFreeHost(ple_host_.raw_pinned);
    cudaFree(argmax_dev_);
    cudaFreeHost(argmax_host_);
    cudaStreamDestroy(stream_);
}

int32_t ForwardRef::argmax(const float* logits_row_dev) {
    argmax_dev(stream_, logits_row_dev, s_.n_vocab, argmax_dev_);
    ck(cudaMemcpyAsync(argmax_host_, argmax_dev_, 4, cudaMemcpyDeviceToHost, stream_), "argmax to host");
    ck(cudaStreamSynchronize(stream_), "argmax");
    return *argmax_host_;
}

void ForwardRef::reset() {
    for (int il : s_.gdn_layers) reset_gdn_state(s_, gdn_[il], stream_);
    for (int il : s_.ple_layers) reset_ple_state(s_, ple_, ple_state_[il], stream_);
    for (int il : s_.qsa_layers)   // K/V and pooled keys are overwritten as positions advance; the ring is not
        ck(cudaMemsetAsync(kv_[il].idx_ring, 0, size_t(s_.qsa_block) * s_.idx_dim * 4, stream_), "memset ring");
    ck(cudaStreamSynchronize(stream_), "reset");
    pos_ = 0;
}

void ForwardRef::forward(const int32_t* seq, int T, int out_from, float* logits_dev) {
    if (T < 1 || T > max_batch_) throw std::runtime_error("ForwardRef: bad batch size");
    const Spec& s = s_;
    const int n = s.d_model, hc = s.hc_count;
    const BlockCtx c{s, w_, scratch_, stream_};
    // the PLE rows come from the SSD: read them on another thread while the embedding and the
    // layers before the first PLE layer are enqueued
    std::future<void> ple_rows;
    if (!s.ple_layers.empty()) ple_rows = std::async(std::launch::async, [&] { ple_fetch(ple_host_, seq, pos_, T); });
    embed(c, seq + pos_, T, emb_);
    for (int t = 0; t < T; ++t)
        for (int st = 0; st < hc; ++st)
            ck(cudaMemcpyAsync(x_ + (size_t(t) * hc + st) * n, emb_ + size_t(t) * n, size_t(n) * 4, cudaMemcpyDeviceToDevice, stream_),
               "hc init");
    const bool db = T == 1 && fast_cache_ && fast_host_->doorbell;
    if (db) doorbell_begin_token(*fast_host_);
    for (int il = 0; il < s.n_layer; ++il) {
        if (ple_rows.valid() && il == s.ple_layers.front()) {
            ple_rows.get();
            ple_upload(c, ple_host_, T, pemb_);
        }
        for (int pl : s.ple_layers)
            if (pl == il) ple_block(c, il, ple_, pemb_, x_, T, ple_state_[il]);
        hc_mix(c, il, 0, x_, T, mixed_, inject_);
        if (s.mixer[il] == Mixer::QSA) qsa_mixer(c, il, mixed_, T, pos_, kv_[il], blk_);
        else gdn_mixer(c, il, mixed_, T, gdn_[il], blk_);
        hc_combine(c, x_, blk_, inject_, T);
        hc_mix(c, il, 1, x_, T, mixed_, inject_);
        if (T == 1 && fast_cache_) moe_block_fast(c, il, mixed_, *fast_cache_, *fast_host_, blk_);
        else moe_block(c, il, mixed_, T, moe_host_, blk_);
        hc_combine(c, x_, blk_, inject_, T);
    }
    if (out_from < T && logits_dev) {
        const int R = T - out_from;
        hc_mix(c, -1, 2, x_ + size_t(out_from) * hc * n, R, norm_, nullptr);
        head_logits(c, norm_, R, logits_dev);
    }
    ck(cudaStreamSynchronize(stream_), "forward");
    if (db) doorbell_end_token(*fast_host_);
    pos_ += T;
}

}  // namespace flashrt::qwen4exp
