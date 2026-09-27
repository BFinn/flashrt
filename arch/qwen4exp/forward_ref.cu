// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/forward_ref.hpp"

#include "core/gguf.hpp"

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
}

ForwardRef::~ForwardRef() {
    for (int il : s_.gdn_layers) free_gdn_state(gdn_[il]);
    for (int il : s_.qsa_layers) free_qsa_cache(kv_[il]);
    for (int il : s_.ple_layers) free_ple_state(ple_state_[il]);
    free_block_scratch(scratch_);
    for (float* p : {emb_, x_, mixed_, inject_, blk_, pemb_, norm_}) cudaFree(p);
    if (ple_host_.raw_dev) cudaFree(ple_host_.raw_dev);
    cudaStreamDestroy(stream_);
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
    embed(c, seq + pos_, T, emb_);
    for (int t = 0; t < T; ++t)
        for (int st = 0; st < hc; ++st)
            ck(cudaMemcpyAsync(x_ + (size_t(t) * hc + st) * n, emb_ + size_t(t) * n, size_t(n) * 4, cudaMemcpyDeviceToDevice, stream_),
               "hc init");
    if (!s.ple_layers.empty()) ple_embed(c, ple_host_, seq, pos_, T, pemb_);
    for (int il = 0; il < s.n_layer; ++il) {
        for (int pl : s.ple_layers)
            if (pl == il) ple_block(c, il, ple_, pemb_, x_, T, ple_state_[il]);
        hc_mix(c, il, 0, x_, T, mixed_, inject_);
        if (s.mixer[il] == Mixer::QSA) qsa_mixer(c, il, mixed_, T, pos_, kv_[il], blk_);
        else gdn_mixer(c, il, mixed_, T, gdn_[il], blk_);
        hc_combine(c, x_, blk_, inject_, T);
        hc_mix(c, il, 1, x_, T, mixed_, inject_);
        moe_block(c, il, mixed_, T, moe_host_, blk_);
        hc_combine(c, x_, blk_, inject_, T);
    }
    if (out_from < T && logits_dev) {
        const int R = T - out_from;
        hc_mix(c, -1, 2, x_ + size_t(out_from) * hc * n, R, norm_, nullptr);
        head_logits(c, norm_, R, logits_dev);
    }
    ck(cudaStreamSynchronize(stream_), "forward");
    pos_ += T;
}

}  // namespace flashrt::qwen4exp
