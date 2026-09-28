// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/mtp.hpp"

#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemv.h"

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <typename T>
T* dalloc(size_t elems) {
    T* p = nullptr;
    ck(cudaMalloc(&p, elems * sizeof(T)), "cudaMalloc MTP buffer");
    return p;
}

// cat[t][s] = [en[t] | hn[t][s]], each n wide
__global__ void k_mtp_concat(const float* en, const float* hn, float* cat, int n, int hc) {
    const int row = blockIdx.y, t = row / hc;   // row = t * hc + s
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 * n; i += gridDim.x * blockDim.x)
        cat[size_t(row) * 2 * n + i] = i < n ? en[size_t(t) * n + i] : hn[size_t(row) * n + i - n];
}

// One block per token, E threads (E <= 1024): softmax over the router logits, top-k by
// probability (ties to the lower index), weights renormalised (sum clamped at 6.1e-5), as
// moe_block's routing.
__global__ void k_mtp_route(const float* logits, int E, int K, int32_t* ids, float* wts) {
    __shared__ float p[1024];
    __shared__ float red[32];
    __shared__ float selp[32];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5, nw = blockDim.x >> 5;
    const float lg = e < E ? logits[size_t(t) * E + e] : -INFINITY;
    float m = lg;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < nw; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    const float ex = e < E ? __expf(lg - m) : 0.0f;
    float sum = ex;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int w = 0; w < nw; ++w) sum += red[w];
    const float pe = e < E ? ex / sum : -1.0f;
    p[e] = pe;
    __syncthreads();
    if (e < E) {
        int rank = 0;
        for (int j = 0; j < E; ++j) {
            const float pj = p[j];
            rank += (pj > pe) | ((pj == pe) & (j < e));
        }
        if (rank < K) {
            ids[size_t(t) * K + rank] = e;
            selp[rank] = pe;
        }
    }
    __syncthreads();
    if (e == 0) {
        float ws = 0.0f;
        for (int k = 0; k < K; ++k) ws += selp[k];
        ws = fmaxf(ws, 6.103515625e-5f);
        for (int k = 0; k < K; ++k) wts[size_t(t) * K + k] = selp[k] / ws;
    }
}

__global__ void k_swiglu_rows(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

// out[t][i] = sum_k wts[t][k] * yd[t][k][i] + sh[t][i] * sigmoid(gate[t])
__global__ void k_mtp_moe_combine(float* out, const float* yd, const float* wts, const float* sh, const float* gate, int n, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += wts[t * K + k] * yd[(size_t(t) * K + k) * n + i];
    out[size_t(t) * n + i] = acc + sh[size_t(t) * n + i] / (1.0f + __expf(-gate[t]));
}

}  // namespace

MtpHead::MtpHead(const Gguf& g, const Spec& target, const GpuWeights& target_w, cudaStream_t stream, int max_ctx, int max_batch,
                 bool kv_q8, int kv_hot_blocks)
    : s_(target), ts_(target), tw_(target_w), max_batch_(max_batch), stream_(stream) {
    if (g.get_string("general.architecture") != "qwen4exp") throw std::runtime_error("MTP: the draft GGUF is not qwen4exp");
    // the draft block is the one with nextn tensors; the borrowed embedding and head are skipped
    static const std::regex blk(R"(^blk\.(\d+)\.(.+)$)");
    il_ = -1;
    WeightPlan plan;
    for (const GgufTensor& t : g.tensors) {
        std::smatch m;
        if (!std::regex_match(t.name, m, blk)) continue;
        const std::string rest = m[2];
        if (rest == "nextn.embed_tokens.weight" || rest == "nextn.shared_head_head.weight") continue;
        const int l = std::stoi(m[1]);
        if (rest == "nextn.eh_proj.weight") {
            if (il_ >= 0 && il_ != l) throw std::runtime_error("MTP: more than one draft block");
            il_ = l;
        }
        plan.tensors.push_back({&t, Tier::VramDense, l});
    }
    if (il_ < 0) throw std::runtime_error("MTP: no nextn.eh_proj in the draft GGUF");
    if (il_ != target.n_layer) throw std::runtime_error("MTP: the draft block does not follow the target's layers");
    w_.load(g, plan, false);
    s_.n_layer = il_ + 1;
    s_.mixer.push_back(Mixer::QSA);
    s_.qsa_layers.push_back(il_);
    const GpuTensor& gexp = w_.layer(il_, "ffn_gate_exps.weight");
    if (gexp.dims != std::vector<int64_t>{s_.d_model, s_.d_ff_expert, s_.n_expert} || s_.n_expert > 1024 || s_.top_k > 32)
        throw std::runtime_error("MTP: unexpected expert shape");

    scratch_ = alloc_block_scratch(s_, max_batch);
    kv_ = alloc_qsa_cache(s_, max_ctx, kv_q8, kv_hot_blocks);
    const size_t n = s_.d_model, hc = s_.hc_count, B = max_batch, K = s_.top_k, ff = s_.d_ff_expert, cb = std::min(max_batch, 8);
    x_ = dalloc<float>(B * hc * n);
    emb_ = dalloc<float>(B * n);
    en_ = dalloc<float>(B * n);
    hn_ = dalloc<float>(B * hc * n);
    cat_ = dalloc<float>(B * hc * 2 * n);
    mixed_ = dalloc<float>(B * n);
    inject_ = dalloc<float>(B * hc);
    blk_ = dalloc<float>(B * n);
    norm_ = dalloc<float>(B * n);
    ids_ = dalloc<int32_t>(B * K);
    wts_ = dalloc<float>(B * K);
    logits_e_ = dalloc<float>(B * s_.n_expert);
    xq_ = dalloc<uint8_t>(gemv::q8_1_bytes(n, int(cb)));
    hq_ = dalloc<uint8_t>(gemv::q8_1_bytes(std::max<size_t>(ff, s_.d_ff_shared), int(cb * K)));
    hid_ = dalloc<float>(cb * K * ff);
    yd_ = dalloc<float>(cb * K * n);
    sg_ = dalloc<float>(B * s_.d_ff_shared);
    su_ = dalloc<float>(B * s_.d_ff_shared);
    sh_ = dalloc<float>(B * n);
    gate_ = dalloc<float>(B);
}

MtpHead::~MtpHead() {
    free_block_scratch(scratch_);
    free_qsa_cache(kv_);
    for (void* p : {static_cast<void*>(x_), static_cast<void*>(emb_), static_cast<void*>(en_), static_cast<void*>(hn_),
                    static_cast<void*>(cat_), static_cast<void*>(mixed_), static_cast<void*>(inject_), static_cast<void*>(blk_),
                    static_cast<void*>(norm_), static_cast<void*>(ids_), static_cast<void*>(wts_), static_cast<void*>(logits_e_), xq_,
                    hq_, static_cast<void*>(hid_), static_cast<void*>(yd_), static_cast<void*>(sg_), static_cast<void*>(su_),
                    static_cast<void*>(sh_), static_cast<void*>(gate_)})
        if (p) cudaFree(p);
}

void MtpHead::reset() {
    ck(cudaMemsetAsync(kv_.idx_ring, 0, size_t(qsa_ring_slots(s_)) * s_.idx_dim * 4, stream_), "memset MTP ring");
    reset_qsa_hot(s_, kv_, stream_);
    ck(cudaStreamSynchronize(stream_), "MTP reset");
}

// The draft block's MoE: every expert in VRAM (the GGUF's type), routed on the GPU, in chunks
// of up to 8 tokens.
void MtpHead::moe(const BlockCtx& c, const float* x, int T, float* out) {
    const int n = s_.d_model, E = s_.n_expert, K = s_.top_k, ff = s_.d_ff_expert, ffs = s_.d_ff_shared;
    const GpuTensor& wg = w_.layer(il_, "ffn_gate_exps.weight");
    const GpuTensor& wu = w_.layer(il_, "ffn_up_exps.weight");
    const GpuTensor& wd = w_.layer(il_, "ffn_down_exps.weight");
    const int64_t gu_stride = gemv::row_bytes(wu.type, n) * ff, d_stride = gemv::row_bytes(wd.type, ff) * n;
    linear(c, w_.layer(il_, "ffn_gate_inp.weight"), x, logits_e_, T);
    k_mtp_route<<<T, ((E + 31) / 32) * 32, 0, c.stream>>>(logits_e_, E, K, ids_, wts_);
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int nt = std::min(8, T - t0);
        gemv::quantize_q8_1(x + size_t(t0) * n, n, nt, xq_, c.stream);
        gemv::moe_q(wu.type, wu.dev, wg.dev, xq_, ids_ + size_t(t0) * K, hid_, nt, K, n, ff, gu_stride, false, c.stream);
        gemv::quantize_q8_1(hid_, ff, nt * K, hq_, c.stream);
        gemv::moe_q(wd.type, wd.dev, nullptr, hq_, ids_ + size_t(t0) * K, yd_ + size_t(0), nt, K, ff, n, d_stride, true, c.stream);
        // shared expert
        linear(c, w_.layer(il_, "ffn_gate_shexp.weight"), x + size_t(t0) * n, sg_, nt);
        linear(c, w_.layer(il_, "ffn_up_shexp.weight"), x + size_t(t0) * n, su_, nt);
        k_swiglu_rows<<<(nt * ffs + 255) / 256, 256, 0, c.stream>>>(sg_, su_, nt * ffs);
        linear(c, w_.layer(il_, "ffn_down_shexp.weight"), sg_, sh_, nt);
        linear(c, w_.layer(il_, "ffn_gate_inp_shexp.weight"), x + size_t(t0) * n, gate_, nt);
        k_mtp_moe_combine<<<dim3((n + 255) / 256, nt), 256, 0, c.stream>>>(out + size_t(t0) * n, yd_, wts_ + size_t(t0) * K, sh_, gate_,
                                                                          n, K);
    }
    ck(cudaGetLastError(), "MTP moe");
}

void MtpHead::forward(const float* h_prev, const int32_t* tokens, int T, int pos0, int out_from, float* logits_dev) {
    if (T < 1 || T > max_batch_) throw std::runtime_error("MtpHead: bad batch size");
    const int n = s_.d_model, hc = s_.hc_count, il = il_;
    const BlockCtx c{s_, w_, scratch_, stream_};
    const BlockCtx ct{ts_, tw_, scratch_, stream_};   // the target's embedding and head
    // input: per stream, eh_proj([enorm(embed(x)) | hnorm(h_prev)]); h_prev is read before x_ is written
    rms_norm_rows(c, h_prev, static_cast<const float*>(w_.layer(il, "nextn.hnorm.weight").dev), hn_, n, hc, T * hc);
    embed(ct, tokens, T, emb_);
    rms_norm_rows(c, emb_, static_cast<const float*>(w_.layer(il, "nextn.enorm.weight").dev), en_, n, 1, T);
    k_mtp_concat<<<dim3((2 * n + 255) / 256, T * hc), 256, 0, stream_>>>(en_, hn_, cat_, n, hc);
    linear(c, w_.layer(il, "nextn.eh_proj.weight"), cat_, x_, T * hc);
    // the block
    hc_mix(c, il, 0, x_, T, mixed_, inject_);
    qsa_mixer(c, il, mixed_, T, pos0, kv_, blk_);
    hc_combine(c, x_, blk_, inject_, T);
    hc_mix(c, il, 1, x_, T, mixed_, inject_);
    moe(c, mixed_, T, blk_);
    hc_combine(c, x_, blk_, inject_, T);
    // head
    if (out_from < T && logits_dev) {
        const int R = T - out_from;
        hc_mix(c, il, 3, x_ + size_t(out_from) * hc * n, R, norm_, nullptr);
        head_logits(ct, norm_, R, logits_dev);
    }
    ck(cudaGetLastError(), "MtpHead::forward");
}

}  // namespace flashrt::qwen4exp
