// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/mtp.hpp"
#include "arch/qwen4exp/graph_capture.hpp"

#include "core/fp16.hpp"
#include "core/platform.hpp"
#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

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

// Q8_0 blocks -> Q4_0 blocks (ggml's quantize_row_q4_0_ref on the dequantized values)
void q8_0_to_q4_0(const uint8_t* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b, src += 34, dst += 18) {
        const float d8 = fp16_to_fp32(uint16_t(src[0] | (src[1] << 8)));
        float x[32], amax = 0.0f, mx = 0.0f;
        for (int j = 0; j < 32; ++j) {
            x[j] = d8 * float(int8_t(src[2 + j]));
            if (std::fabs(x[j]) > amax) {
                amax = std::fabs(x[j]);
                mx = x[j];
            }
        }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = fp32_to_fp16(d);
        dst[0] = uint8_t(dh & 0xff);
        dst[1] = uint8_t(dh >> 8);
        for (int j = 0; j < 16; ++j) {
            const int q0 = std::min(15, int(int8_t(x[j] * id + 8.5f)));
            const int q1 = std::min(15, int(int8_t(x[j + 16] * id + 8.5f)));
            dst[2 + j] = uint8_t(q0 | (q1 << 4));
        }
    }
}

// Q8_0 blocks (32) -> ggml Q2_0 blocks (64: an fp16 scale d and 2-bit codes q, value (q - 1) * d,
// element j in byte j / 4 at bit 2 * (j % 4)). Round to nearest with the scale that minimises
// the block's squared error over a small set of candidates.
// The code of a value v = x / d is clamp(lround(v), -1, 2). For finite v that is three
// comparisons, which vectorise; lround was a library call per value (the head's experts took 32 s
// on 8 threads, 3.0 s now on 24; sw98). A zero or infinite d (fp16 underflow or overflow) keeps
// lround, as before. The
// error sum keeps its expressions and order, so the chosen scales and codes are bit-identical
// (test_mtp_convert).
inline int q2_code(float v) { return int(v >= 0.5f) + int(v >= 1.5f) - int(v <= -0.5f); }
void q2_codes(const float* x, float d, int* q) {
    if (d > 0.0f && std::isfinite(d)) {
        for (int j = 0; j < 64; ++j) q[j] = q2_code(x[j] / d);
    } else {
        for (int j = 0; j < 64; ++j) q[j] = std::min(2, std::max(-1, int(std::lround(x[j] / d))));
    }
}
void q8_0_to_q2_0(const uint8_t* src, uint8_t* dst, size_t nblocks64) {
    for (size_t b = 0; b < nblocks64; ++b, src += 68, dst += 18) {
        float x[64], amax = 0.0f;
        for (int h = 0; h < 2; ++h) {
            const uint8_t* s8 = src + 34 * h;
            const float d8 = fp16_to_fp32(uint16_t(s8[0] | (s8[1] << 8)));
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * h + j] = d8 * float(int8_t(s8[2 + j]))));
        }
        float best_d = 0.0f, best_e = INFINITY;
        int q[64];
        for (int c = 0; c < 24 && amax > 0.0f; ++c) {   // d from amax / 2.5 up to amax
            const float d = amax / (2.5f - 1.5f * float(c) / 23.0f);
            const uint16_t dh = fp32_to_fp16(d);
            const float dq = fp16_to_fp32(dh);
            q2_codes(x, dq, q);
            float e = 0.0f;
            for (int j = 0; j < 64; ++j) {
                const float r = x[j] - float(q[j]) * dq;
                e += r * r;
            }
            if (e < best_e) { best_e = e; best_d = dq; }
        }
        const uint16_t dh = fp32_to_fp16(best_d);
        dst[0] = uint8_t(dh & 0xff);
        dst[1] = uint8_t(dh >> 8);
        std::memset(dst + 2, 0, 16);
        if (best_d > 0.0f) q2_codes(x, best_d, q);
        else std::fill(q, q + 64, 0);
        for (int j = 0; j < 64; ++j) dst[2 + j / 4] |= uint8_t((q[j] + 1) << (2 * (j % 4)));
    }
}

// v = [argmax index, token, p]: token = ids[index] (or the index itself), and with want_p, p =
// 1 / sum exp(x - x[index]), the top token's softmax probability. One block of 1024 threads.
__global__ void k_draft_top(const float* x, int n, const int32_t* ids, int32_t* v, bool want_p) {
    const int idx = v[0];
    if (threadIdx.x == 0) v[1] = ids ? ids[idx] : idx;
    if (!want_p) return;
    const float mx = x[idx];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - mx);
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) t += red[w];
        reinterpret_cast<float*>(v)[2] = 1.0f / t;
    }
}

// rows[i] = src row ids[i], row_bytes each; one block per row
__global__ void k_gather_rows(const uint8_t* src, const int32_t* ids, uint8_t* dst, size_t row_bytes) {
    const uint8_t* s = src + size_t(ids[blockIdx.x]) * row_bytes;
    uint8_t* d = dst + size_t(blockIdx.x) * row_bytes;
    for (size_t i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
}

}  // namespace

void convert_q8_0_to_q2_0(const uint8_t* src, uint8_t* dst, size_t nblocks64, int threads) {
    std::vector<std::thread> th;
    for (int k = 0; k < threads; ++k)
        th.emplace_back([=] {
            // not on the creator's CPU: the engine pins its thread to one CPU before the head loads,
            // and threads inherit that, so the conversion ran on one core (the ~250 s load, sw98)
            unpin_current_thread();
            const size_t a = nblocks64 * size_t(k) / size_t(threads), e = nblocks64 * size_t(k + 1) / size_t(threads);
            q8_0_to_q2_0(src + a * 68, dst + a * 18, e - a);
        });
    for (auto& t : th) t.join();
}

MtpHead::MtpHead(const Gguf& g, const Spec& target, const GpuWeights& target_w, cudaStream_t stream, int max_ctx, int max_batch,
                 bool kv_q8, int kv_hot_blocks, int expert_bits)
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
        if (expert_bits != 8 && (rest == "ffn_gate_exps.weight" || rest == "ffn_up_exps.weight" || rest == "ffn_down_exps.weight")) continue;
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
    if (expert_bits == 8) {
        exps_[0] = w_.layer(il_, "ffn_gate_exps.weight");
        exps_[1] = w_.layer(il_, "ffn_up_exps.weight");
        exps_[2] = w_.layer(il_, "ffn_down_exps.weight");
    } else if (expert_bits == 4 || expert_bits == 2) {
        load_experts_q4(g, expert_bits == 2);
    } else {
        throw std::runtime_error("MTP: expert bits must be 8, 4 or 2");
    }
    if (exps_[0].dims != std::vector<int64_t>{s_.d_model, s_.d_ff_expert, s_.n_expert} || s_.n_expert > 1024 || s_.top_k > 32)
        throw std::runtime_error("MTP: unexpected expert shape");
    ck(cudaMalloc(&amax_dev_, 16), "cudaMalloc MTP argmax");
    ck(cudaHostAlloc(&amax_host_, 64, cudaHostAllocDefault), "cudaHostAlloc MTP argmax");
    ck(cudaMalloc(&chain_dp_, 16 * 4), "cudaMalloc MTP chain");
    ck(cudaMalloc(&chain_drafts_, 16 * 4), "cudaMalloc MTP chain");
    ck(cudaMalloc(&dcfg_dev_, sizeof(sample::DraftCfg)), "cudaMalloc MTP draft sampling");
    ck(cudaMalloc(&q_ids_, 8 * sample::kMaxTopK * 4), "cudaMalloc MTP q");
    ck(cudaMalloc(&q_p_, 8 * sample::kMaxTopK * 4), "cudaMalloc MTP q");
    ck(cudaMalloc(&q_n_, 8 * 4), "cudaMalloc MTP q");
    ck(cudaMalloc(&h_in_, size_t(s_.hc_count) * s_.d_model * 4), "cudaMalloc MTP chain");
    ck(cudaMalloc(&chain_logits_, size_t(ts_.n_vocab) * 4), "cudaMalloc MTP chain");

    kv_ = alloc_qsa_cache(s_, max_ctx, kv_q8, kv_hot_blocks);
    alloc_bufs(dec_, max_batch, false);
    use_bufs(dec_);
}

// The buffers of calls of up to T rows. The decode set has the 8-row MoE slices' buffers; the
// chunk set has the grouped MoE's instead (gate and up rows, outputs for every (row, slot), the
// gemm workspace) and the h_prev rows of a call.
void MtpHead::alloc_bufs(Bufs& b, int T, bool chunk) {
    const size_t n = s_.d_model, hc = s_.hc_count, B = size_t(T), K = s_.top_k, ff = s_.d_ff_expert, ffs = s_.d_ff_shared;
    b.scratch = alloc_block_scratch(s_, T);
    b.x = dalloc<float>(B * hc * n);
    b.emb = dalloc<float>(B * n);
    b.en = dalloc<float>(B * n);
    b.hn = dalloc<float>(B * hc * n);
    b.cat = dalloc<float>(B * hc * 2 * n);
    b.mixed = dalloc<float>(B * n);
    b.inject = dalloc<float>(B * hc);
    b.blk = dalloc<float>(B * n);
    b.norm = dalloc<float>(B * n);
    b.ids = dalloc<int32_t>(B * K);
    b.wts = dalloc<float>(B * K);
    b.logits_e = dalloc<float>(B * s_.n_expert);
    b.sg = dalloc<float>(B * ffs);
    b.su = dalloc<float>(B * ffs);
    b.sh = dalloc<float>(B * n);
    b.gate = dalloc<float>(B);
    if (chunk) {
        b.hg = dalloc<float>(B * K * ff);
        b.hu = dalloc<float>(B * K * ff);
        b.yd = dalloc<float>(B * K * n);
        b.hin = dalloc<float>(B * hc * n);
        b.ws_bytes = gemm::workspace_bytes(int64_t(std::max(n, ff)), int64_t(B * K), false);   // Q8_1 activations only
        b.ws = dalloc<uint8_t>(b.ws_bytes);
    } else {
        const size_t cb = std::min<size_t>(B, 8);
        b.xq = dalloc<uint8_t>(gemv::q8_1_bytes(n, int(cb)));
        b.hq = dalloc<uint8_t>(gemv::q8_1_bytes(std::max(ff, ffs), int(cb * K)));
        b.hid = dalloc<float>(cb * K * ff);
        b.yd = dalloc<float>(cb * K * n);
    }
    b.cap = T;
}

void MtpHead::free_bufs(Bufs& b) {
    if (!b.cap) return;
    free_block_scratch(b.scratch);
    for (void* p : {static_cast<void*>(b.x), static_cast<void*>(b.emb), static_cast<void*>(b.en), static_cast<void*>(b.hn),
                    static_cast<void*>(b.cat), static_cast<void*>(b.mixed), static_cast<void*>(b.inject), static_cast<void*>(b.blk),
                    static_cast<void*>(b.norm), static_cast<void*>(b.ids), static_cast<void*>(b.wts), static_cast<void*>(b.logits_e),
                    b.xq, b.hq, static_cast<void*>(b.hid), static_cast<void*>(b.yd), static_cast<void*>(b.sg), static_cast<void*>(b.su),
                    static_cast<void*>(b.sh), static_cast<void*>(b.gate), static_cast<void*>(b.hg), static_cast<void*>(b.hu),
                    static_cast<void*>(b.hin), b.ws})
        if (p) cudaFree(p);
    b = Bufs{};
}

void MtpHead::use_bufs(Bufs& b) {
    x_ = b.x, emb_ = b.emb, en_ = b.en, hn_ = b.hn, cat_ = b.cat, mixed_ = b.mixed, inject_ = b.inject, blk_ = b.blk, norm_ = b.norm;
    ids_ = b.ids, wts_ = b.wts, logits_e_ = b.logits_e, xq_ = b.xq, hq_ = b.hq, hid_ = b.hid, yd_ = b.yd;
    sg_ = b.sg, su_ = b.su, sh_ = b.sh, gate_ = b.gate, hg_ = b.hg, hu_ = b.hu, mws_ = b.ws, mws_bytes_ = b.ws_bytes;
    scr_ = &b.scratch;
}

void MtpHead::load_experts_q4(const Gguf& g, bool q2) {
    using ggml_type::kQ2_0, ggml_type::kQ4_0, ggml_type::kQ8_0;
    const size_t in_blk = q2 ? 68 : 34, out_blk = 18, per_blk = q2 ? 64 : 32;   // bytes per output block, elements
    const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    const GgufTensor* src[3];
    size_t off[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        src[i] = g.tensor("blk." + std::to_string(il_) + "." + names[i]);
        if (!src[i] || src[i]->type != kQ8_0) throw std::runtime_error(std::string("MTP: expert tensor missing or not Q8_0: ") + names[i]);
        const size_t qb = size_t(src[i]->n_elements()) / per_blk * out_blk;
        off[i + 1] = off[i] + ((qb + gemv::kWeightTailPad + 255) & ~size_t(255));
    }
    exp_bytes_ = off[3];
    ck(cudaMalloc(&exp_dev_, exp_bytes_), "cudaMalloc MTP experts");
    ck(cudaMemset(exp_dev_, 0, exp_bytes_), "memset MTP experts");
    // per tensor: read in chunks of whole blocks, convert on a few threads, upload
    constexpr size_t kChunkBlocks = size_t(1) << 20;   // output blocks per chunk
    std::vector<uint8_t> in(kChunkBlocks * in_blk), out(kChunkBlocks * out_blk);
    for (int i = 0; i < 3; ++i) {
        const GgufTensor& t = *src[i];
        const int fd = open(g.shards[t.shard].c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open " + g.shards[t.shard]);
        const size_t nb = size_t(t.n_elements()) / per_blk;
        for (size_t b0 = 0; b0 < nb; b0 += kChunkBlocks) {
            const size_t n = std::min(kChunkBlocks, nb - b0);
            for (size_t r = 0; r < n * in_blk;) {
                const ssize_t got = pread(fd, in.data() + r, n * in_blk - r, off_t(t.file_offset + b0 * in_blk + r));
                if (got <= 0) throw std::runtime_error("short read of " + t.name);
                r += size_t(got);
            }
            const int nt = int(std::max(8u, std::thread::hardware_concurrency()));
            if (q2) convert_q8_0_to_q2_0(in.data(), out.data(), n, nt);
            else {
                std::vector<std::thread> th;
                for (int k = 0; k < nt; ++k)
                    th.emplace_back([&, k] {
                        unpin_current_thread();
                        const size_t a = n * k / nt, e = n * (k + 1) / nt;
                        q8_0_to_q4_0(in.data() + a * in_blk, out.data() + a * out_blk, e - a);
                    });
                for (auto& x : th) x.join();
            }
            ck(cudaMemcpy(static_cast<uint8_t*>(exp_dev_) + off[i] + b0 * out_blk, out.data(), n * out_blk, cudaMemcpyHostToDevice),
               "upload MTP experts");
        }
        close(fd);
        exps_[i].dev = static_cast<uint8_t*>(exp_dev_) + off[i];
        exps_[i].type = q2 ? kQ2_0 : kQ4_0;
        exps_[i].dims = t.dims;
        exps_[i].bytes = nb * out_blk;
    }
}

void MtpHead::reserve_vocab(int n) {
    if (n <= head_cap_) return;
    const GpuTensor& full = tw_.get("output.weight");
    const size_t rb = size_t(gemv::row_bytes(full.type, full.cols()));
    if (head_.dev) cudaFree(head_.dev);
    head_bytes_ = rb * size_t(n) + gemv::kWeightTailPad + size_t(n) * 4;
    ck(cudaMalloc(&head_.dev, head_bytes_), "cudaMalloc MTP head");
    head_cap_ = n;
    vocab_ids_.clear();
}

void MtpHead::set_draft_sampling(const sample::Params& p, uint64_t seed) {
    sampled_ = p.temperature > 0.0f;
    dcfg_host_ = sample::DraftCfg{p, seed};
    ck(cudaMemcpyAsync(dcfg_dev_, &dcfg_host_, sizeof(dcfg_host_), cudaMemcpyHostToDevice, stream_), "MTP draft sampling");
}

void MtpHead::set_vocab(const std::vector<int32_t>& ids) {
    if (chain_graph_) {
        cudaGraphExecDestroy(chain_graph_);
        chain_graph_ = nullptr;
    }
    if (chain_graph_s_) {
        cudaGraphExecDestroy(chain_graph_s_);
        chain_graph_s_ = nullptr;
    }
    vocab_ids_ = ids;
    if (ids.empty()) return;
    reserve_vocab(int(ids.size()));
    const GpuTensor& full = tw_.get("output.weight");
    const size_t rb = size_t(gemv::row_bytes(full.type, full.cols()));
    uint8_t* dev = static_cast<uint8_t*>(head_.dev);
    // rows, the tail pad (zeros) after the last used row, then the ids
    ck(cudaMemsetAsync(dev + rb * ids.size(), 0, gemv::kWeightTailPad, stream_), "memset MTP head pad");
    int32_t* ids_dev = reinterpret_cast<int32_t*>(dev + rb * ids.size() + gemv::kWeightTailPad);
    ck(cudaMemcpy(ids_dev, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "MTP head ids");
    k_gather_rows<<<unsigned(ids.size()), 256, 0, stream_>>>(static_cast<const uint8_t*>(full.dev), ids_dev, dev, rb);
    ck(cudaStreamSynchronize(stream_), "MTP head gather");
    head_.type = full.type;
    head_.dims = {full.cols(), int64_t(ids.size())};
    head_.bytes = rb * ids.size();
}

size_t MtpHead::checkpoint_bytes() const {
    return (size_t(qsa_ring_slots(s_)) * s_.idx_dim + size_t(s_.hc_count) * s_.d_model) * 4;
}

void MtpHead::reserve_checkpoint() {
    if (!ckpt_) ck(cudaMalloc(&ckpt_, checkpoint_bytes()), "cudaMalloc MTP checkpoint");
}

void MtpHead::save_checkpoint_to(void* dst, const float* h_dev) {
    const size_t ring = size_t(qsa_ring_slots(s_)) * s_.idx_dim, h = size_t(s_.hc_count) * s_.d_model;
    float* d = static_cast<float*>(dst);
    ck(cudaMemcpyAsync(d, kv_.idx_ring, ring * 4, cudaMemcpyDefault, stream_), "MTP checkpoint");
    ck(cudaMemcpyAsync(d + ring, h_dev, h * 4, cudaMemcpyDefault, stream_), "MTP checkpoint");
    ck(cudaStreamSynchronize(stream_), "MTP checkpoint");
}

void MtpHead::save_checkpoint(const float* h_dev) {
    reserve_checkpoint();
    save_checkpoint_to(ckpt_, h_dev);
}

void MtpHead::restore_checkpoint_from(const void* src, float* h_dev) {
    const size_t ring = size_t(qsa_ring_slots(s_)) * s_.idx_dim, h = size_t(s_.hc_count) * s_.d_model;
    const float* c = static_cast<const float*>(src);
    ck(cudaMemcpyAsync(kv_.idx_ring, c, ring * 4, cudaMemcpyDefault, stream_), "MTP restore");
    ck(cudaMemcpyAsync(h_dev, c + ring, h * 4, cudaMemcpyDefault, stream_), "MTP restore");
    reset_qsa_hot(s_, kv_, stream_);
    ck(cudaStreamSynchronize(stream_), "MTP restore");
}

void MtpHead::restore_checkpoint(float* h_dev) {
    if (!ckpt_) throw std::runtime_error("MTP restore_checkpoint: none saved");
    restore_checkpoint_from(ckpt_, h_dev);
}

void MtpHead::save_state(const std::string& path, int pos, const float* h_carry_dev) {
    ck(cudaStreamSynchronize(stream_), "MTP state");
    std::unique_ptr<FILE, int (*)(FILE*)> fh(std::fopen(path.c_str(), "wb"), std::fclose);
    FILE* f = fh.get();
    if (!f) throw std::runtime_error("cannot open " + path);
    const int64_t hdr[3] = {0x544d5246 /* "FRMT" */, pos, kv_.q8 ? 1 : 0};
    if (std::fwrite(hdr, sizeof(hdr), 1, f) != 1) throw std::runtime_error("MTP state write failed: " + path);
    qsa_state_io(f, s_, kv_, pos, true, kv_.q8, stream_);
    std::vector<float> h(size_t(s_.hc_count) * s_.d_model);
    ck(cudaMemcpy(h.data(), h_carry_dev, h.size() * 4, cudaMemcpyDeviceToHost), "MTP h to host");
    if (std::fwrite(h.data(), 4, h.size(), f) != h.size() || std::fclose(fh.release()) != 0)
        throw std::runtime_error("MTP state write failed: " + path);
}

int MtpHead::load_state(const std::string& path, float* h_carry_dev) {
    std::unique_ptr<FILE, int (*)(FILE*)> fh(std::fopen(path.c_str(), "rb"), std::fclose);
    FILE* f = fh.get();
    if (!f) throw std::runtime_error("cannot open " + path);
    int64_t hdr[3] = {0, 0, 0};
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != 0x544d5246 || hdr[1] < 0 || hdr[1] > kv_.capacity)
        throw std::runtime_error("not an MTP state file, or longer than the KV capacity: " + path);
    const int pos = int(hdr[1]);
    qsa_state_io(f, s_, kv_, pos, false, hdr[2] == 1, stream_);
    std::vector<float> h(size_t(s_.hc_count) * s_.d_model);
    if (std::fread(h.data(), 4, h.size(), f) != h.size()) throw std::runtime_error("MTP state file truncated: " + path);
    ck(cudaMemcpy(h_carry_dev, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "MTP h to device");
    return pos;
}

namespace {
// the chain's bookkeeping after a step: the drafted token becomes the next step's input
// (dp[0]), the position advances, and the token is recorded, with (argmax drafts) its
// probability under the head at drafts[8 + step] as a float
__global__ void k_chain_next(const int32_t* v, int32_t* dp, int32_t* drafts) {
    const int j = dp[3];
    drafts[j] = v[1];
    drafts[8 + j] = v[2];
    dp[0] = v[1];
    dp[1] += 1;
    dp[3] = j + 1;
}
}  // namespace

std::vector<int32_t> MtpHead::draft_chain(int row, int pos, int k) {
    if (k < 1 || k > 8) throw std::runtime_error("draft_chain: 1..8 drafts");
    const int n = s_.d_model, hc = s_.hc_count;
    const int32_t* ids_dev =
        vocab_ids_.empty() ? nullptr
                           : reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(head_.dev) + head_.bytes + gemv::kWeightTailPad);
    // the chain's parameters: [token, position, -, step]; the input streams are row `row` of x_
    chain_init_[0] = 0;
    chain_init_[1] = pos - 1;   // k_chain_next advances it to pos
    chain_init_[2] = chain_init_[3] = 0;
    ck(cudaMemcpyAsync(chain_dp_, chain_init_, sizeof(chain_init_), cudaMemcpyHostToDevice, stream_), "chain params");
    // the first draft: from the logits of the last forward() row (chain_logits_ holds it)
    auto draft_step = [&] {   // the draft for position dp[1] + 1 into amax_dev_[1]
        if (sampled_) sample::draft_row(chain_logits_, vocab(), dcfg_dev_, chain_dp_, ids_dev, amax_dev_, q_ids_, q_p_, q_n_, stream_);
        else {
            argmax_dev(stream_, chain_logits_, vocab(), amax_dev_);
            k_draft_top<<<1, 1024, 0, stream_>>>(chain_logits_, vocab(), ids_dev, amax_dev_, true);
        }
    };
    draft_step();
    k_chain_next<<<1, 1, 0, stream_>>>(amax_dev_, chain_dp_, chain_drafts_);
    ck(cudaMemcpyAsync(h_in_, x_ + size_t(row) * hc * n, size_t(hc) * n * 4, cudaMemcpyDeviceToDevice, stream_), "chain h");
    if (k > 1) {
        cudaGraphExec_t& graph = sampled_ ? chain_graph_s_ : chain_graph_;   // one per draft kind
        if ((chain_graph_ || chain_graph_s_) && chain_scratch_version_ != dec_.scratch.version) {   // an eager call grew the scratch
            for (cudaGraphExec_t* gp : {&chain_graph_, &chain_graph_s_})
                if (*gp) {
                    cudaGraphExecDestroy(*gp);
                    *gp = nullptr;
                }
        }
        if (!graph) {   // one chained step: forward at (dp[0], dp[1]), its draft, the bookkeeping, h for the next
            qsa_scratch_reserve(s_, dec_.scratch, 1, kv_.capacity / s_.qsa_block);
            cudaGraph_t g = capture_graph(
                stream_,
                [&] {
                    enqueue(h_in_, nullptr, 1, 0, 0, chain_logits_, chain_dp_);
                    draft_step();
                    k_chain_next<<<1, 1, 0, stream_>>>(amax_dev_, chain_dp_, chain_drafts_);
                    ck(cudaMemcpyAsync(h_in_, x_, size_t(hc) * n * 4, cudaMemcpyDeviceToDevice, stream_), "chain h");
                },
                "MTP capture");
            ck(cudaGraphInstantiate(&graph, g, 0), "instantiate MTP graph");
            cudaGraphDestroy(g);
            chain_scratch_version_ = dec_.scratch.version;
        }
        for (int j = 1; j < k; ++j) ck(cudaGraphLaunch(graph, stream_), "launch MTP graph");
    }
    ck(cudaMemcpyAsync(amax_host_, chain_drafts_, size_t(k) * 4, cudaMemcpyDeviceToHost, stream_), "drafts to host");
    ck(cudaStreamSynchronize(stream_), "draft chain");
    return std::vector<int32_t>(amax_host_, amax_host_ + k);
}

std::vector<float> MtpHead::draft_probs(int k) {
    std::vector<float> pr(size_t(k), 0.0f);
    if (!sampled_) {
        ck(cudaMemcpyAsync(pr.data(), chain_drafts_ + 8, size_t(k) * 4, cudaMemcpyDeviceToHost, stream_), "draft probs");
        ck(cudaStreamSynchronize(stream_), "draft probs");
        return pr;
    }
    const int m = sample::kMaxTopK;
    std::vector<int32_t> ids(size_t(k) * m), n(k), d(k);
    std::vector<float> q(size_t(k) * m);
    ck(cudaMemcpyAsync(ids.data(), q_ids_, ids.size() * 4, cudaMemcpyDeviceToHost, stream_), "draft q");
    ck(cudaMemcpyAsync(q.data(), q_p_, q.size() * 4, cudaMemcpyDeviceToHost, stream_), "draft q");
    ck(cudaMemcpyAsync(n.data(), q_n_, n.size() * 4, cudaMemcpyDeviceToHost, stream_), "draft q");
    ck(cudaMemcpyAsync(d.data(), chain_drafts_, d.size() * 4, cudaMemcpyDeviceToHost, stream_), "drafts");
    ck(cudaStreamSynchronize(stream_), "draft q");
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n[size_t(j)]; ++i)
            if (ids[size_t(j) * m + i] == d[size_t(j)]) pr[size_t(j)] = q[size_t(j) * m + i];
    return pr;
}

int32_t MtpHead::argmax(const float* logits_row_dev, float* p_top) {
    argmax_dev(stream_, logits_row_dev, vocab(), amax_dev_);
    const int32_t* ids_dev =
        vocab_ids_.empty() ? nullptr
                           : reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(head_.dev) + head_.bytes + gemv::kWeightTailPad);
    k_draft_top<<<1, 1024, 0, stream_>>>(logits_row_dev, vocab(), ids_dev, amax_dev_, p_top != nullptr);
    ck(cudaMemcpyAsync(amax_host_, amax_dev_, 12, cudaMemcpyDeviceToHost, stream_), "MTP argmax to host");
    ck(cudaStreamSynchronize(stream_), "MTP argmax");
    if (p_top) std::memcpy(p_top, &amax_host_[2], 4);
    return amax_host_[1];
}

MtpHead::~MtpHead() {
    if (ckpt_) cudaFree(ckpt_);
    if (chain_graph_) cudaGraphExecDestroy(chain_graph_);
    if (chain_graph_s_) cudaGraphExecDestroy(chain_graph_s_);
    for (void* p : {static_cast<void*>(dcfg_dev_), static_cast<void*>(q_ids_), static_cast<void*>(q_p_), static_cast<void*>(q_n_)})
        if (p) cudaFree(p);
    for (void* p : {static_cast<void*>(chain_dp_), static_cast<void*>(chain_drafts_), static_cast<void*>(h_in_), static_cast<void*>(chain_logits_)})
        if (p) cudaFree(p);
    if (exp_dev_) cudaFree(exp_dev_);
    if (head_.dev) cudaFree(head_.dev);
    if (amax_dev_) cudaFree(amax_dev_);
    if (amax_host_) cudaFreeHost(amax_host_);
    free_qsa_cache(kv_);
    free_bufs(chunk_);
    free_bufs(dec_);
}

void MtpHead::prefill_begin(int pos, int end_pos) {
    qsa_mirror_begin(s_, kv_, pos, end_pos, stream_);
    if (!chunk_.cap) alloc_bufs(chunk_, kChunkRows, true);
}

void MtpHead::prefill_end() {
    ck(cudaStreamSynchronize(stream_), "MTP prefill end");
    qsa_mirror_end(kv_);
    use_bufs(dec_);
    free_bufs(chunk_);
}

void MtpHead::reset() {
    ck(cudaMemsetAsync(kv_.idx_ring, 0, size_t(qsa_ring_slots(s_)) * s_.idx_dim * 4, stream_), "memset MTP ring");
    reset_qsa_hot(s_, kv_, stream_);
    ck(cudaStreamSynchronize(stream_), "MTP reset");
}

// The draft block's MoE: every expert in VRAM (the GGUF's type), routed on the GPU. Decode and
// small batches: mat-vec kernels in slices of up to 8 tokens. Chunk calls (the chunk set, during
// a chunked prefill): grouped expert GEMMs over the whole call.
void MtpHead::moe(const BlockCtx& c, const float* x, int T, float* out) {
    const int n = s_.d_model, E = s_.n_expert, K = s_.top_k, ff = s_.d_ff_expert, ffs = s_.d_ff_shared;
    const GpuTensor &wg = exps_[0], &wu = exps_[1], &wd = exps_[2];
    const int64_t gu_stride = gemv::row_bytes(wu.type, n) * ff, d_stride = gemv::row_bytes(wd.type, ff) * n;
    linear(c, w_.layer(il_, "ffn_gate_inp.weight"), x, logits_e_, T);
    k_mtp_route<<<T, ((E + 31) / 32) * 32, 0, c.stream>>>(logits_e_, E, K, ids_, wts_);
    if (hg_) {   // a chunk: the experts as grouped GEMMs over every row of the call (sw102)
        const gemm::MoePlan pgu = gemm::moe_prepare(wu.type, E, x, false, ids_, T, K, n, mws_, mws_bytes_, c.stream);
        gemm::moe_run(pgu, wg.dev, gu_stride, hg_, ff, c.stream);
        gemm::moe_run(pgu, wu.dev, gu_stride, hu_, ff, c.stream);
        const size_t nh = size_t(T) * K * ff;
        k_swiglu_rows<<<unsigned((nh + 255) / 256), 256, 0, c.stream>>>(hg_, hu_, int(nh));
        const gemm::MoePlan pd = gemm::moe_prepare(wd.type, E, hg_, true, ids_, T, K, ff, mws_, mws_bytes_, c.stream);
        gemm::moe_run(pd, wd.dev, d_stride, yd_, n, c.stream);
        linear(c, w_.layer(il_, "ffn_gate_shexp.weight"), x, sg_, T);
        linear(c, w_.layer(il_, "ffn_up_shexp.weight"), x, su_, T);
        k_swiglu_rows<<<unsigned((size_t(T) * ffs + 255) / 256), 256, 0, c.stream>>>(sg_, su_, T * ffs);
        linear(c, w_.layer(il_, "ffn_down_shexp.weight"), sg_, sh_, T);
        linear(c, w_.layer(il_, "ffn_gate_inp_shexp.weight"), x, gate_, T);
        k_mtp_moe_combine<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(out, yd_, wts_, sh_, gate_, n, K);
        ck(cudaGetLastError(), "MTP moe (grouped)");
        return;
    }
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
    enqueue(h_prev, tokens, T, pos0, out_from, logits_dev, nullptr);
}

void MtpHead::enqueue(const float* h_prev, const int32_t* tokens, int T, int pos0, int out_from, float* logits_dev, const int32_t* dp) {
    if (T < 1 || T > std::max(max_batch_, chunk_.cap)) throw std::runtime_error("MtpHead: bad batch size");
    use_bufs(T > max_batch_ ? chunk_ : dec_);
    const int n = s_.d_model, hc = s_.hc_count, il = il_;
    const BlockCtx c{s_, w_, *scr_, stream_, dp};
    const BlockCtx ct{ts_, tw_, *scr_, stream_, dp};   // the target's embedding and head
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
        if (!vocab_ids_.empty()) linear(ct, head_, norm_, logits_dev, R);
        else head_logits(ct, norm_, R, logits_dev);
    }
    ck(cudaGetLastError(), "MtpHead::forward");
}

}  // namespace flashrt::qwen4exp
