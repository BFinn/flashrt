// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// The blocks' shared pieces: scratch, the linear layers, the token embedding, argmax, the head.
// The blocks themselves are in hc.cu, gdn.cu, qsa.cu, moe_ref.cu and ple.cu.
#include "arch/qwen4exp/blocks_common.cuh"

#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/q3r.h"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace flashrt::qwen4exp {

namespace {
// enough for the widest per-token intermediates of any block (hc_dim-wide norm output, gate,
// etc.) several times over
size_t scratch_f32_elems(const Spec& s, int max_tokens) { return size_t(max_tokens) * size_t(s.hc_count) * s.d_model * 5 + (size_t(1) << 20); }
size_t scratch_q8_bytes(const Spec& s) { return gemv::q8_1_bytes(std::max<int64_t>(int64_t(s.hc_count) * s.d_model, s.ssm_inner * 2), 8); }
size_t scratch_hc_bf16_bytes(const Spec& s) { return size_t(2) * s.hc_rank * s.hc_count * s.d_model * 2; }   // down and up
}  // namespace

uint64_t new_scratch_version() {
    static std::atomic<uint64_t> v{0};
    return ++v;
}

BlockScratch alloc_block_scratch(const Spec& s, int max_tokens) {
    BlockScratch b;
    b.version = new_scratch_version();
    b.f32_elems = scratch_f32_elems(s, max_tokens);
    ck(cudaMalloc(&b.f32, b.f32_elems * 4), "cudaMalloc block scratch");
    b.q8_bytes = scratch_q8_bytes(s);
    ck(cudaMalloc(&b.q8, b.q8_bytes), "cudaMalloc q8 scratch");
    b.hc_bf16_bytes = scratch_hc_bf16_bytes(s);
    if (b.hc_bf16_bytes) ck(cudaMalloc(&b.hc_bf16, b.hc_bf16_bytes), "cudaMalloc hc BF16 scratch");
    return b;
}

size_t block_scratch_bytes(const Spec& s, int max_tokens) {
    return scratch_f32_elems(s, max_tokens) * 4 + scratch_q8_bytes(s) + scratch_hc_bf16_bytes(s);
}

void free_block_scratch(BlockScratch& b) {
    if (b.f32) cudaFree(b.f32);
    if (b.q8) cudaFree(b.q8);
    if (b.idx_scores) cudaFree(b.idx_scores);
    if (b.idx_cells) cudaFree(b.idx_cells);
    if (b.idx_counts) cudaFree(b.idx_counts);
    if (b.attn_part) cudaFree(b.attn_part);
    if (b.gemm_ws) cudaFree(b.gemm_ws);
    if (b.q3k_tmp) cudaFree(b.q3k_tmp);
    if (b.q8_tmp) cudaFree(b.q8_tmp);
    if (b.gdn_ws) cudaFree(b.gdn_ws);
    if (b.hc_bf16) cudaFree(b.hc_bf16);
    if (b.tok_dev) cudaFree(b.tok_dev);
    b = BlockScratch{};
}

namespace {
struct Bf16Seg {
    const uint4* w;   // rows of K bf16, 8 per uint4
    float* y;
    const float* p0;
    const float* p1;
    int rows, epi;
};
struct Bf16Segs {
    Bf16Seg s[4];
    int n;
};
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// a block of 4 warps per output row over all segments: threads stride the row in pieces of 8
// (K % 256 == 0), partial sums meet in shared memory
template <int NT>
__global__ void __launch_bounds__(128) k_bf16_multi(Bf16Segs segs, const float* __restrict__ x, int K, int T) {
    __shared__ float red[4][NT];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int r = blockIdx.x, si = 0;
    while (si < segs.n && r >= segs.s[si].rows) r -= segs.s[si++].rows;
    if (si >= segs.n) return;   // block-uniform
    const Bf16Seg sg = si == 0 ? segs.s[0] : si == 1 ? segs.s[1] : si == 2 ? segs.s[2] : segs.s[3];
    const uint4* w = sg.w + size_t(r) * (K / 8);
    float acc[NT] = {};
    for (int c = threadIdx.x; c < K / 8; c += 128) {
        const uint4 wv = __ldg(w + c);
        const float wf[8] = {bf_lo(wv.x), bf_hi(wv.x), bf_lo(wv.y), bf_hi(wv.y), bf_lo(wv.z), bf_hi(wv.z), bf_lo(wv.w), bf_hi(wv.w)};
#pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < T) {
                const float4 a = __ldg(reinterpret_cast<const float4*>(x + size_t(t) * K) + 2 * c);
                const float4 b = __ldg(reinterpret_cast<const float4*>(x + size_t(t) * K) + 2 * c + 1);
                acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
            }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(~0u, acc[t], o);
        if (lane == 0) red[warp][t] = acc[t];
    }
    __syncthreads();
    if (threadIdx.x < T) {
        const int t = threadIdx.x;
        float v = red[0][t] + red[1][t] + red[2][t] + red[3][t];
        if (sg.epi == 1) {   // as k_gdn_gates
            const float z = v + sg.p0[r];
            v = (z > 20.0f ? z : log1pf(__expf(z))) * sg.p1[r];
        } else if (sg.epi == 2)
            v = 1.0f / (1.0f + __expf(-v));
        sg.y[size_t(t) * sg.rows + r] = v;
    }
}
}  // namespace

bool q8_act(const GpuTensor& W, int T) {
    return T >= 1 && T <= gemv::kMaxTokens && W.type < kTypeQ3R && gemv::supported(W.type) && !gemv::is_float(W.type);
}

void linear_shared(const BlockCtx& c, const GpuTensor* const* Ws, float* const* ys, int n, const float* x, int T) {
    using ggml_type::kBF16;
    if (T >= kGemmMinTokens) {   // prefill: the BF16 products share one conversion of x
        int nb = 0;
        for (int i = 0; i < n; ++i) nb += Ws[i]->type == kBF16 && Ws[i]->cols() == Ws[0]->cols();
        if (nb >= 2) {
            BlockScratch& bs = c.scratch;
            const int64_t cols = Ws[0]->cols();
            const size_t need = gemm::workspace_bytes(cols, T);
            if (bs.gemm_ws_bytes < need) {
                if (bs.gemm_ws) cudaFree(bs.gemm_ws);
                ck(cudaMalloc(&bs.gemm_ws, need), "cudaMalloc gemm workspace");
                bs.gemm_ws_bytes = need;
            }
            void* xb = gemm::bf16_staging(bs.gemm_ws, bs.gemm_ws_bytes, cols, T);
            gemm::to_bf16(x, xb, size_t(T) * cols, c.stream);
            for (int i = 0; i < n; ++i)   // the BF16 ones first: the others' gemm() reuses the staging
                if (Ws[i]->type == kBF16 && Ws[i]->cols() == cols) gemm::gemm_bf16(Ws[i]->dev, xb, ys[i], cols, Ws[i]->rows(), T, c.stream);
            for (int i = 0; i < n; ++i)
                if (!(Ws[i]->type == kBF16 && Ws[i]->cols() == cols)) linear(c, *Ws[i], x, ys[i], T);
            return;
        }
    }
    int nq = 0;
    for (int i = 0; i < n; ++i) nq += q8_act(*Ws[i], T);
    const bool share = nq >= 2 && Ws[0]->cols() > 0;
    if (share) gemv::quantize_q8_1(x, Ws[0]->cols(), T, c.scratch.q8, c.stream);
    for (int i = 0; i < n; ++i) {
        if (share && q8_act(*Ws[i], T)) {
            if (Ws[i]->cols() != Ws[0]->cols()) throw std::runtime_error("linear_shared: row lengths differ");
            gemv::matvec_q(Ws[i]->type, Ws[i]->dev, c.scratch.q8, ys[i], Ws[i]->cols(), Ws[i]->rows(), T, c.stream);
        } else linear(c, *Ws[i], x, ys[i], T);
    }
}

bool linear_multi_ok(const LinearOut* outs, int n, int T) {
    using ggml_type::kBF16;
    if (n < 1 || n > 4 || T < 1 || T > gemv::kMaxTokens) return false;
    for (int i = 0; i < n; ++i)
        if (outs[i].W->type != kBF16 || outs[i].W->cols() != outs[0].W->cols() || outs[i].W->cols() % 256) return false;
    return true;
}

void linear_multi(const BlockCtx& c, const LinearOut* outs, int n, const float* x, int T) {
    Bf16Segs segs{};
    segs.n = n;
    int rows = 0;
    for (int i = 0; i < n; ++i) {
        segs.s[i] = Bf16Seg{static_cast<const uint4*>(outs[i].W->dev), outs[i].y, outs[i].p0, outs[i].p1, int(outs[i].W->rows()), outs[i].epi};
        rows += segs.s[i].rows;
    }
    const int K = int(outs[0].W->cols());
    const unsigned grid = unsigned(rows);
    if (T == 1) k_bf16_multi<1><<<grid, 128, 0, c.stream>>>(segs, x, K, T);
    else if (T == 2) k_bf16_multi<2><<<grid, 128, 0, c.stream>>>(segs, x, K, T);
    else if (T <= 4) k_bf16_multi<4><<<grid, 128, 0, c.stream>>>(segs, x, K, T);
    else k_bf16_multi<8><<<grid, 128, 0, c.stream>>>(segs, x, K, T);
    ck(cudaGetLastError(), "linear_multi");
}

void linear(const BlockCtx& c, const GpuTensor& W, const float* x, float* y, int T) {
    const int64_t cols = W.cols(), rows = W.rows();
    uint32_t mm_type = W.type == kTypeQ3R ? ggml_type::kQ3_K : W.type;
    if (T >= kGemmMinTokens && gemm::supported(mm_type)) {
        BlockScratch& bs = c.scratch;
        const void* Wp = W.dev;
        if (W.type == kTypeQ3R) {   // back to Q3_K for ggml's kernel
            const size_t need = size_t(rows) * (cols / 256) * 110 + gemv::kWeightTailPad;
            if (bs.q3k_tmp_bytes < need) {
                if (bs.q3k_tmp) cudaFree(bs.q3k_tmp);
                ck(cudaMalloc(&bs.q3k_tmp, need), "cudaMalloc q3k copy");
                ck(cudaMemset(bs.q3k_tmp, 0, need), "memset q3k copy");
                bs.q3k_tmp_bytes = need;
            }
            q3r::unpack(W.dev, bs.q3k_tmp, rows, cols, c.stream);
            Wp = bs.q3k_tmp;
        }
        // Q3_K -> Q8_0 (exact; ggml's Q8_0 MMQ is about 1.37x faster than its Q3_K, sw69)
        if (mm_type == ggml_type::kQ3_K && cols % 256 == 0) {
            const size_t need = size_t(rows) * (cols / 32) * 34 + gemv::kWeightTailPad;
            if (bs.q8_tmp_bytes < need) {
                if (bs.q8_tmp) cudaFree(bs.q8_tmp);
                ck(cudaMalloc(&bs.q8_tmp, need), "cudaMalloc q8_0 copy");
                ck(cudaMemset(bs.q8_tmp, 0, need), "memset q8_0 copy");
                bs.q8_tmp_bytes = need;
            }
            q3r::q3k_to_q8_0(Wp, bs.q8_tmp, rows, cols, c.stream);
            Wp = bs.q8_tmp;
            mm_type = ggml_type::kQ8_0;
        }
        const size_t ws = gemm::workspace_bytes(cols, T);
        if (bs.gemm_ws_bytes < ws) {
            if (bs.gemm_ws) cudaFree(bs.gemm_ws);
            ck(cudaMalloc(&bs.gemm_ws, ws), "cudaMalloc gemm workspace");
            bs.gemm_ws_bytes = ws;
        }
        gemm::gemm(mm_type, Wp, x, y, cols, rows, T, bs.gemm_ws, bs.gemm_ws_bytes, c.stream);
        return;
    }
    if (W.type == kTypeQ3R) {
        q3r::matvec(W.dev, x, y, rows, cols, T, c.stream);
        return;
    }
    for (int t0 = 0; t0 < T; t0 += gemv::kMaxTokens) {
        const int n = std::min(gemv::kMaxTokens, T - t0);
        gemv::matvec(W.type, W.dev, x + size_t(t0) * cols, y + size_t(t0) * rows, cols, rows, n, c.scratch.q8, c.stream);
    }
}

namespace {
// element e (of K) of row `row` of a Q3_K table, to float (110-byte blocks of 256 values)
__device__ __forceinline__ float q3k_value(const uint8_t* table, int32_t row, int e, int K) {
    const uint8_t* b = table + (size_t(row) * (K / 256) + e / 256) * 110;
    const int el = e % 256, n = el / 128, j = (el % 128) / 32, l = el % 32, is = el / 16;
    const int q = ((b[32 + 32 * n + l] >> (2 * j)) & 3) | (((b[l] >> (4 * n + j)) & 1) << 2);
    const uint8_t* sc = b + 96;
    const int us = is < 4 ? (sc[is] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4)
                 : is < 8 ? (sc[is] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4)
                 : is < 12 ? (sc[is - 8] >> 4) | (((sc[is] >> 4) & 3) << 4)
                           : (sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3) << 4);
    return __half2float(*reinterpret_cast<const __half*>(b + 108)) * float(us - 32) * float(q - 4);
}
// one Q3_K row per token (row t: token dp[t ? 2 + t : 0], see BlockCtx::dparams) to float; one
// thread per element
__global__ void k_embed_q3k(const uint8_t* table, const int32_t* dp, float* out, int K) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e < K) out[size_t(t) * K + e] = q3k_value(table, dp[t ? 2 + t : 0], e, K);
}
// the same for many tokens: row t (blockIdx.y) of token tokens[t]
__global__ void k_embed_q3k_tok(const uint8_t* table, const int32_t* tokens, float* out, int K) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e < K) out[size_t(t) * K + e] = q3k_value(table, tokens[t], e, K);
}
}  // namespace

bool embed_graph_capable(const GpuWeights& w) {
    const GpuTensor& e = w.get("token_embd.weight");
    return e.type == ggml_type::kQ3_K && e.cols() % 256 == 0;
}

void embed(const BlockCtx& c, const int32_t* tokens, int T, float* out) {
    const GpuTensor& e = c.w.get("token_embd.weight");
    if (c.dparams) {   // graph mode: the token id is on the device
        if (T > kMaxGraphTokens || !embed_graph_capable(c.w)) throw std::runtime_error("embed: graph mode needs <= 8 tokens and a Q3_K table");
        k_embed_q3k<<<dim3(unsigned((e.cols() + 255) / 256), T), 256, 0, c.stream>>>(static_cast<const uint8_t*>(e.dev), c.dparams, out,
                                                                          int(e.cols()));
        ck(cudaGetLastError(), "embed");
        return;
    }
    if (T > 8 && embed_graph_capable(c.w)) {   // many tokens: one kernel over the token ids
        BlockScratch& bs = c.scratch;
        if (bs.tok_cap < size_t(T)) {
            if (bs.tok_dev) cudaFree(bs.tok_dev);
            ck(cudaMalloc(&bs.tok_dev, size_t(T) * 4), "cudaMalloc tokens");
            bs.tok_cap = size_t(T);
        }
        ck(cudaMemcpyAsync(bs.tok_dev, tokens, size_t(T) * 4, cudaMemcpyHostToDevice, c.stream), "tokens to device");
        k_embed_q3k_tok<<<dim3(unsigned((e.cols() + 255) / 256), T), 256, 0, c.stream>>>(static_cast<const uint8_t*>(e.dev), bs.tok_dev, out,
                                                                                       int(e.cols()));
        ck(cudaGetLastError(), "embed");
        return;
    }
    const size_t rb = size_t(gemv::row_bytes(e.type, e.cols()));
    for (int t = 0; t < T; ++t)
        gemv::dequantize(e.type, static_cast<const char*>(e.dev) + size_t(tokens[t]) * rb, out + size_t(t) * e.cols(), e.cols(),
                         c.stream);
    ck(cudaGetLastError(), "embed");
}

namespace {
// argmax over x[0 .. n), lowest index on ties, NaN never taken (all -inf or NaN: INT32_MAX), on a
// cluster of kArgmaxCluster CTAs (P-5): one CTA read the 1 MB of logits at about 13 GB/s (79 us per
// call at 262K, sw122; 5 us here, sw126). Each thread takes every (8 * 1024)-th value, four loads in
// flight, in rising index order; each CTA reduces its warps' results, and CTA 0 takes the CTAs'
// results in rank order through distributed shared memory.
constexpr int kArgmaxCluster = 8;
__global__ void __cluster_dims__(kArgmaxCluster, 1, 1) __launch_bounds__(1024) k_argmax_cl(const float* x, int n, int32_t* out) {
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    __shared__ float bv[32];
    __shared__ int bi[32];
    constexpr int S = kArgmaxCluster * 1024;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    int i = int(cl.block_rank()) * 1024 + threadIdx.x;
    for (; i + 3 * S < n; i += 4 * S) {
        float xi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) xi[k] = x[i + k * S];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (xi[k] > v) { v = xi[k]; idx = i + k * S; }
    }
    for (; i < n; i += S) {
        const float xi = x[i];
        if (xi > v) { v = xi; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffff, v, o);
        const int i2 = __shfl_xor_sync(0xffffffff, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w)
            if (bv[w] > v || (bv[w] == v && bi[w] < idx)) { v = bv[w]; idx = bi[w]; }
        bv[0] = v;
        bi[0] = idx;
    }
    cl.sync();   // every CTA's result is in its bv[0] / bi[0]
    if (cl.block_rank() == 0 && threadIdx.x == 0) {
        for (int k = 1; k < kArgmaxCluster; ++k) {
            const float v2 = *cl.map_shared_rank(&bv[0], k);
            const int i2 = *cl.map_shared_rank(&bi[0], k);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        out[0] = idx;
    }
    cl.sync();   // no CTA exits while CTA 0 still reads its result
}
}  // namespace

void argmax_dev(cudaStream_t stream, const float* x, int n, int32_t* out_dev) {
    k_argmax_cl<<<kArgmaxCluster, 1024, 0, stream>>>(x, n, out_dev);
    ck(cudaGetLastError(), "argmax");
}

void head_logits(const BlockCtx& c, const float* norm, int T, float* logits) {
    linear(c, c.w.get("output.weight"), norm, logits, T);
}

}  // namespace flashrt::qwen4exp
