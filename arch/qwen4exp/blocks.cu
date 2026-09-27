// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
#include "arch/qwen4exp/blocks.hpp"

#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/q3r.h"
#include "quant/q2_0/moe_cpu.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ float block_sum(float v) {
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? red[threadIdx.x] : 0.0f;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

// one block per (token, stream): y = x / rms(x) * w[stream]
__global__ void k_grouped_rms_norm(const float* x, const float* w, float* y, int n, int hc, float eps) {
    const int row = blockIdx.x;            // t * hc + s
    const int s = row % hc;
    const float* xr = x + size_t(row) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[size_t(row) * n + i] = xr[i] * inv * w[size_t(s) * n + i];
}

__global__ void k_scale_silu(float* x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i] * scale;
        x[i] = v / (1.0f + __expf(-v));
    }
}

// mixed[t][i] = mean over s of xn[t][s][i] * sigmoid(gate[t][s][i])
// As k_grouped_rms_norm for n % 4 == 0 and blockDim.x == n / 4: one float4 per thread, kept in
// registers between the reduction and the write.
__global__ void k_grouped_rms_norm_v4(const float* x, const float* w, float* y, int n, int hc, float eps) {
    const int row = blockIdx.x, s = row % hc, i = threadIdx.x;
    const float4 v = reinterpret_cast<const float4*>(x + size_t(row) * n)[i];
    const float ss = block_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float inv = rsqrtf(ss / n + eps);
    const float4 wv = reinterpret_cast<const float4*>(w + size_t(s) * n)[i];
    reinterpret_cast<float4*>(y + size_t(row) * n)[i] = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
}

// Decode (one token): the hyper-connection RMS norm, down-projection and inject projection in
// one kernel. Block (rb, g) normalises stream g (xn = x * inv_g * w) into shared memory (block
// row 0 also writes it out), then its warps take one row each of [W_down; W_inject] (BF16,
// rank + n_inject rows of hc*n columns) over stream g's n columns. part[g][row] holds the
// per-stream partial dot products; consumers sum them over g in order (deterministic).
__global__ void k_hc_down(const float* x, const float* w_norm, const uint16_t* Wd, const uint16_t* Wi, float* xn_out,
                          float* part, int n, int rank, int n_inject, float eps) {
    extern __shared__ __align__(16) float xs[];   // [n]
    const int g = blockIdx.y, rows = rank + n_inject;
    const float* xg = x + size_t(g) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xg[i] * xg[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xg[i] * inv * w_norm[size_t(g) * n + i];
        xs[i] = v;
        if (blockIdx.x == 0) xn_out[size_t(g) * n + i] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const size_t hcn = size_t(gridDim.y) * n;
    const uint16_t* wr = (r < rank ? Wd + size_t(r) * hcn : Wi + size_t(r - rank) * hcn) + size_t(g) * n;
    const uint4* w4 = reinterpret_cast<const uint4*>(wr);
    float acc = 0.0f;
#pragma unroll 5
    for (int ch = lane; ch < n / 8; ch += 32) {
        const uint4 u = w4[ch];
        const float4 x0 = reinterpret_cast<const float4*>(xs)[2 * ch], x1 = reinterpret_cast<const float4*>(xs)[2 * ch + 1];
        const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc += __uint_as_float(wv[e] << 16) * xv[2 * e];
            acc += __uint_as_float(wv[e] & 0xffff0000u) * xv[2 * e + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) part[size_t(g) * rows + r] = acc;
}

// Decode (one token), hc == 4: the hyper-connection up-projection fused with its neighbours.
// gate = W_up (BF16, [hc * n rows][rank]) * silu(lo * scale); mixed[i] = mean_s xn[s][i] *
// sigmoid(gate[s][i]). One block covers columns i0 .. i0+7 of all 4 streams (32 rows); 8 lanes
// per row, each lane 16-byte loads of 8 weights. Needs rank % 64 == 0 and rank <= 1024.
// lo comes as k_hc_down's partials [HC][rank + n_inject]; block 0 also sums the inject rows.
__global__ void k_hc_up_mix(const uint16_t* W, const float* part, int n_inject, float scale, const float* xn, float* mixed,
                            float* inject, int n, int rank) {
    constexpr int HC = 4;
    __shared__ __align__(16) float xs[1024];
    __shared__ float contrib[HC][8];
    const int prow = rank + n_inject;
    for (int j = threadIdx.x; j < rank; j += blockDim.x) {
        float lo = 0.0f;
#pragma unroll
        for (int g = 0; g < HC; ++g) lo += part[size_t(g) * prow + j];
        const float v = lo * scale;
        xs[j] = v / (1.0f + __expf(-v));
    }
    if (blockIdx.x == 0 && threadIdx.x < n_inject) {
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < HC; ++g) a += part[size_t(g) * prow + rank + threadIdx.x];
        inject[threadIdx.x] = a;
    }
    __syncthreads();
    const int grp = threadIdx.x >> 3, l8 = threadIdx.x & 7;
    const int st = grp >> 3, il = grp & 7, i = blockIdx.x * 8 + il;
    float acc = 0.0f;
    if (i < n) {
        const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t(st) * n + i) * rank);
        for (int ch = l8; ch < rank / 8; ch += 8) {
            const uint4 u = wr[ch];
            const float4 x0 = reinterpret_cast<const float4*>(xs)[2 * ch], x1 = reinterpret_cast<const float4*>(xs)[2 * ch + 1];
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc += __uint_as_float(wv[e] << 16) * xv[2 * e];
                acc += __uint_as_float(wv[e] & 0xffff0000u) * xv[2 * e + 1];
            }
        }
    }
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    if (l8 == 0 && i < n) {
        const float x = xn[size_t(st) * n + i];
        contrib[st][il] = x / (1.0f + __expf(-acc));
    }
    __syncthreads();
    if (threadIdx.x < 8) {
        const int ii = blockIdx.x * 8 + threadIdx.x;
        if (ii < n) {
            float m = 0.0f;
            for (int s2 = 0; s2 < HC; ++s2) m += contrib[s2][threadIdx.x];
            mixed[ii] = m * (1.0f / HC);
        }
    }
}

__global__ void k_gated_mean(const float* xn, const float* gate, float* mixed, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t k = (size_t(t) * hc + s) * n + i;
        acc += xn[k] / (1.0f + __expf(-gate[k]));
    }
    mixed[size_t(t) * n + i] = acc * (1.0f / hc);
}

__global__ void k_hc_combine(float* x, const float* out, const float* inject, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    const float o = out[size_t(t) * n + i];
    for (int s = 0; s < hc; ++s) {
        const float wv = 2.0f / (1.0f + __expf(-inject[t * hc + s] / hc));
        x[(size_t(t) * hc + s) * n + i] += o * wv;
    }
}

// ---- GDN kernels

// one thread per channel: causal conv over the (conv-1)-token history and T new tokens, silu;
// the history is advanced to the last (conv-1) inputs
__global__ void k_gdn_conv(const float* qkv, float* hist, const float* w, float* y, int channels, int T, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[8];
    for (int k = 0; k < K - 1; ++k) win[k] = hist[size_t(k) * channels + c];
    for (int t = 0; t < T; ++t) {
        win[K - 1] = qkv[size_t(t) * channels + c];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += win[k] * w[size_t(c) * K + k];
        y[size_t(t) * channels + c] = acc / (1.0f + __expf(-acc));
        for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
    }
    for (int k = 0; k < K - 1; ++k) hist[size_t(k) * channels + c] = win[k];
}

// one block per (token, head) of `dim` values: x /= sqrt(sum x^2 + eps), in place
__global__ void k_l2_norm(float* x, int dim, int stride_tok, int heads, float eps) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    float* v = x + size_t(t) * stride_tok + size_t(h) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += v[i] * v[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] *= inv;
}

// per (token, value head): g = softplus(alpha + dt_bias) * a, beta = sigmoid(beta)
__global__ void k_gdn_gates(float* alpha, float* beta, const float* dt_bias, const float* a, int heads, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * T) return;
    const int h = i % heads;
    const float z = alpha[i] + dt_bias[h];
    const float sp = z > 20.0f ? z : log1pf(__expf(z));
    alpha[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + __expf(-beta[i]));
}

// one block per value head, one thread per value row i; S is [head][j][i]. For each token:
// S *= exp(g); d = beta * (v - S k); S += d k^T; o = S q / sqrt(dk). Key head = h % k_heads.
__global__ void k_gdn_delta(float* S, const float* conv_out, const float* g, const float* beta, float* o, int T,
                            int k_heads, int v_heads, int dk, int channels) {
    extern __shared__ float sh[];   // q[dk], k[dk]
    float* qs = sh;
    float* ks = sh + dk;
    const int h = blockIdx.x, i = threadIdx.x, hk = h % k_heads;
    float* Sh = S + size_t(h) * dk * dk;
    const float scale = rsqrtf(float(dk));
    for (int t = 0; t < T; ++t) {
        const float* row = conv_out + size_t(t) * channels;
        for (int j = threadIdx.x; j < dk; j += blockDim.x) {
            qs[j] = row[size_t(hk) * dk + j] * scale;
            ks[j] = row[size_t(k_heads) * dk + size_t(hk) * dk + j];
        }
        __syncthreads();
        const float decay = __expf(g[t * v_heads + h]);
        const float b = beta[t * v_heads + h];
        const float vi = row[size_t(2 * k_heads) * dk + size_t(h) * dk + i];
        float sk = 0.0f;
        for (int j = 0; j < dk; ++j) {
            const float sv = Sh[size_t(j) * dk + i] * decay;
            Sh[size_t(j) * dk + i] = sv;
            sk += sv * ks[j];
        }
        const float d = b * (vi - sk);
        float oi = 0.0f;
        for (int j = 0; j < dk; ++j) {
            const float sv = Sh[size_t(j) * dk + i] + d * ks[j];
            Sh[size_t(j) * dk + i] = sv;
            oi += sv * qs[j];
        }
        o[(size_t(t) * v_heads + h) * dk + i] = oi;
        __syncthreads();
    }
}

// The same delta rule with the state in registers: one block per (head, 32 columns), 32 x 8
// threads; thread (x, y) holds S[j][i] for column i = 32 * blockIdx.y + x and rows
// j = y * DK/8 .. +DK/8, so S is read and written once per call however many tokens it has.
template <int DK>
__global__ void k_gdn_delta_reg(float* S, const float* conv_out, const float* g, const float* beta, float* o, int T,
                                int k_heads, int v_heads, int channels) {
    constexpr int JG = 8, JPT = DK / JG;
    __shared__ float qs[DK], ks[DK];
    __shared__ float red[JG][33];
    const int h = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y, i = blockIdx.y * 32 + tx, hk = h % k_heads;
    const int tid = ty * 32 + tx;
    float* Sh = S + size_t(h) * DK * DK;
    float st[JPT];
#pragma unroll
    for (int jj = 0; jj < JPT; ++jj) st[jj] = Sh[size_t(ty * JPT + jj) * DK + i];
    const float scale = rsqrtf(float(DK));
    for (int t = 0; t < T; ++t) {
        const float* row = conv_out + size_t(t) * channels;
        for (int j = tid; j < DK; j += 32 * JG) {
            qs[j] = row[size_t(hk) * DK + j] * scale;
            ks[j] = row[size_t(k_heads) * DK + size_t(hk) * DK + j];
        }
        __syncthreads();
        const float decay = __expf(g[t * v_heads + h]);
        float sk = 0.0f;
#pragma unroll
        for (int jj = 0; jj < JPT; ++jj) {
            st[jj] *= decay;
            sk += st[jj] * ks[ty * JPT + jj];
        }
        red[ty][tx] = sk;
        __syncthreads();
        float tot = 0.0f;
#pragma unroll
        for (int y = 0; y < JG; ++y) tot += red[y][tx];
        const float vi = row[size_t(2 * k_heads) * DK + size_t(h) * DK + i];
        const float d = beta[t * v_heads + h] * (vi - tot);
        float oi = 0.0f;
#pragma unroll
        for (int jj = 0; jj < JPT; ++jj) {
            st[jj] += d * ks[ty * JPT + jj];
            oi += st[jj] * qs[ty * JPT + jj];
        }
        __syncthreads();
        red[ty][tx] = oi;
        __syncthreads();
        if (ty == 0) {
            float r = 0.0f;
#pragma unroll
            for (int y = 0; y < JG; ++y) r += red[y][tx];
            o[(size_t(t) * v_heads + h) * DK + i] = r;
        }
        __syncthreads();
    }
#pragma unroll
    for (int jj = 0; jj < JPT; ++jj) Sh[size_t(ty * JPT + jj) * DK + i] = st[jj];
}

// per (token, head): y = rms_norm(o) * w * sigmoid(z)
__global__ void k_gated_rms_norm(const float* o, const float* w, const float* z, float* y, int dim, float eps) {
    const size_t base = size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += o[base + i] * o[base + i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y[base + i] = o[base + i] * inv * w[i] / (1.0f + __expf(-z[base + i]));
}

}  // namespace

BlockScratch alloc_block_scratch(const Spec& s, int max_tokens) {
    BlockScratch b;
    // enough for the widest per-token intermediates of any block (hc_dim-wide norm output,
    // gate, etc.) several times over
    b.f32_elems = size_t(max_tokens) * size_t(s.hc_count) * s.d_model * 4 + (size_t(1) << 20);
    ck(cudaMalloc(&b.f32, b.f32_elems * 4), "cudaMalloc block scratch");
    b.q8_bytes = gemv::q8_1_bytes(std::max<int64_t>(int64_t(s.hc_count) * s.d_model, s.ssm_inner * 2), 8);
    ck(cudaMalloc(&b.q8, b.q8_bytes), "cudaMalloc q8 scratch");
    return b;
}

void free_block_scratch(BlockScratch& b) {
    if (b.f32) cudaFree(b.f32);
    if (b.q8) cudaFree(b.q8);
    if (b.idx_scores) cudaFree(b.idx_scores);
    if (b.idx_cells) cudaFree(b.idx_cells);
    if (b.idx_counts) cudaFree(b.idx_counts);
    if (b.attn_part) cudaFree(b.attn_part);
    b = BlockScratch{};
}

void linear(const BlockCtx& c, const GpuTensor& W, const float* x, float* y, int T) {
    const int64_t cols = W.cols(), rows = W.rows();
    if (T == 1 && W.q3r) {
        q3r::matvec(W.q3r, x, y, rows, cols, c.stream);
        return;
    }
    for (int t0 = 0; t0 < T; t0 += gemv::kMaxTokens) {
        const int n = std::min(gemv::kMaxTokens, T - t0);
        gemv::matvec(W.type, W.dev, x + size_t(t0) * cols, y + size_t(t0) * rows, cols, rows, n, c.scratch.q8, c.stream);
    }
}

void hc_mix(const BlockCtx& c, int il, int which, const float* x, int T, float* mixed, float* inject, float* xn_out) {
    const Spec& s = c.s;
    const int n = s.d_model, hc = s.hc_count, hcd = hc * n;
    std::string pre;
    if (which == 2) pre = "output_hc_";
    else pre = "blk." + std::to_string(il) + (which == 0 ? ".hc_attn_" : ".hc_ffn_");
    const GpuTensor& w_norm = c.w.get(pre + "norm.weight");
    const GpuTensor& w_down = c.w.get(pre + "down.weight");
    const GpuTensor& w_up = c.w.get(pre + "up.weight");

    float* xn = xn_out ? xn_out : c.scratch.f32;                          // [T][hcd]
    float* lo = c.scratch.f32 + size_t(T) * hcd;                          // [T][rank]
    float* gate = lo + size_t(T) * s.hc_rank;                             // [T][hcd]
    const GpuTensor* w_inj = which != 2 ? &c.w.get(pre + "inject.weight") : nullptr;
    constexpr uint32_t kBF16 = 30;   // GGML_TYPE_BF16
    if (T == 1 && hc == 4 && s.hc_rank % 64 == 0 && s.hc_rank <= 1024 && n % 8 == 0 && size_t(n) * 4 <= 48 * 1024 &&
        w_up.type == kBF16 && w_down.type == kBF16 && (!w_inj || w_inj->type == kBF16)) {
        // decode: norm + down + inject in one kernel, then up + silu + gated mean in another
        const int n_inj = w_inj ? hc : 0, rows = s.hc_rank + n_inj;
        float* part = lo;   // [hc][rows], fits: lo is followed by gate [hcd]
        k_hc_down<<<dim3((rows + 15) / 16, hc), 512, size_t(n) * 4, c.stream>>>(
            x, static_cast<const float*>(w_norm.dev), static_cast<const uint16_t*>(w_down.dev),
            w_inj ? static_cast<const uint16_t*>(w_inj->dev) : nullptr, xn, part, n, s.hc_rank, n_inj, float(s.rms_eps));
        k_hc_up_mix<<<(n + 7) / 8, 256, 0, c.stream>>>(static_cast<const uint16_t*>(w_up.dev), part, n_inj, 1.0f / hc, xn, mixed,
                                                       inject, n, s.hc_rank);
        ck(cudaGetLastError(), "hc_mix");
        return;
    }
    if (n % 4 == 0 && n / 4 <= 1024 && (n / 4) % 32 == 0)
        k_grouped_rms_norm_v4<<<T * hc, n / 4, 0, c.stream>>>(x, static_cast<const float*>(w_norm.dev), xn, n, hc, float(s.rms_eps));
    else
        k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(x, static_cast<const float*>(w_norm.dev), xn, n, hc, float(s.rms_eps));
    linear(c, w_down, xn, lo, T);
    k_scale_silu<<<(T * s.hc_rank + 255) / 256, 256, 0, c.stream>>>(lo, T * s.hc_rank, 1.0f / hc);
    linear(c, w_up, lo, gate, T);
    k_gated_mean<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(xn, gate, mixed, n, hc, T);
    if (w_inj) linear(c, *w_inj, xn, inject, T);
    ck(cudaGetLastError(), "hc_mix");
}

void hc_combine(const BlockCtx& c, float* x, const float* out, const float* inject, int T) {
    const int n = c.s.d_model;
    k_hc_combine<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(x, out, inject, n, c.s.hc_count, T);
    ck(cudaGetLastError(), "hc_combine");
}

}  // namespace flashrt::qwen4exp

namespace flashrt::qwen4exp {

namespace {
int gdn_channels(const Spec& s) { return 2 * s.ssm_groups * s.ssm_state + s.ssm_heads * s.ssm_state; }
}

GdnState alloc_gdn_state(const Spec& s) {
    GdnState st;
    ck(cudaMalloc(&st.S, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4), "cudaMalloc gdn S");
    ck(cudaMalloc(&st.conv, size_t(s.ssm_conv - 1) * gdn_channels(s) * 4), "cudaMalloc gdn conv");
    reset_gdn_state(s, st, nullptr);
    return st;
}

void reset_gdn_state(const Spec& s, GdnState& st, cudaStream_t stream) {
    ck(cudaMemsetAsync(st.S, 0, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4, stream), "memset gdn S");
    ck(cudaMemsetAsync(st.conv, 0, size_t(s.ssm_conv - 1) * gdn_channels(s) * 4, stream), "memset gdn conv");
}

void free_gdn_state(GdnState& st) {
    if (st.S) cudaFree(st.S);
    if (st.conv) cudaFree(st.conv);
    st = GdnState{};
}

void gdn_mixer(const BlockCtx& c, int il, const float* x, int T, GdnState& st, float* out, float* o_inner) {
    const Spec& s = c.s;
    const int ch = gdn_channels(s), H = s.ssm_heads, dk = s.ssm_state, inner = H * dk;
    if (s.ssm_conv > 8 || dk > 1024) throw std::runtime_error("gdn_mixer: unsupported shape");
    float* qkv = c.scratch.f32;                    // [T][ch]
    float* conv = qkv + size_t(T) * ch;           // [T][ch]
    float* z = conv + size_t(T) * ch;             // [T][inner]
    float* alpha = z + size_t(T) * inner;         // [T][H]
    float* beta = alpha + size_t(T) * H;          // [T][H]
    float* o = beta + size_t(T) * H;              // [T][inner]
    float* fin = o + size_t(T) * inner;           // [T][inner]
    if (size_t(fin + size_t(T) * inner - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("gdn_mixer: scratch too small");

    linear(c, c.w.layer(il, "attn_qkv.weight"), x, qkv, T);
    linear(c, c.w.layer(il, "attn_gate.weight"), x, z, T);
    linear(c, c.w.layer(il, "ssm_beta.weight"), x, beta, T);
    linear(c, c.w.layer(il, "ssm_alpha.weight"), x, alpha, T);
    k_gdn_conv<<<(ch + 127) / 128, 128, 0, c.stream>>>(qkv, st.conv, static_cast<const float*>(c.w.layer(il, "ssm_conv1d.weight").dev),
                                                      conv, ch, T, s.ssm_conv);
    // L2-normalise the q and k heads (the first 2 * groups heads of each token's channels)
    k_l2_norm<<<T * 2 * s.ssm_groups, 128, 0, c.stream>>>(conv, dk, ch, 2 * s.ssm_groups, float(s.rms_eps));
    k_gdn_gates<<<(T * H + 127) / 128, 128, 0, c.stream>>>(alpha, beta, static_cast<const float*>(c.w.layer(il, "ssm_dt.bias").dev),
                                                          static_cast<const float*>(c.w.layer(il, "ssm_a").dev), H, T);
    if (dk == 128) k_gdn_delta_reg<128><<<dim3(H, 128 / 32), dim3(32, 8), 0, c.stream>>>(st.S, conv, alpha, beta, o, T, s.ssm_groups, H, ch);
    else k_gdn_delta<<<H, dk, size_t(2) * dk * 4, c.stream>>>(st.S, conv, alpha, beta, o, T, s.ssm_groups, H, dk, ch);
    if (o_inner) ck(cudaMemcpyAsync(o_inner, o, size_t(T) * inner * 4, cudaMemcpyDeviceToDevice, c.stream), "copy o");
    k_gated_rms_norm<<<T * H, 128, 0, c.stream>>>(o, static_cast<const float*>(c.w.layer(il, "ssm_norm.weight").dev), z, fin, dk,
                                                 float(s.rms_eps));
    linear(c, c.w.layer(il, "ssm_out.weight"), fin, out, T);
    ck(cudaGetLastError(), "gdn_mixer");
}

}  // namespace flashrt::qwen4exp

namespace flashrt::qwen4exp {

namespace {

// per (token, head): RMS norm over dim with weight w, then NEOX rope on the first n_rot dims
// at position pos0 + token; src rows are `src_stride` floats apart per token and `head_stride`
// per head, dst is [T][heads][dim]

// round_fp16: store values rounded to fp16 (the KV cache format of the parity reference)
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_out(__half* p, float v) { *p = __float2half(v); }

// OutT float: plain (queries); OutT __half: the K cache (round to fp16, as llama.cpp's F16 cache)
template <typename OutT>
__global__ void k_norm_rope(const float* src, int src_stride, int head_stride, const float* w, OutT* dst, int heads,
                            int dim, int n_rot, int pos0, float theta_scale, float eps) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const float* x = src + size_t(t) * src_stride + size_t(h) * head_stride;
    OutT* y = dst + (size_t(t) * heads + h) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    const int half = n_rot / 2;
    const float pos = float(pos0 + t);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i < half) {
            const float theta = pos * powf(theta_scale, float(i));
            float sn, cs;
            sincosf(theta, &sn, &cs);
            const float x0 = x[i] * inv * w[i], x1 = x[i + half] * inv * w[i + half];
            store_out(y + i, x0 * cs - x1 * sn);
            store_out(y + i + half, x0 * sn + x1 * cs);
        } else if (i >= n_rot) {
            store_out(y + i, x[i] * inv * w[i]);
        }
    }
}

__global__ void k_copy_h(const float* src, __half* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ---- split-K flash-decode attention
constexpr int kAttnSplit = 64;     // cells per partial
constexpr int kAttnMaxGroup = 16;  // query heads per KV head

// One block per (split, kv head, token), 256 threads, head dim D == 256: the partial softmax of
// the group's G query heads over up to kAttnSplit cells. Cells come from the token's list, or
// are 0..pos when counts[t] < 0. Writes, per (token, head, split): max, sum, acc[D].
template <int G>
__global__ void k_attn_part(const float* q, const __half* K, const __half* V, int heads, int kv_heads, int pos0, float scale,
                            const int32_t* cells, const int32_t* counts, int ldc, int n_splits, float* part) {
    constexpr int D = 256;
    __shared__ __align__(16) float qs[G][D];
    __shared__ float sc[G][kAttnSplit];
    __shared__ int cell_ids[kAttnSplit];
    const int split = blockIdx.x, hk = blockIdx.y, t = blockIdx.z;
    const int cnt = counts ? counts[t] : -1;
    const int n = cnt >= 0 ? cnt : pos0 + t + 1;
    const int j0 = split * kAttnSplit, nj = min(kAttnSplit, n - j0);
    float* pt = part + ((size_t(t) * heads + hk * G) * n_splits + split) * (D + 2);
    if (nj <= 0) {   // empty split: neutral partial
        for (int h = 0; h < G; ++h)
            if (threadIdx.x == 0) {
                pt[size_t(h) * n_splits * (D + 2) + 0] = -INFINITY;
                pt[size_t(h) * n_splits * (D + 2) + 1] = 0.0f;
            }
        return;
    }
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) qs[i / D][i % D] = q[(size_t(t) * heads + hk * G + i / D) * D + i % D];
    for (int j = threadIdx.x; j < nj; j += blockDim.x) cell_ids[j] = cnt >= 0 ? cells[size_t(t) * ldc + j0 + j] : j0 + j;
    __syncthreads();
    // scores: warp w takes cells w, w + 8, ...; lane covers dims [lane*4, +4) and [128 + lane*4, +4)
    // (coalesced K rows, conflict-free shared q)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int j = warp; j < nj; j += blockDim.x >> 5) {
        const uint2* kr = reinterpret_cast<const uint2*>(K + (size_t(cell_ids[j]) * kv_heads + hk) * D);   // 4 halves
        const uint2 u0 = kr[lane], u1 = kr[32 + lane];
        const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u0.x)), a1 = __half22float2(*reinterpret_cast<const __half2*>(&u0.y));
        const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&u1.x)), b1 = __half22float2(*reinterpret_cast<const __half2*>(&u1.y));
        const float4 k0 = make_float4(a0.x, a0.y, a1.x, a1.y), k1 = make_float4(b0.x, b0.y, b1.x, b1.y);
        float d[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 q0 = reinterpret_cast<const float4*>(qs[h])[lane], q1 = reinterpret_cast<const float4*>(qs[h])[32 + lane];
            d[h] = q0.x * k0.x + q0.y * k0.y + q0.z * k0.z + q0.w * k0.w + q1.x * k1.x + q1.y * k1.y + q1.z * k1.z + q1.w * k1.w;
        }
#pragma unroll
        for (int h = 0; h < G; ++h)
            for (int o = 16; o > 0; o >>= 1) d[h] += __shfl_xor_sync(0xffffffff, d[h], o);
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < G; ++h) sc[h][j] = d[h] * scale;
    }
    __syncthreads();
    // per head: max and exp over the split
    __shared__ float mx[G], sm[G];
    if (threadIdx.x < G) {
        const int h = threadIdx.x;
        float m = -INFINITY;
        for (int j = 0; j < nj; ++j) m = fmaxf(m, sc[h][j]);
        float s = 0.0f;
        for (int j = 0; j < nj; ++j) {
            const float e = expf(sc[h][j] - m);
            sc[h][j] = e;
            s += e;
        }
        mx[h] = m;
        sm[h] = s;
    }
    __syncthreads();
    // V: thread d accumulates dimension d for every head
    const int dd = threadIdx.x;
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int j = 0; j < nj; ++j) {
        const float v = __half2float(V[(size_t(cell_ids[j]) * kv_heads + hk) * D + dd]);
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] += sc[h][j] * v;
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        float* ph = pt + size_t(h) * n_splits * (D + 2);
        ph[2 + dd] = acc[h];
        if (dd == 0) {
            ph[0] = mx[h];
            ph[1] = sm[h];
        }
    }
}

// One block per (token, head): combine the splits, then the sigmoid output gate.
__global__ void k_attn_combine(const float* part, int n_splits, const float* qfull, float* o, int heads, int dim,
                               int gate_stride, int gate_head_stride, int gate_off) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const float* ph = part + (size_t(t) * heads + h) * n_splits * (dim + 2);
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, ph[size_t(s) * (dim + 2)]);
    float denom = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float ms = ph[size_t(s) * (dim + 2)];
        if (ms > -INFINITY) denom += ph[size_t(s) * (dim + 2) + 1] * expf(ms - m);
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float ms = ph[size_t(s) * (dim + 2)];
            if (ms > -INFINITY) acc += ph[size_t(s) * (dim + 2) + 2 + i] * expf(ms - m);
        }
        const float g = qfull[size_t(t) * gate_stride + size_t(h) * gate_head_stride + gate_off + i];
        o[(size_t(t) * heads + h) * dim + i] = acc / denom / (1.0f + expf(-g));
    }
}

// Pool every block completed by a token of this call: mean of the block's raw keys (from this
// call's keys, or the ring for earlier positions), RMS norm, NEOX rope at the block's first
// position. One CUDA block per token; tokens that complete no block exit.
__global__ void k_idx_pool(const float* kraw, const float* ring, const float* w, float* pooled, int pos0, int r, int dim,
                           int n_rot, float theta_scale, float eps) {
    const int t = blockIdx.x, p = pos0 + t;
    if (p % r != r - 1) return;
    const int b = p / r;
    __shared__ float m[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < r; ++k) {
            const int pk = p - (r - 1) + k;
            acc += pk >= pos0 ? kraw[size_t(pk - pos0) * dim + i] : ring[size_t(pk % r) * dim + i];
        }
        acc /= float(r);
        m[i] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    __syncthreads();
    const int half = n_rot / 2;
    float* y = pooled + size_t(b) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i < half) {
            float sn, cs;
            sincosf(float(b * r) * powf(theta_scale, float(i)), &sn, &cs);
            const float x0 = m[i] * inv * w[i], x1 = m[i + half] * inv * w[i + half];
            y[i] = x0 * cs - x1 * sn;
            y[i + half] = x0 * sn + x1 * cs;
        } else if (i >= n_rot) {
            y[i] = m[i] * inv * w[i];
        }
    }
}

// the last r raw keys of this call go to ring slot position % r
__global__ void k_idx_ring(const float* kraw, float* ring, int pos0, int T, int r, int dim) {
    const int t = T - 1 - int(blockIdx.x);   // the last min(T, r) tokens
    const int p = pos0 + t;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ring[size_t(p % r) * dim + i] = kraw[size_t(t) * dim + i];
}

// score[t][b] = sum over heads of relu(q[t][h] . pooled[b]), for the blocks complete at token t
__global__ void k_idx_scores(const float* qi, const float* pooled, float* scores, int ld, int pos0, int r, int heads,
                             int dim) {
    const int t = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = (pos0 + t + 1) / r;
    if (b >= nb) return;
    const float* kb = pooled + size_t(b) * dim;
    float sum = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const float* qh = qi + (size_t(t) * heads + h) * dim;
        float d = 0.0f;
        for (int i = 0; i < dim; ++i) d += qh[i] * kb[i];
        sum += fmaxf(d, 0.0f);
    }
    scores[size_t(t) * ld + b] = sum;
}

__device__ __forceinline__ unsigned ordered_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// block-wide exclusive scan of 0/1 flags (blockDim a multiple of 32, <= 1024); returns the
// thread's rank and sets *total
__device__ int block_scan_flags(bool flag, int* total) {
    __shared__ int warp_tot[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned bal = __ballot_sync(0xffffffff, flag);
    const int in_warp = __popc(bal & ((1u << lane) - 1));
    if (lane == 0) warp_tot[warp] = __popc(bal);
    __syncthreads();
    if (warp == 0) {
        int v = lane < nw ? warp_tot[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffff, v, o);
            if (lane >= o) v += n;
        }
        if (lane < nw) warp_tot[lane] = v;   // inclusive
    }
    __syncthreads();
    const int before = warp > 0 ? warp_tot[warp - 1] : 0;
    *total = warp_tot[nw - 1];
    __syncthreads();
    return before + in_warp;
}

// Per token (one CUDA block): the cells to attend to. Dense (counts = -1) while q + 1 <= width;
// otherwise the top M blocks by score, M = nsel minus one when the incomplete tail exists, then
// the tail cells. The M-th largest score is found by a 32-pass radix select; ties at it are taken
// in block order.
__global__ void k_idx_select(const float* scores, int ld, int32_t* cells, int32_t* counts, int ldc, int pos0, int r,
                             int nsel, int width) {
    const int t = blockIdx.x, q = pos0 + t;
    if (q + 1 <= width) {
        if (threadIdx.x == 0) counts[t] = -1;
        return;
    }
    const int nb = (q + 1) / r, tail = (q + 1) - nb * r, M = nsel - (tail > 0 ? 1 : 0);
    const float* sc = scores + size_t(t) * ld;
    int32_t* out = cells + size_t(t) * ldc;
    __shared__ int cnt;
    int filled;
    if (nb <= M) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
            for (int k = 0; k < r; ++k) out[b * r + k] = b * r + k;
        filled = nb;
    } else {
        unsigned prefix = 0;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned cand = prefix | (1u << bit);
            if (threadIdx.x == 0) cnt = 0;
            __syncthreads();
            int local = 0;
            for (int b = threadIdx.x; b < nb; b += blockDim.x) local += ordered_key(sc[b]) >= cand;
            atomicAdd(&cnt, local);
            __syncthreads();
            if (cnt >= M) prefix = cand;
            __syncthreads();
        }
        const unsigned tau = prefix;
        // blocks above tau, plus the lowest-index blocks equal to tau up to M, written in block
        // order (deterministic, so the attention sums in a fixed order)
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        int local = 0;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) local += ordered_key(sc[b]) > tau;
        atomicAdd(&cnt, local);
        __syncthreads();
        const int tie_budget = M - cnt;
        int ties = 0, written = 0;
        for (int base = 0; base < nb; base += blockDim.x) {
            const int b = base + threadIdx.x;
            const unsigned key = b < nb ? ordered_key(sc[b]) : 0u;
            const bool tie = b < nb && key == tau;
            int tie_total, sel_total;
            const int tie_rank = block_scan_flags(tie, &tie_total);
            const bool sel = b < nb && (key > tau || (tie && ties + tie_rank < tie_budget));
            const int pos = block_scan_flags(sel, &sel_total);
            if (sel)
                for (int k = 0; k < r; ++k) out[(written + pos) * r + k] = b * r + k;
            ties += tie_total;
            written += sel_total;
        }
        filled = M;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < tail; k += blockDim.x) out[filled * r + k] = nb * r + k;
    if (threadIdx.x == 0) counts[t] = filled * r + tail;
}

}  // namespace

QsaCache alloc_qsa_cache(const Spec& s, int capacity) {
    QsaCache kv;
    kv.capacity = capacity;
    const size_t n = size_t(capacity) * s.n_head_kv * s.head_dim_k;
    ck(cudaMalloc(&kv.K, n * 2), "cudaMalloc K cache");
    ck(cudaMalloc(&kv.V, n * 2), "cudaMalloc V cache");
    ck(cudaMalloc(&kv.idx_pooled, size_t(capacity / s.qsa_block + 1) * s.idx_dim * 4), "cudaMalloc pooled keys");
    ck(cudaMalloc(&kv.idx_ring, size_t(s.qsa_block) * s.idx_dim * 4), "cudaMalloc key ring");
    ck(cudaMemset(kv.idx_ring, 0, size_t(s.qsa_block) * s.idx_dim * 4), "memset key ring");
    return kv;
}

void free_qsa_cache(QsaCache& kv) {
    if (kv.K) cudaFree(kv.K);
    if (kv.V) cudaFree(kv.V);
    if (kv.idx_pooled) cudaFree(kv.idx_pooled);
    if (kv.idx_ring) cudaFree(kv.idx_ring);
    kv = QsaCache{};
}

void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out, float* gated_out,
               std::vector<std::vector<int32_t>>* sel_out) {
    const Spec& s = c.s;
    const int H = s.n_head, KH = s.n_head_kv, D = s.head_dim_k;
    const int r = s.qsa_block, IH = s.idx_heads, ID = s.idx_dim;
    const int width = s.idx_top_k + r - 1, nsel = (width + r - 1) / r;
    if (pos0 + T > kv.capacity) throw std::runtime_error("qsa_mixer: KV cache full");
    if (s.head_dim_v != D || ID > 256) throw std::runtime_error("qsa_mixer: unsupported shape");
    float* qfull = c.scratch.f32;                   // [T][H * 2D]  (q | gate per head)
    float* kraw = qfull + size_t(T) * H * 2 * D;    // [T][KH * D]
    float* vraw = kraw + size_t(T) * KH * D;        // [T][KH * D]
    float* q = vraw + size_t(T) * KH * D;           // [T][H][D]
    float* o = q + size_t(T) * H * D;               // [T][H][D]
    float* qi = o + size_t(T) * H * D;              // [T][IH * ID] indexer queries (normed, roped in place)
    float* ki = qi + size_t(T) * IH * ID;           // [T][ID] indexer raw keys
    if (size_t(ki + size_t(T) * ID - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("qsa_mixer: scratch too small");

    linear(c, c.w.layer(il, "attn_q.weight"), x, qfull, T);
    linear(c, c.w.layer(il, "attn_k.weight"), x, kraw, T);
    linear(c, c.w.layer(il, "attn_v.weight"), x, vraw, T);
    const float theta_scale = powf(float(s.rope_base), -2.0f / float(s.rope_dims));
    const float eps = float(s.rms_eps);
    k_norm_rope<float><<<T * H, 128, 0, c.stream>>>(qfull, H * 2 * D, 2 * D, static_cast<const float*>(c.w.layer(il, "attn_q_norm.weight").dev),
                                                   q, H, D, s.rope_dims, pos0, theta_scale, eps);
    // K goes straight into the cache rows pos0.. (cell = position); V is copied as is
    __half* Kc = reinterpret_cast<__half*>(kv.K);
    __half* Vc = reinterpret_cast<__half*>(kv.V);
    k_norm_rope<__half><<<T * KH, 128, 0, c.stream>>>(kraw, KH * D, D, static_cast<const float*>(c.w.layer(il, "attn_k_norm.weight").dev),
                                                     Kc + size_t(pos0) * KH * D, KH, D, s.rope_dims, pos0, theta_scale, eps);
    k_copy_h<<<(T * KH * D + 255) / 256, 256, 0, c.stream>>>(vraw, Vc + size_t(pos0) * KH * D, T * KH * D);

    // indexer: queries, raw keys, pooling of the blocks completed here, then per-token selection
    linear(c, c.w.layer(il, "indexer.q_proj.weight"), x, qi, T);
    linear(c, c.w.layer(il, "indexer.k_proj.weight"), x, ki, T);
    k_norm_rope<float><<<T * IH, 128, 0, c.stream>>>(qi, IH * ID, ID, static_cast<const float*>(c.w.layer(il, "indexer.q_norm.weight").dev),
                                                    qi, IH, ID, s.rope_dims, pos0, theta_scale, eps);
    k_idx_pool<<<T, 128, 0, c.stream>>>(ki, kv.idx_ring, static_cast<const float*>(c.w.layer(il, "indexer.k_norm.weight").dev),
                                       kv.idx_pooled, pos0, r, ID, s.rope_dims, theta_scale, eps);
    k_idx_ring<<<std::min(T, r), 128, 0, c.stream>>>(ki, kv.idx_ring, pos0, T, r, ID);

    const int32_t* cells = nullptr;
    const int32_t* counts = nullptr;
    const int ldc = nsel * r;
    if (pos0 + T > width) {   // some token needs a selection
        const int max_nb = (pos0 + T) / r;
        BlockScratch& bs = c.scratch;
        if (bs.idx_scores_elems < size_t(T) * max_nb) {
            if (bs.idx_scores) cudaFree(bs.idx_scores);
            bs.idx_scores_elems = size_t(T) * max_nb * 2;
            ck(cudaMalloc(&bs.idx_scores, bs.idx_scores_elems * 4), "cudaMalloc idx scores");
        }
        if (bs.idx_cells_elems < size_t(T) * ldc) {
            if (bs.idx_cells) cudaFree(bs.idx_cells);
            if (bs.idx_counts) cudaFree(bs.idx_counts);
            bs.idx_cells_elems = size_t(T) * ldc * 2;
            ck(cudaMalloc(&bs.idx_cells, bs.idx_cells_elems * 4), "cudaMalloc idx cells");
            ck(cudaMalloc(&bs.idx_counts, bs.idx_cells_elems / ldc * 4), "cudaMalloc idx counts");
        }
        k_idx_scores<<<dim3((max_nb + 127) / 128, T), 128, 0, c.stream>>>(qi, kv.idx_pooled, bs.idx_scores, max_nb, pos0, r, IH, ID);
        k_idx_select<<<T, 1024, 0, c.stream>>>(bs.idx_scores, max_nb, bs.idx_cells, bs.idx_counts, ldc, pos0, r, nsel, width);
        cells = bs.idx_cells;
        counts = bs.idx_counts;
        if (sel_out) {
            std::vector<int32_t> hc(size_t(T) * ldc), hn(T);
            ck(cudaMemcpyAsync(hc.data(), cells, hc.size() * 4, cudaMemcpyDeviceToHost, c.stream), "cells to host");
            ck(cudaMemcpyAsync(hn.data(), counts, hn.size() * 4, cudaMemcpyDeviceToHost, c.stream), "counts to host");
            ck(cudaStreamSynchronize(c.stream), "sync");
            sel_out->assign(T, {});
            for (int t = 0; t < T; ++t)
                if (hn[t] >= 0) (*sel_out)[t].assign(hc.begin() + size_t(t) * ldc, hc.begin() + size_t(t) * ldc + hn[t]);
        }
    } else if (sel_out) {
        sel_out->assign(T, {});
    }
    // split-K flash decode: partials per (token, head, 64-cell split), then a combine
    const int G = H / KH;
    if (D != 256 || G > kAttnMaxGroup || H % KH) throw std::runtime_error("qsa_mixer: attention needs head_dim 256 and a group <= 16");
    const int n_splits = (std::max(width, ldc) + kAttnSplit - 1) / kAttnSplit;
    {
        BlockScratch& bs = c.scratch;
        const size_t need = size_t(T) * H * n_splits * (D + 2);
        if (bs.attn_part_elems < need) {
            if (bs.attn_part) cudaFree(bs.attn_part);
            bs.attn_part_elems = need;
            ck(cudaMalloc(&bs.attn_part, need * 4), "cudaMalloc attention partials");
        }
    }
    const dim3 grid(n_splits, KH, T);
    const float scale = 1.0f / sqrtf(float(D));
    switch (G) {
        case 12: k_attn_part<12><<<grid, 256, 0, c.stream>>>(q, Kc, Vc, H, KH, pos0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part); break;
        case 8: k_attn_part<8><<<grid, 256, 0, c.stream>>>(q, Kc, Vc, H, KH, pos0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part); break;
        case 16: k_attn_part<16><<<grid, 256, 0, c.stream>>>(q, Kc, Vc, H, KH, pos0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part); break;
        default: throw std::runtime_error("qsa_mixer: unsupported GQA group size");
    }
    k_attn_combine<<<T * H, 256, 0, c.stream>>>(c.scratch.attn_part, n_splits, qfull, o, H, D, H * 2 * D, 2 * D, D);
    if (gated_out) ck(cudaMemcpyAsync(gated_out, o, size_t(T) * H * D * 4, cudaMemcpyDeviceToDevice, c.stream), "copy gated");
    linear(c, c.w.layer(il, "attn_output.weight"), o, out, T);
    ck(cudaGetLastError(), "qsa_mixer");
}

}  // namespace flashrt::qwen4exp

namespace flashrt::qwen4exp {

namespace {

// out = moe + shexp * sigmoid(gate): shexp [T][n], gate [T] (one logit per token)
__global__ void k_add_gated(float* out, const float* moe, const float* shexp, const float* gate, int n, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    const float g = 1.0f / (1.0f + __expf(-gate[t]));
    out[size_t(t) * n + i] = moe[size_t(t) * n + i] + shexp[size_t(t) * n + i] * g;
}

__global__ void k_swiglu(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

}  // namespace

void moe_block(const BlockCtx& c, int il, const float* x, int T, MoeHost& h, float* out, MoeTrace* trace) {
    const Spec& s = c.s;
    const int n = s.d_model, E = s.n_expert, K = s.top_k, ff = s.d_ff_shared;
    float* logits = c.scratch.f32;                 // [T][E]
    float* sg = logits + size_t(T) * E;            // [T][ff]
    float* su = sg + size_t(T) * ff;               // [T][ff]
    float* sh = su + size_t(T) * ff;               // [T][n]
    float* gate = sh + size_t(T) * n;              // [T]
    float* moe_dev = gate + T;                     // [T][n]
    if (size_t(moe_dev + size_t(T) * n - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("moe_block: scratch too small");

    // GPU: router logits and the shared expert
    linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, logits, T);
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, sg, T);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, su, T);
    k_swiglu<<<(T * ff + 255) / 256, 256, 0, c.stream>>>(sg, su, T * ff);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), sg, sh, T);
    linear(c, c.w.layer(il, "ffn_gate_inp_shexp.weight"), x, gate, T);

    // host: routing and the routed experts
    h.x.resize(size_t(T) * n);
    h.logits.resize(size_t(T) * E);
    h.out.resize(size_t(T) * n);
    ck(cudaMemcpyAsync(h.x.data(), x, h.x.size() * 4, cudaMemcpyDeviceToHost, c.stream), "x to host");
    ck(cudaMemcpyAsync(h.logits.data(), logits, h.logits.size() * 4, cudaMemcpyDeviceToHost, c.stream), "logits to host");
    ck(cudaStreamSynchronize(c.stream), "sync");

    const q2_0::ExpertShape es{n, s.d_ff_expert};
    const size_t qb = q2_0::q8_bytes(n);
    h.act_mem.resize(size_t(T) * qb + 64);
    std::vector<q2_0::Q8Act> acts(T);
    auto base = (reinterpret_cast<uintptr_t>(h.act_mem.data()) + 63) & ~uintptr_t(63);
    for (int t = 0; t < T; ++t) {
        acts[t] = q2_0::q8_view(reinterpret_cast<void*>(base + size_t(t) * qb), n);
        q2_0::quantize_q8(h.x.data() + size_t(t) * n, acts[t]);
    }
    if (trace) {
        trace->topk.assign(size_t(T) * K, 0);
        trace->probs.assign(size_t(T) * K, 0.0f);
    }
    // per expert, the tokens routed to it (in token order) with their weights
    std::vector<std::vector<std::pair<int, float>>> routed(E);
    std::vector<float> p(E);
    std::vector<int> idx(E);
    for (int t = 0; t < T; ++t) {
        const float* lg = h.logits.data() + size_t(t) * E;
        const float mx = *std::max_element(lg, lg + E);
        double sum = 0;
        for (int e = 0; e < E; ++e) sum += p[e] = std::exp(lg[e] - mx);
        for (int e = 0; e < E; ++e) p[e] = float(p[e] / sum);
        for (int e = 0; e < E; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return p[a] > p[b]; });
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) wsum += p[idx[k]];
        wsum = std::max(wsum, 6.103515625e-5f);
        for (int k = 0; k < K; ++k) {
            if (h.counts) ++(*h.counts)[size_t(il) * E + idx[k]];
            routed[idx[k]].push_back({t, p[idx[k]] / wsum});
            if (trace) {
                trace->topk[size_t(t) * K + k] = idx[k];
                trace->probs[size_t(t) * K + k] = p[idx[k]];
            }
        }
    }
    std::vector<q2_0::Miss> miss;
    for (int e = 0; e < E; ++e) {
        const auto& r = routed[e];
        for (size_t i = 0; i < r.size(); i += 4) {
            q2_0::Miss m{h.arena->blob(il, e), 0, {}, {}};
            for (size_t j = i; j < std::min(r.size(), i + 4); ++j) {
                m.tok[m.n_tok] = r[j].first;
                m.w[m.n_tok] = r[j].second;
                ++m.n_tok;
            }
            miss.push_back(m);
        }
    }
    h.scratch.resize(q2_0::moe_cpu_scratch_bytes(es, int(miss.size()), h.pool->size()));
    q2_0::moe_cpu(*h.pool, es, miss.data(), int(miss.size()), acts.data(), T, h.out.data(), n, h.scratch.data());

    ck(cudaMemcpyAsync(moe_dev, h.out.data(), h.out.size() * 4, cudaMemcpyHostToDevice, c.stream), "moe to device");
    k_add_gated<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(out, moe_dev, sh, gate, n, T);
    ck(cudaStreamSynchronize(c.stream), "moe_block");
}

}  // namespace flashrt::qwen4exp

namespace flashrt::qwen4exp {

namespace {

// per (token, stream): s = <keyn, queryn> / sqrt(n); gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))
__global__ void k_ple_gate(const float* keyn, const float* queryn, float* gate, int n) {
    const size_t base = size_t(blockIdx.x) * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += keyn[base + i] * queryn[base + i];
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        const float sv = acc / sqrtf(float(n));
        const float mag = sqrtf(fmaxf(fabsf(sv), 1e-6f));
        const float sg = sv > 0.0f ? 1.0f : (sv < 0.0f ? -1.0f : 0.0f);
        gate[blockIdx.x] = 1.0f / (1.0f + __expf(-sg * mag));
    }
}

// gated[t][s][i] = value[t][i] * gate[t][s]
__global__ void k_ple_gated(const float* value, const float* gate, float* gated, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    for (int s = 0; s < hc; ++s) gated[(size_t(t) * hc + s) * n + i] = value[size_t(t) * n + i] * gate[t * hc + s];
}

// one thread per channel c of hc*n: out[t] = silu(sum_k w[k] * x[t - (K-1-k)*dil]) over a history of
// (K-1)*dil earlier inputs; x += gated + out. The history is advanced.
__global__ void k_ple_conv(float* x, const float* gated, const float* normed, float* hist, const __half* w, int C, int T,
                           int K, int dil) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int H = (K - 1) * dil;          // history length (<= 32)
    float ring[32];
    for (int j = 0; j < H; ++j) ring[j] = hist[size_t(j) * C + c];   // oldest first
    for (int t = 0; t < T; ++t) {
        const float cur = normed[size_t(t) * C + c];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int back = (K - 1 - k) * dil;              // positions back from t
            const float v = back == 0 ? cur : ring[H - back];
            acc += __half2float(w[size_t(c) * K + k]) * v;
        }
        const float o = acc / (1.0f + __expf(-acc));
        x[size_t(t) * C + c] += gated[size_t(t) * C + c] + o;
        for (int j = 0; j + 1 < H; ++j) ring[j] = ring[j + 1];
        if (H > 0) ring[H - 1] = cur;
    }
    for (int j = 0; j < H; ++j) hist[size_t(j) * C + c] = ring[j];
}

}  // namespace

void embed(const BlockCtx& c, const int32_t* tokens, int T, float* out) {
    const GpuTensor& e = c.w.get("token_embd.weight");
    const size_t rb = size_t(gemv::row_bytes(e.type, e.cols()));
    for (int t = 0; t < T; ++t)
        gemv::dequantize(e.type, static_cast<const char*>(e.dev) + size_t(tokens[t]) * rb, out + size_t(t) * e.cols(), e.cols(),
                         c.stream);
    ck(cudaGetLastError(), "embed");
}

PleState alloc_ple_state(const Spec& s, const Ple& p) {
    PleState st;
    ck(cudaMalloc(&st.hist, size_t(p.ngram) * 3 * size_t(s.hc_count) * s.d_model * 4 + 4), "cudaMalloc ple hist");
    reset_ple_state(s, p, st, nullptr);
    return st;
}

void reset_ple_state(const Spec& s, const Ple& p, PleState& st, cudaStream_t stream) {
    ck(cudaMemsetAsync(st.hist, 0, size_t(p.ngram) * 3 * size_t(s.hc_count) * s.d_model * 4 + 4, stream), "memset ple hist");
}

void free_ple_state(PleState& st) {
    if (st.hist) cudaFree(st.hist);
    st = PleState{};
}

void ple_embed(const BlockCtx& c, PleHost& h, const int32_t* seq, int64_t pos0, int T, float* emb) {
    const Ple& p = *h.ple;
    h.rows.resize(size_t(T) * p.n_heads);
    for (int t = 0; t < T; ++t) ple_rows(p, seq, pos0 + t, h.rows.data() + size_t(t) * p.n_heads);
    h.raw.resize(h.rows.size() * p.row_bytes);
    h.reader->fetch(h.rows.data(), h.rows.size(), h.raw.data());
    if (h.raw_dev_bytes < h.raw.size()) {
        if (h.raw_dev) cudaFree(h.raw_dev);
        ck(cudaMalloc(&h.raw_dev, h.raw.size()), "cudaMalloc ple rows");
        h.raw_dev_bytes = h.raw.size();
    }
    ck(cudaMemcpyAsync(h.raw_dev, h.raw.data(), h.raw.size(), cudaMemcpyHostToDevice, c.stream), "ple rows to device");
    // rows are whole IQ4_NL blocks back to back: one dequantize covers every head of every token
    const int64_t per_row = int64_t(p.row_bytes) / 18 * 32;
    gemv::dequantize(20 /* IQ4_NL */, h.raw_dev, emb, int64_t(h.rows.size()) * per_row, c.stream);
    ck(cudaStreamSynchronize(c.stream), "ple_embed");   // h.raw is reused by the next call
}

void ple_fetch(PleHost& h, const int32_t* seq, int64_t pos0, int T) {
    const Ple& p = *h.ple;
    h.rows.resize(size_t(T) * p.n_heads);
    for (int t = 0; t < T; ++t) ple_rows(p, seq, pos0 + t, h.rows.data() + size_t(t) * p.n_heads);
    const size_t bytes = h.rows.size() * p.row_bytes;
    if (h.raw_pinned_bytes < bytes) {
        if (h.raw_pinned) cudaFreeHost(h.raw_pinned);
        ck(cudaHostAlloc(&h.raw_pinned, bytes, cudaHostAllocDefault), "cudaHostAlloc ple rows");
        h.raw_pinned_bytes = bytes;
    }
    h.reader->fetch(h.rows.data(), h.rows.size(), h.raw_pinned);
}

void ple_upload(const BlockCtx& c, PleHost& h, int T, float* emb) {
    const Ple& p = *h.ple;
    const size_t bytes = size_t(T) * p.n_heads * p.row_bytes;
    if (h.raw_dev_bytes < bytes) {
        if (h.raw_dev) cudaFree(h.raw_dev);
        ck(cudaMalloc(&h.raw_dev, bytes), "cudaMalloc ple rows");
        h.raw_dev_bytes = bytes;
    }
    ck(cudaMemcpyAsync(h.raw_dev, h.raw_pinned, bytes, cudaMemcpyHostToDevice, c.stream), "ple rows to device");
    const int64_t per_row = int64_t(p.row_bytes) / 18 * 32;
    gemv::dequantize(20 /* IQ4_NL */, h.raw_dev, emb, int64_t(T) * p.n_heads * per_row, c.stream);
}

namespace {
// one block of 1024 threads: argmax over x[0 .. n), lowest index on ties
__global__ void k_argmax(const float* x, int n, int32_t* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = x[i];
        if (xi > v) { v = xi; idx = i; }   // i rises, so ties keep the lowest index
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffff, v, o);
        const int i2 = __shfl_xor_sync(0xffffffff, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (blockDim.x >> 5); ++w)
            if (bv[w] > v || (bv[w] == v && bi[w] < idx)) { v = bv[w]; idx = bi[w]; }
        out[0] = idx;
    }
}
}  // namespace

void argmax_dev(cudaStream_t stream, const float* x, int n, int32_t* out_dev) {
    k_argmax<<<1, 1024, 0, stream>>>(x, n, out_dev);
    ck(cudaGetLastError(), "argmax");
}

void ple_block(const BlockCtx& c, int il, const Ple& p, const float* emb, float* x, int T, PleState& st) {
    const Spec& s = c.s;
    const int n = s.d_model, hc = s.hc_count, C = hc * n;
    const int K = int(c.w.layer(il, "ple_conv1d.weight").dims.at(0));
    if ((K - 1) * p.ngram > 32 || (K - 1) * p.ngram > 3 * p.ngram) throw std::runtime_error("ple_block: conv history too long");
    float* key = c.scratch.f32;                      // [T][C]
    float* value = key + size_t(T) * C;              // [T][n]
    float* qn = value + size_t(T) * n;               // [T][C]
    float* gate = qn + size_t(T) * C;                // [T][hc]
    float* gated = gate + size_t(T) * hc;            // [T][C]
    float* normed = gated + size_t(T) * C;           // [T][C]
    if (size_t(normed + size_t(T) * C - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("ple_block: scratch too small");
    const float eps = float(s.rms_eps);
    linear(c, c.w.layer(il, "ple_key.weight"), emb, key, T);
    linear(c, c.w.layer(il, "ple_value.weight"), emb, value, T);
    k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(key, static_cast<const float*>(c.w.layer(il, "ple_norm_key.weight").dev), key, n, hc, eps);
    k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(x, static_cast<const float*>(c.w.layer(il, "ple_norm_query.weight").dev), qn, n, hc, eps);
    k_ple_gate<<<T * hc, 256, 0, c.stream>>>(key, qn, gate, n);
    k_ple_gated<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(value, gate, gated, n, hc, T);
    k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(gated, static_cast<const float*>(c.w.layer(il, "ple_norm_conv.weight").dev), normed, n, hc, eps);
    k_ple_conv<<<(C + 127) / 128, 128, 0, c.stream>>>(x, gated, normed, st.hist, static_cast<const __half*>(c.w.layer(il, "ple_conv1d.weight").dev),
                                                     C, T, K, p.ngram);
    ck(cudaGetLastError(), "ple_block");
}

void head_logits(const BlockCtx& c, const float* norm, int T, float* logits) {
    linear(c, c.w.get("output.weight"), norm, logits, T);
}

}  // namespace flashrt::qwen4exp
