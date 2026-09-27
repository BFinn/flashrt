// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
#include "arch/qwen4exp/blocks.hpp"

#include "kernels/cuda/ggml_gemv.h"

#include <algorithm>
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
    b = BlockScratch{};
}

void linear(const BlockCtx& c, const GpuTensor& W, const float* x, float* y, int T) {
    const int64_t cols = W.cols(), rows = W.rows();
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
    k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(x, static_cast<const float*>(w_norm.dev), xn, n, hc, float(s.rms_eps));
    linear(c, w_down, xn, lo, T);
    k_scale_silu<<<(T * s.hc_rank + 255) / 256, 256, 0, c.stream>>>(lo, T * s.hc_rank, 1.0f / hc);
    linear(c, w_up, lo, gate, T);
    k_gated_mean<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(xn, gate, mixed, n, hc, T);
    if (which != 2) linear(c, c.w.get(pre + "inject.weight"), xn, inject, T);
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
    k_gdn_delta<<<H, dk, size_t(2) * dk * 4, c.stream>>>(st.S, conv, alpha, beta, o, T, s.ssm_groups, H, dk, ch);
    if (o_inner) ck(cudaMemcpyAsync(o_inner, o, size_t(T) * inner * 4, cudaMemcpyDeviceToDevice, c.stream), "copy o");
    k_gated_rms_norm<<<T * H, 128, 0, c.stream>>>(o, static_cast<const float*>(c.w.layer(il, "ssm_norm.weight").dev), z, fin, dk,
                                                 float(s.rms_eps));
    linear(c, c.w.layer(il, "ssm_out.weight"), fin, out, T);
    ck(cudaGetLastError(), "gdn_mixer");
}

}  // namespace flashrt::qwen4exp
