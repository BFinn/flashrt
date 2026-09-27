// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
#include "arch/qwen4exp/blocks.hpp"

#include "kernels/cuda/ggml_gemv.h"
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

namespace flashrt::qwen4exp {

namespace {

// per (token, head): RMS norm over dim with weight w, then NEOX rope on the first n_rot dims
// at position pos0 + token; src rows are `src_stride` floats apart per token and `head_stride`
// per head, dst is [T][heads][dim]
__device__ __forceinline__ float round_h(float v, bool r) { return r ? __half2float(__float2half(v)) : v; }

// round_fp16: store values rounded to fp16 (the KV cache format of the parity reference)
__global__ void k_norm_rope(const float* src, int src_stride, int head_stride, const float* w, float* dst, int heads,
                            int dim, int n_rot, int pos0, float theta_scale, float eps, bool round_fp16) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const float* x = src + size_t(t) * src_stride + size_t(h) * head_stride;
    float* y = dst + (size_t(t) * heads + h) * dim;
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
            y[i] = round_h(x0 * cs - x1 * sn, round_fp16);
            y[i + half] = round_h(x0 * sn + x1 * cs, round_fp16);
        } else if (i >= n_rot) {
            y[i] = round_h(x[i] * inv * w[i], round_fp16);
        }
    }
}

__global__ void k_copy_round_h(const float* src, float* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(__float2half(src[i]));
}

// dense causal attention, one block per (token, head), blockDim = dim threads:
// scores over cells 0..pos (pos = pos0 + t), softmax, weighted V; output gated by sigmoid(gate)
__global__ void k_attn_dense(const float* q, const float* K, const float* V, const float* qfull, float* o, int heads,
                             int kv_heads, int dim, int pos0, float scale, int gate_stride, int gate_head_stride,
                             int gate_off) {
    extern __shared__ float sc[];   // scores [pos + 1]
    const int t = blockIdx.x / heads, h = blockIdx.x % heads, hk = h / (heads / kv_heads);
    const int n = pos0 + t + 1;
    const float* qh = q + (size_t(t) * heads + h) * dim;
    // scores: each thread takes cells j = tid, tid + blockDim, ...
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float* kj = K + (size_t(j) * kv_heads + hk) * dim;
        float d = 0.0f;
        for (int i = 0; i < dim; ++i) d += qh[i] * kj[i];
        d *= scale;
        sc[j] = d;
        mx = fmaxf(mx, d);
    }
    // block max
    __shared__ float red[32];
    for (int of = 16; of > 0; of >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, of));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        float m = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : -INFINITY;
        for (int of = 16; of > 0; of >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, of));
        if (threadIdx.x == 0) red[0] = m;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = __expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum);
    __syncthreads();
    // output dim i = threadIdx.x
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) acc += sc[j] * V[(size_t(j) * kv_heads + hk) * dim + i];
        const float g = qfull[size_t(t) * gate_stride + size_t(h) * gate_head_stride + gate_off + i];
        o[(size_t(t) * heads + h) * dim + i] = acc / sum / (1.0f + __expf(-g));
    }
}

}  // namespace

QsaCache alloc_qsa_cache(const Spec& s, int capacity) {
    QsaCache kv;
    kv.capacity = capacity;
    const size_t n = size_t(capacity) * s.n_head_kv * s.head_dim_k;
    ck(cudaMalloc(&kv.K, n * 4), "cudaMalloc K cache");
    ck(cudaMalloc(&kv.V, n * 4), "cudaMalloc V cache");
    return kv;
}

void free_qsa_cache(QsaCache& kv) {
    if (kv.K) cudaFree(kv.K);
    if (kv.V) cudaFree(kv.V);
    kv = QsaCache{};
}

void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out, float* gated_out) {
    const Spec& s = c.s;
    const int H = s.n_head, KH = s.n_head_kv, D = s.head_dim_k;
    const int width = s.idx_top_k + s.qsa_block - 1;
    if (pos0 + T > width) throw std::runtime_error("qsa_mixer: contexts beyond the selection width need the indexer (not implemented yet)");
    if (pos0 + T > kv.capacity) throw std::runtime_error("qsa_mixer: KV cache full");
    if (s.head_dim_v != D) throw std::runtime_error("qsa_mixer: head_dim_v != head_dim_k");
    float* qfull = c.scratch.f32;                   // [T][H * 2D]  (q | gate per head)
    float* kraw = qfull + size_t(T) * H * 2 * D;    // [T][KH * D]
    float* vraw = kraw + size_t(T) * KH * D;        // [T][KH * D]
    float* q = vraw + size_t(T) * KH * D;           // [T][H][D]
    float* o = q + size_t(T) * H * D;               // [T][H][D]
    if (size_t(o + size_t(T) * H * D - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("qsa_mixer: scratch too small");

    linear(c, c.w.layer(il, "attn_q.weight"), x, qfull, T);
    linear(c, c.w.layer(il, "attn_k.weight"), x, kraw, T);
    linear(c, c.w.layer(il, "attn_v.weight"), x, vraw, T);
    const float theta_scale = powf(float(s.rope_base), -2.0f / float(s.rope_dims));
    const float eps = float(s.rms_eps);
    k_norm_rope<<<T * H, 128, 0, c.stream>>>(qfull, H * 2 * D, 2 * D, static_cast<const float*>(c.w.layer(il, "attn_q_norm.weight").dev),
                                            q, H, D, s.rope_dims, pos0, theta_scale, eps, false);
    // K goes straight into the cache rows pos0.. (cell = position); V is copied as is
    k_norm_rope<<<T * KH, 128, 0, c.stream>>>(kraw, KH * D, D, static_cast<const float*>(c.w.layer(il, "attn_k_norm.weight").dev),
                                             kv.K + size_t(pos0) * KH * D, KH, D, s.rope_dims, pos0, theta_scale, eps, true);
    k_copy_round_h<<<(T * KH * D + 255) / 256, 256, 0, c.stream>>>(vraw, kv.V + size_t(pos0) * KH * D, T * KH * D);
    const size_t smem = size_t(pos0 + T) * 4;
    k_attn_dense<<<T * H, D, smem, c.stream>>>(q, kv.K, kv.V, qfull, o, H, KH, D, pos0, 1.0f / sqrtf(float(D)), H * 2 * D, 2 * D, D);
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
