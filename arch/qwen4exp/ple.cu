// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// Per-layer embeddings (PLE): n-gram rows read from the SSD, the gate and the causal conv.
#include "arch/qwen4exp/blocks_common.cuh"

#include "kernels/cuda/ggml_gemv.h"

#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

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

PleWindow alloc_ple_window(const Spec& s, const Ple& p, int max_tokens) {
    PleWindow w;
    w.max_tokens = max_tokens;
    ck(cudaMalloc(&w.hist_old, size_t(p.ngram) * 3 * size_t(s.hc_count) * s.d_model * 4 + 4), "cudaMalloc ple backup");
    ck(cudaMalloc(&w.rows, size_t(max_tokens) * s.hc_count * s.d_model * 4), "cudaMalloc ple window");
    return w;
}

void free_ple_window(PleWindow& w) {
    if (w.hist_old) cudaFree(w.hist_old);
    if (w.rows) cudaFree(w.rows);
    w = PleWindow{};
}

void ple_rewind(const BlockCtx& c, int il, const Ple& p, PleState& st, const PleWindow& win, int T, int n) {
    if (n >= T) return;
    const Spec& s = c.s;
    const int C = s.hc_count * s.d_model, K = int(c.w.layer(il, "ple_conv1d.weight").dims.at(0)), H = (K - 1) * p.ngram;
    if (H == 0) return;
    hist_rewind(c.stream, st.hist, win.hist_old, win.rows, H, C, n);
    ck(cudaGetLastError(), "ple_rewind");
}

void ple_block(const BlockCtx& c, int il, const Ple& p, const float* emb, float* x, int T, PleState& st, PleWindow* win) {
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
    if (win) {   // the conv inputs go to the window, and the history is backed up
        if (T > win->max_tokens) throw std::runtime_error("ple_block: window too long");
        normed = win->rows;
        ck(cudaMemcpyAsync(win->hist_old, st.hist, size_t((K - 1) * p.ngram) * C * 4, cudaMemcpyDeviceToDevice, c.stream), "ple backup");
    }
    linear(c, c.w.layer(il, "ple_key.weight"), emb, key, T);
    linear(c, c.w.layer(il, "ple_value.weight"), emb, value, T);
    rms_norm_rows(c, key, static_cast<const float*>(c.w.layer(il, "ple_norm_key.weight").dev), key, n, hc, T * hc);
    rms_norm_rows(c, x, static_cast<const float*>(c.w.layer(il, "ple_norm_query.weight").dev), qn, n, hc, T * hc);
    k_ple_gate<<<T * hc, 256, 0, c.stream>>>(key, qn, gate, n);
    k_ple_gated<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(value, gate, gated, n, hc, T);
    rms_norm_rows(c, gated, static_cast<const float*>(c.w.layer(il, "ple_norm_conv.weight").dev), normed, n, hc, T * hc);
    k_ple_conv<<<(C + 127) / 128, 128, 0, c.stream>>>(x, gated, normed, st.hist, static_cast<const __half*>(c.w.layer(il, "ple_conv1d.weight").dev),
                                                     C, T, K, p.ngram);
    ck(cudaGetLastError(), "ple_block");
}

}  // namespace flashrt::qwen4exp
