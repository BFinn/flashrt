// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_stream.hpp"

#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

const uint8_t* arena_register(const ExpertArena& arena);   // moe_fast.cu

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr uint32_t kQ2_0 = 42;   // GGML_TYPE_Q2_0

// One 64-value block per thread: the arena's planar Q2_0 (codes [rows][nb][16], element j in byte
// j % 16 at bits 2 * (j / 16); fp16 scales [rows][nb] after all codes) to ggml's Q2_0 blocks
// (fp16 d, then element j in byte j / 4 at bits 2 * (j % 4)), for every expert of a layer:
// gate and up [E][ff][n/64] and down [E][n][ff/64], each tensor back to back.
__global__ void k_planar_to_ggml(const uint8_t* planar, size_t stride, uint8_t* gate, uint8_t* up, uint8_t* down, int E, int n, int ff) {
    const int nb = n / 64, nbd = ff / 64;
    const int64_t per_gu = int64_t(ff) * nb, per_d = int64_t(n) * nbd, per_e = 2 * per_gu + per_d;
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= per_e * E) return;
    const int e = int(g / per_e);
    int64_t i = g % per_e;
    const uint8_t* blob = planar + size_t(e) * stride;
    const size_t mb = size_t(ff) * nb * 18;   // one gate/up matrix, planar
    const uint8_t* mat;
    int64_t rows_nb;   // rows * blocks of this matrix
    uint8_t* dst;
    if (i < per_gu) {
        mat = blob;
        rows_nb = per_gu;
        dst = gate + (size_t(e) * per_gu + i) * 18;
    } else if (i < 2 * per_gu) {
        i -= per_gu;
        mat = blob + mb;
        rows_nb = per_gu;
        dst = up + (size_t(e) * per_gu + i) * 18;
    } else {
        i -= 2 * per_gu;
        mat = blob + 2 * mb;
        rows_nb = per_d;
        dst = down + (size_t(e) * per_d + i) * 18;
    }
    const uint4 c = reinterpret_cast<const uint4*>(mat)[i];   // (row, block) i: 16 code bytes
    const uint16_t d = reinterpret_cast<const uint16_t*>(mat + size_t(rows_nb) * 16)[i];
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
    uint32_t out[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        const int src = j % 16;
        const uint32_t code = (cw[src / 4] >> (8 * (src % 4) + 2 * (j / 16))) & 3u;
        out[j / 16] |= code << (2 * (j % 16));   // byte j / 4 at bits 2 * (j % 4): word j / 16, bit 2 * (j % 16)
    }
    uint16_t* d16 = reinterpret_cast<uint16_t*>(dst);   // 18-byte blocks: 2-byte aligned
    d16[0] = d;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        d16[1 + 2 * w] = uint16_t(out[w] & 0xffff);
        d16[2 + 2 * w] = uint16_t(out[w] >> 16);
    }
}

__global__ void k_swiglu_rows(float* g, const float* u, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

// out[t][i] = sum_k wts[t][k] * yd[t][k][i] + sh[t][i] * sigmoid(gate[t])
__global__ void k_stream_combine(float* out, const float* yd, const float* wts, const float* sh, const float* gate, int n, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += wts[size_t(t) * K + k] * yd[(size_t(t) * K + k) * n + i];
    out[size_t(t) * n + i] = acc + sh[size_t(t) * n + i] / (1.0f + __expf(-gate[t]));
}

template <typename T>
T* dalloc(size_t n, size_t& total) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc expert stream");
    total += n * sizeof(T) + 256;
    return p;
}

}  // namespace

struct ExpertStream {
    const Spec* s = nullptr;
    const ExpertArena* arena = nullptr;
    size_t slice_bytes = 0;
    uint8_t* planar[2] = {nullptr, nullptr};
    int planar_layer[2] = {-1, -1};
    uint8_t *g_gate = nullptr, *g_up = nullptr, *g_down = nullptr;
    cudaStream_t copy = nullptr;
    cudaEvent_t uploaded[2] = {nullptr, nullptr}, released[2] = {nullptr, nullptr};
    int max_tokens = 0;
    float *logits = nullptr, *wts = nullptr, *hg = nullptr, *hu = nullptr, *yd = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *gate = nullptr;
    int32_t* ids = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    size_t total = 0;
};

ExpertStream* create_expert_stream(const Spec& s, const ExpertArena& arena, int max_tokens) {
    auto* es = new ExpertStream;
    es->s = &s;
    es->arena = &arena;
    es->max_tokens = max_tokens;
    arena_register(arena);
    const int E = s.n_expert, K = s.top_k, n = s.d_model, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (n % 64 || ff % 64) throw std::runtime_error("expert stream: unsupported expert shape");
    es->slice_bytes = size_t(E) * arena.stride;
    size_t& tot = es->total;
    for (int b = 0; b < 2; ++b) es->planar[b] = dalloc<uint8_t>(es->slice_bytes, tot);
    const size_t gu = size_t(E) * ff * (n / 64) * 18, dn = size_t(E) * n * (ff / 64) * 18;
    es->g_gate = dalloc<uint8_t>(gu + gemv::kWeightTailPad, tot);
    es->g_up = dalloc<uint8_t>(gu + gemv::kWeightTailPad, tot);
    es->g_down = dalloc<uint8_t>(dn + gemv::kWeightTailPad, tot);
    ck(cudaMemset(es->g_gate + gu, 0, gemv::kWeightTailPad), "memset");
    ck(cudaMemset(es->g_up + gu, 0, gemv::kWeightTailPad), "memset");
    ck(cudaMemset(es->g_down + dn, 0, gemv::kWeightTailPad), "memset");
    const size_t T = size_t(max_tokens);
    es->logits = dalloc<float>(T * E, tot);
    es->ids = dalloc<int32_t>(T * K, tot);
    es->wts = dalloc<float>(T * K, tot);
    es->hg = dalloc<float>(T * K * ff, tot);
    es->hu = dalloc<float>(T * K * ff, tot);
    es->yd = dalloc<float>(T * K * n, tot);
    es->sg = dalloc<float>(T * ffs, tot);
    es->su = dalloc<float>(T * ffs, tot);
    es->sh = dalloc<float>(T * n, tot);
    es->gate = dalloc<float>(T, tot);
    es->ws_bytes = gemm::workspace_bytes(std::max(n, ff), int64_t(T) * K);
    es->ws = dalloc<uint8_t>(es->ws_bytes, tot);
    ck(cudaStreamCreateWithFlags(&es->copy, cudaStreamNonBlocking), "cudaStreamCreate expert copy");
    for (int b = 0; b < 2; ++b) {
        ck(cudaEventCreateWithFlags(&es->uploaded[b], cudaEventDisableTiming), "cudaEventCreate");
        ck(cudaEventCreateWithFlags(&es->released[b], cudaEventDisableTiming), "cudaEventCreate");
        ck(cudaEventRecord(es->released[b], es->copy), "cudaEventRecord");
    }
    return es;
}

void destroy_expert_stream(ExpertStream* es) {
    if (!es) return;
    if (es->copy) cudaStreamSynchronize(es->copy);
    for (void* p : {static_cast<void*>(es->planar[0]), static_cast<void*>(es->planar[1]), static_cast<void*>(es->g_gate),
                    static_cast<void*>(es->g_up), static_cast<void*>(es->g_down), static_cast<void*>(es->logits),
                    static_cast<void*>(es->ids), static_cast<void*>(es->wts), static_cast<void*>(es->hg), static_cast<void*>(es->hu),
                    static_cast<void*>(es->yd), static_cast<void*>(es->sg), static_cast<void*>(es->su), static_cast<void*>(es->sh),
                    static_cast<void*>(es->gate), es->ws})
        if (p) cudaFree(p);
    for (int b = 0; b < 2; ++b) {
        if (es->uploaded[b]) cudaEventDestroy(es->uploaded[b]);
        if (es->released[b]) cudaEventDestroy(es->released[b]);
    }
    if (es->copy) cudaStreamDestroy(es->copy);
    delete es;
}

size_t expert_stream_bytes(const ExpertStream* es) { return es ? es->total : 0; }

void expert_stream_prefetch(ExpertStream* es, int il) {
    const int b = il % 2;
    if (es->planar_layer[b] == il) return;
    ck(cudaStreamWaitEvent(es->copy, es->released[b], 0), "wait slice buffer");
    ck(cudaMemcpyAsync(es->planar[b], es->arena->blob(il, 0), es->slice_bytes, cudaMemcpyHostToDevice, es->copy), "upload experts");
    ck(cudaEventRecord(es->uploaded[b], es->copy), "cudaEventRecord");
    es->planar_layer[b] = il;
}

void moe_block_stream(const BlockCtx& c, int il, const float* x, int T, ExpertStream& es, float* out, uint32_t* counts) {
    const Spec& s = c.s;
    const int E = s.n_expert, K = s.top_k, n = s.d_model, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (T > es.max_tokens) throw std::runtime_error("moe_block_stream: chunk larger than the stream's buffers");
    // 1. this layer's experts: wait for the copy, convert, and start the next layer's copy
    const int b = il % 2;
    expert_stream_prefetch(&es, il);
    ck(cudaStreamWaitEvent(c.stream, es.uploaded[b], 0), "wait experts");
    const int64_t blocks = int64_t(E) * (2 * int64_t(ff) * (n / 64) + int64_t(n) * (ff / 64));
    k_planar_to_ggml<<<unsigned((blocks + 255) / 256), 256, 0, c.stream>>>(es.planar[b], es.arena->stride, es.g_gate, es.g_up, es.g_down,
                                                                          E, n, ff);
    ck(cudaGetLastError(), "planar to ggml");
    ck(cudaEventRecord(es.released[b], c.stream), "cudaEventRecord");
    if (il + 1 < s.n_layer) expert_stream_prefetch(&es, il + 1);
    // 2. routing
    linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, es.logits, T);
    moe_route_topk(c.stream, es.logits, T, E, K, es.ids, es.wts, counts ? counts + size_t(il) * E : nullptr);
    // 3. routed experts: gate and up, SwiGLU, down
    const int64_t gu_stride = int64_t(ff) * (n / 64) * 18, d_stride = int64_t(n) * (ff / 64) * 18;
    gemm::moe(kQ2_0, es.g_gate, gu_stride, E, x, false, es.ids, T, K, es.hg, n, ff, es.ws, es.ws_bytes, c.stream);
    gemm::moe(kQ2_0, es.g_up, gu_stride, E, x, false, es.ids, T, K, es.hu, n, ff, es.ws, es.ws_bytes, c.stream);
    const size_t nh = size_t(T) * K * ff;
    k_swiglu_rows<<<unsigned((nh + 255) / 256), 256, 0, c.stream>>>(es.hg, es.hu, nh);
    gemm::moe(kQ2_0, es.g_down, d_stride, E, es.hg, true, es.ids, T, K, es.yd, ff, n, es.ws, es.ws_bytes, c.stream);
    // 4. shared expert, and the sum
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, es.sg, T);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, es.su, T);
    const size_t ns = size_t(T) * ffs;
    k_swiglu_rows<<<unsigned((ns + 255) / 256), 256, 0, c.stream>>>(es.sg, es.su, ns);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), es.sg, es.sh, T);
    linear(c, c.w.layer(il, "ffn_gate_inp_shexp.weight"), x, es.gate, T);
    k_stream_combine<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(out, es.yd, es.wts, es.sh, es.gate, n, K);
    ck(cudaGetLastError(), "moe_block_stream");
}

}  // namespace flashrt::qwen4exp
