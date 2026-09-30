// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_stream.hpp"

#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/moe_q2.h"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// FLASHRT_MOE_AB64=1: activations quantized per 64 for moe_q2 (moe_q2::run's block64)
bool use_ab64() {
    static const bool on = [] {
        const char* e = std::getenv("FLASHRT_MOE_AB64");
        return e && e[0] == '1';
    }();
    return on;
}

__global__ void k_swiglu_rows(float* g, const float* u, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

// out[t][i] = sum_k wts[t][k] * yd[t][k][i] + sh[t][i] * sigmoid(gate[t]); yd in BF16 (sw59)
__global__ void k_stream_combine(float* out, const __nv_bfloat16* yd, const float* wts, const float* sh, const float* gate, int n, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += wts[size_t(t) * K + k] * __bfloat162float(yd[(size_t(t) * K + k) * n + i]);
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
    cudaStream_t copy = nullptr;
    cudaEvent_t uploaded[2] = {nullptr, nullptr}, released[2] = {nullptr, nullptr};
    int max_tokens = 0;
    void* yd = nullptr;   // [T * K][n], BF16
    float *logits = nullptr, *wts = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *gate = nullptr;
    int32_t* ids = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    size_t total = 0;
};

size_t expert_stream_bytes_for(const Spec& s, const ExpertArena& arena, int max_tokens) {
    const size_t E = s.n_expert, K = s.top_k, n = s.d_model, ff = s.d_ff_expert, ffs = s.d_ff_shared, T = size_t(max_tokens);
    const size_t common[] = {E * arena.stride, E * arena.stride, T * E * 4, T * K * 4, T * K * 4, T * K * n * 2,
                             T * ffs * 4,      T * ffs * 4,      T * n * 4, T * 4};
    size_t tot = 0;
    for (size_t p : common) tot += p + 256;
    return tot + moe_q2::workspace_bytes(int(T), int(K), int(n), int(ff), int(E)) + 256;
}

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
    const size_t T = size_t(max_tokens);
    es->logits = dalloc<float>(T * E, tot);
    es->ids = dalloc<int32_t>(T * K, tot);
    es->wts = dalloc<float>(T * K, tot);
    es->yd = dalloc<uint8_t>(T * K * n * 2, tot);
    es->sg = dalloc<float>(T * ffs, tot);
    es->su = dalloc<float>(T * ffs, tot);
    es->sh = dalloc<float>(T * n, tot);
    es->gate = dalloc<float>(T, tot);
    es->ws_bytes = moe_q2::workspace_bytes(max_tokens, K, n, ff, E);
    es->ws = dalloc<uint8_t>(es->ws_bytes, tot);
    if (tot != expert_stream_bytes_for(s, arena, max_tokens)) throw std::logic_error("expert_stream_bytes_for is out of date");
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
    for (void* p : {static_cast<void*>(es->planar[0]), static_cast<void*>(es->planar[1]), static_cast<void*>(es->logits),
                    static_cast<void*>(es->ids), static_cast<void*>(es->wts), static_cast<void*>(es->yd), static_cast<void*>(es->sg), static_cast<void*>(es->su), static_cast<void*>(es->sh),
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

namespace {
// the router logits and the shared expert's gate logit: one BF16 conversion of x for both
void route_and_gate(const BlockCtx& c, int il, const float* x, ExpertStream& es, int T) {
    const GpuTensor* ws[2] = {&c.w.layer(il, "ffn_gate_inp.weight"), &c.w.layer(il, "ffn_gate_inp_shexp.weight")};
    float* ys[2] = {es.logits, es.gate};
    linear_shared(c, ws, ys, 2, x, T);
}
}  // namespace

void moe_block_stream(const BlockCtx& c, int il, const float* x, int T, ExpertStream& es, float* out, uint32_t* counts,
                      uint32_t* tail_counts, int tail_from) {
    const Spec& s = c.s;
    const int E = s.n_expert, K = s.top_k, n = s.d_model, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (T > es.max_tokens) throw std::runtime_error("moe_block_stream: chunk larger than the stream's buffers");
    const int b = il % 2;
    expert_stream_prefetch(&es, il);
    ck(cudaStreamWaitEvent(c.stream, es.uploaded[b], 0), "wait experts");
    // routing, then the experts straight from the planar slice (released after them), and the next
    // layer's copy into the other buffer meanwhile
    if (il + 1 < s.n_layer) expert_stream_prefetch(&es, il + 1);
    route_and_gate(c, il, x, es, T);
    moe_route_topk(c.stream, es.logits, T, E, K, es.ids, es.wts, counts ? counts + size_t(il) * E : nullptr,
                   tail_counts ? tail_counts + size_t(il) * E : nullptr, tail_from);
    moe_q2::run(es.planar[b], es.arena->stride, E, n, ff, x, es.ids, T, K, es.yd, es.ws, es.ws_bytes, c.stream, use_ab64(), true);
    ck(cudaEventRecord(es.released[b], c.stream), "cudaEventRecord");
    // shared expert, and the sum
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, es.sg, T);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, es.su, T);
    const size_t ns = size_t(T) * ffs;
    k_swiglu_rows<<<unsigned((ns + 255) / 256), 256, 0, c.stream>>>(es.sg, es.su, ns);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), es.sg, es.sh, T);
    k_stream_combine<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(out, static_cast<const __nv_bfloat16*>(es.yd), es.wts, es.sh,
                                                                    es.gate, n, K);
    ck(cudaGetLastError(), "moe_block_stream");
}

}  // namespace flashrt::qwen4exp
