// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_stream.hpp"

#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/moe_q2.h"
#include "quant/q2_0/q2_0.hpp"

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

constexpr uint32_t kQ2_0 = 42;   // GGML_TYPE_Q2_0

// The routed experts run on moe_q2 (planar Q2_0 read directly) unless FLASHRT_MOE_Q2MMA=0, which
// keeps the conversion to ggml's layout and its MMQ kernels.
bool use_q2mma() {
    static const bool on = [] {
        const char* e = std::getenv("FLASHRT_MOE_Q2MMA");
        return !(e && e[0] == '0');
    }();
    return on;
}
// FLASHRT_MOE_AB64=1: activations quantized per 64 for moe_q2 (moe_q2::run's block64)
bool use_ab64() {
    static const bool on = [] {
        const char* e = std::getenv("FLASHRT_MOE_AB64");
        return e && e[0] == '1';
    }();
    return on;
}

// The arena's planar Q2_0 (codes [rows][nb][16], element j in byte j % 16 at bits 2 * (j / 16);
// fp16 scales [rows][nb] after all codes) to ggml's Q2_0 blocks (fp16 d, then element j in byte
// j / 4 at bits 2 * (j % 4)), for every expert of a layer: gate and up [E][ff][n/64] and down
// [E][n][ff/64], each tensor back to back. Grid: (items / 256, E * 3 matrices); each thread
// converts one 64-value block into shared memory, and the CTA writes its 256 blocks out as words.
constexpr int kConvItems = 256;
__global__ void __launch_bounds__(kConvItems) k_planar_to_ggml(const uint8_t* planar, size_t stride, uint8_t* gate, uint8_t* up,
                                                                uint8_t* down, int n, int ff) {
    __shared__ uint32_t sm[kConvItems * 18 / 4];
    const int e = blockIdx.y / 3, m = blockIdx.y % 3;
    const int64_t items = int64_t(m < 2 ? ff : n) * ((m < 2 ? n : ff) / 64);   // (row, block) pairs of the matrix
    const int64_t i0 = int64_t(blockIdx.x) * kConvItems;
    if (i0 >= items) return;
    const size_t mb = size_t(ff) * (n / 64) * 18;   // one gate/up matrix
    const uint8_t* mat = planar + size_t(e) * stride + size_t(m) * mb;
    uint8_t* dst = (m == 0 ? gate : m == 1 ? up : down) + (size_t(e) * items + i0) * 18;
    const int64_t i = i0 + threadIdx.x;
    const int nitem = int(min(int64_t(kConvItems), items - i0));
    if (threadIdx.x < nitem) {
        const uint4 c = reinterpret_cast<const uint4*>(mat)[i];
        const uint16_t d = reinterpret_cast<const uint16_t*>(mat + size_t(items) * 16)[i];
        const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
        uint32_t out[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) {   // ggml word v: elements 16v .. 16v + 15 = field v of every planar byte
            uint32_t o = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t x = (cw[w] >> (2 * v)) & 0x03030303u;   // byte b: element 4w + b
                x = (x | (x >> 6)) & 0x000F000Fu;
                x = (x | (x >> 12)) & 0xFFu;
                o |= x << (8 * w);
            }
            out[v] = o;
        }
        uint16_t* s16 = reinterpret_cast<uint16_t*>(sm) + size_t(threadIdx.x) * 9;
        s16[0] = d;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            s16[1 + 2 * w] = uint16_t(out[w] & 0xffff);
            s16[2 + 2 * w] = uint16_t(out[w] >> 16);
        }
    }
    __syncthreads();
    const int words = nitem * 18 / 4;   // nitem is even, so whole words
    uint32_t* d32 = reinterpret_cast<uint32_t*>(dst);
    for (int k = threadIdx.x; k < words; k += blockDim.x) d32[k] = sm[k];
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

size_t expert_stream_bytes_for(const Spec& s, const ExpertArena& arena, int max_tokens) {
    const size_t E = s.n_expert, K = s.top_k, n = s.d_model, ff = s.d_ff_expert, ffs = s.d_ff_shared, T = size_t(max_tokens);
    const size_t gu = E * ff * (n / 64) * 18, dn = E * n * (ff / 64) * 18;
    const size_t common[] = {E * arena.stride, E * arena.stride, T * E * 4, T * K * 4, T * K * 4, T * K * n * 4,
                             T * ffs * 4,      T * ffs * 4,      T * n * 4, T * 4};
    size_t tot = 0;
    for (size_t p : common) tot += p + 256;
    if (use_q2mma()) return tot + moe_q2::workspace_bytes(int(T), int(K), int(n), int(ff), int(E)) + 256;
    const size_t mmq[] = {gu + gemv::kWeightTailPad, gu + gemv::kWeightTailPad, dn + gemv::kWeightTailPad, T * K * ff * 4,
                          T * K * ff * 4, gemm::workspace_bytes(int64_t(std::max(n, ff)), int64_t(T * K), false)};
    for (size_t p : mmq) tot += p + 256;
    return tot;
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
    es->yd = dalloc<float>(T * K * n, tot);
    es->sg = dalloc<float>(T * ffs, tot);
    es->su = dalloc<float>(T * ffs, tot);
    es->sh = dalloc<float>(T * n, tot);
    es->gate = dalloc<float>(T, tot);
    if (use_q2mma()) {
        es->ws_bytes = moe_q2::workspace_bytes(max_tokens, K, n, ff, E);
        es->ws = dalloc<uint8_t>(es->ws_bytes, tot);
    } else {
        const size_t gu = size_t(E) * ff * (n / 64) * 18, dn = size_t(E) * n * (ff / 64) * 18;
        es->g_gate = dalloc<uint8_t>(gu + gemv::kWeightTailPad, tot);
        es->g_up = dalloc<uint8_t>(gu + gemv::kWeightTailPad, tot);
        es->g_down = dalloc<uint8_t>(dn + gemv::kWeightTailPad, tot);
        ck(cudaMemset(es->g_gate + gu, 0, gemv::kWeightTailPad), "memset");
        ck(cudaMemset(es->g_up + gu, 0, gemv::kWeightTailPad), "memset");
        ck(cudaMemset(es->g_down + dn, 0, gemv::kWeightTailPad), "memset");
        es->hg = dalloc<float>(T * K * ff, tot);
        es->hu = dalloc<float>(T * K * ff, tot);
        es->ws_bytes = gemm::workspace_bytes(std::max(n, ff), int64_t(T) * K, false);   // Q8_1 activations only
        es->ws = dalloc<uint8_t>(es->ws_bytes, tot);
    }
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
    const int b = il % 2;
    expert_stream_prefetch(&es, il);
    ck(cudaStreamWaitEvent(c.stream, es.uploaded[b], 0), "wait experts");
    if (use_q2mma()) {
        // routing, then the experts straight from the planar slice (released after them), and the
        // next layer's copy into the other buffer meanwhile
        if (il + 1 < s.n_layer) expert_stream_prefetch(&es, il + 1);
        linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, es.logits, T);
        moe_route_topk(c.stream, es.logits, T, E, K, es.ids, es.wts, counts ? counts + size_t(il) * E : nullptr);
        moe_q2::run(es.planar[b], es.arena->stride, E, n, ff, x, es.ids, T, K, es.yd, es.ws, es.ws_bytes, c.stream, use_ab64());
        ck(cudaEventRecord(es.released[b], c.stream), "cudaEventRecord");
    } else {
        // convert the slice to ggml's layout (the slice is free after), start the next layer's
        // copy, routing, then MMQ gate and up, SwiGLU, MMQ down
        const int64_t items = std::max(int64_t(ff) * (n / 64), int64_t(n) * (ff / 64));
        k_planar_to_ggml<<<dim3(unsigned((items + kConvItems - 1) / kConvItems), 3 * E), kConvItems, 0, c.stream>>>(
            es.planar[b], es.arena->stride, es.g_gate, es.g_up, es.g_down, n, ff);
        ck(cudaGetLastError(), "planar to ggml");
        ck(cudaEventRecord(es.released[b], c.stream), "cudaEventRecord");
        if (il + 1 < s.n_layer) expert_stream_prefetch(&es, il + 1);
        linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, es.logits, T);
        moe_route_topk(c.stream, es.logits, T, E, K, es.ids, es.wts, counts ? counts + size_t(il) * E : nullptr);
        const int64_t gu_stride = int64_t(ff) * (n / 64) * 18, d_stride = int64_t(n) * (ff / 64) * 18;
        const gemm::MoePlan pgu = gemm::moe_prepare(kQ2_0, E, x, false, es.ids, T, K, n, es.ws, es.ws_bytes, c.stream);
        gemm::moe_run(pgu, es.g_gate, gu_stride, es.hg, ff, c.stream);
        gemm::moe_run(pgu, es.g_up, gu_stride, es.hu, ff, c.stream);
        const size_t nh = size_t(T) * K * ff;
        k_swiglu_rows<<<unsigned((nh + 255) / 256), 256, 0, c.stream>>>(es.hg, es.hu, nh);
        const gemm::MoePlan pd = gemm::moe_prepare(kQ2_0, E, es.hg, true, es.ids, T, K, ff, es.ws, es.ws_bytes, c.stream);
        gemm::moe_run(pd, es.g_down, d_stride, es.yd, n, c.stream);
    }
    // shared expert, and the sum
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
