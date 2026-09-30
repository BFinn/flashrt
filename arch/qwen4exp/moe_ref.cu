// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// Batch routing (top-k over the router logits), and the reference MoE: every routed expert on
// the CPU.
#include "arch/qwen4exp/blocks_common.cuh"

#include "quant/q2_0/moe_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace flashrt::qwen4exp {

namespace {
// One warp per token (sw81): the logits in registers (element 32 i + lane), max and sum by
// shuffles, then K rounds of warp argmax (higher p first, ties to the lower index). It replaced a
// block-per-token kernel whose softmax summed in another order (sw81, sw94).
template <int V>
__global__ void __launch_bounds__(256) k_route_topk_w(const float* logits, int T, int E, int K, int32_t* ids, float* wts, uint32_t* counts,
                                                      uint32_t* tail, int tail_from) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float* l = logits + size_t(t) * E;
    float v[V];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const int e = 32 * i + lane;
        v[i] = e < E ? l[e] : -INFINITY;
        m = fmaxf(m, v[i]);
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] = 32 * i + lane < E ? __expf(v[i] - m) : 0.0f;
        sum += v[i];
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
#pragma unroll
    for (int i = 0; i < V; ++i) v[i] = 32 * i + lane < E ? v[i] / sum : -1.0f;
    float ws = 0.0f, selp = 0.0f;
    for (int k = 0; k < K; ++k) {
        float bv = -2.0f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < V; ++i)
            if (v[i] > bv) {   // ascending index within a lane: the first maximum wins ties
                bv = v[i];
                bi = 32 * i + lane;
            }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(~0u, bv, o);
            const int oi = __shfl_xor_sync(~0u, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if ((bi & 31) == lane)
#pragma unroll
            for (int i = 0; i < V; ++i)
                if (32 * i + lane == bi) v[i] = -1.0f;
        if (lane == 0) {
            ids[size_t(t) * K + k] = bi;
            if (counts) atomicAdd(counts + bi, 1u);
            if (tail && t >= tail_from) atomicAdd(tail + bi, 1u);
        }
        if (lane == k) selp = bv;
        ws += bv;
    }
    ws = fmaxf(ws, 6.103515625e-5f);
    if (lane < K) wts[size_t(t) * K + lane] = selp / ws;
}
}  // namespace

void moe_route_topk(cudaStream_t stream, const float* logits, int T, int E, int k, int32_t* ids, float* wts, uint32_t* counts,
                    uint32_t* tail_counts, int tail_from) {
    if (E > 1024 || k > 32) throw std::runtime_error("moe_route_topk: unsupported shape");
    if (E <= 512) k_route_topk_w<16><<<(T + 7) / 8, 256, 0, stream>>>(logits, T, E, k, ids, wts, counts, tail_counts, tail_from);
    else k_route_topk_w<32><<<(T + 7) / 8, 256, 0, stream>>>(logits, T, E, k, ids, wts, counts, tail_counts, tail_from);
    ck(cudaGetLastError(), "moe_route_topk");
}

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
            if (h.tail_counts && t >= h.tail_from) ++(*h.tail_counts)[size_t(il) * E + idx[k]];
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
