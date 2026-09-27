// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/moe_cpu.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {

constexpr int kMaxK = 16;   // top-k upper bound for the fixed launch shapes

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// planar repacked matrix (codes [rows][nb][16], scales [rows][nb]) -> ggml Q2_0 blocks
// {fp16 d, qs[16]}: qs[k] holds elements 4k..4k+3; planar element j is in byte j%16 at bit 2*(j/16)
__global__ void k_unrepack(const uint8_t* src, uint8_t* dst, size_t nblk) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint8_t* p = src + i * 16;
    const uint16_t d = reinterpret_cast<const uint16_t*>(src + nblk * 16)[i];
    uint8_t out[18];
    out[0] = uint8_t(d & 0xff);
    out[1] = uint8_t(d >> 8);
    for (int k = 0; k < 16; ++k) {
        uint8_t v = 0;
        for (int e = 0; e < 4; ++e) {
            const int j = 4 * k + e;
            v |= uint8_t(((p[j % 16] >> (2 * (j / 16))) & 3) << (2 * e));
        }
        out[2 + k] = v;
    }
    for (int b = 0; b < 18; ++b) dst[i * 18 + b] = out[b];
}

// One block of E threads (E <= 1024): softmax over the router logits, top-k by probability,
// weights renormalised (sum clamped at 6.1e-5, as llama.cpp). Hits get their slot; the hit list
// is padded to k with a real slot (or slot 0) at weight 0 so the grouped launches have a fixed
// shape. route_host gets [n_hits, n_miss, miss experts[k], miss weights[k] as float bits].
__global__ void k_route(const float* logits, const int32_t* table, int E, int K, int32_t* hit_slot, float* hit_w,
                        int32_t* route_dev) {
    __shared__ float p[1024];
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    __shared__ int sel[kMaxK];
    __shared__ float selp[kMaxK];
    const int e = threadIdx.x;
    const float lg = e < E ? logits[e] : -INFINITY;
    // max
    float m = lg;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((e & 31) == 0) red_v[e >> 5] = m;
    __syncthreads();
    if (e < 32) {
        float v = e < (blockDim.x + 31) / 32 ? red_v[e] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (e == 0) red_v[0] = v;
    }
    __syncthreads();
    m = red_v[0];
    __syncthreads();
    const float ex = e < E ? __expf(lg - m) : 0.0f;
    float sum = ex;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((e & 31) == 0) red_v[e >> 5] = sum;
    __syncthreads();
    if (e < 32) {
        float v = e < (blockDim.x + 31) / 32 ? red_v[e] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (e == 0) red_v[0] = v;
    }
    __syncthreads();
    const float total = red_v[0];
    __syncthreads();
    p[e] = e < E ? ex / total : -1.0f;
    __syncthreads();
    // top-k: k rounds of block argmax (lowest index wins ties)
    for (int k = 0; k < K; ++k) {
        float v = p[e];
        int idx = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffff, v, o);
            const int i2 = __shfl_xor_sync(0xffffffff, idx, o);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        if ((e & 31) == 0) { red_v[e >> 5] = v; red_i[e >> 5] = idx; }
        __syncthreads();
        if (e == 0) {
            float bv = red_v[0];
            int bi = red_i[0];
            for (int w = 1; w < (blockDim.x + 31) / 32; ++w)
                if (red_v[w] > bv || (red_v[w] == bv && red_i[w] < bi)) { bv = red_v[w]; bi = red_i[w]; }
            sel[k] = bi;
            selp[k] = bv;
            p[bi] = -1.0f;
        }
        __syncthreads();
    }
    if (e == 0) {
        float ws = 0.0f;
        for (int k = 0; k < K; ++k) ws += selp[k];
        ws = fmaxf(ws, 6.103515625e-5f);
        int nh = 0, nm = 0;
        int pad = 0;
        for (int k = 0; k < K; ++k) {
            const int slot = table[sel[k]];
            const float w = selp[k] / ws;
            if (slot >= 0) {
                hit_slot[nh] = slot;
                hit_w[nh] = w;
                pad = slot;
                ++nh;
            } else {
                route_dev[2 + nm] = sel[k];
                reinterpret_cast<float*>(route_dev)[2 + K + nm] = w;
                ++nm;
            }
        }
        for (int k = nh; k < K; ++k) {
            hit_slot[k] = pad;
            hit_w[k] = 0.0f;
        }
        route_dev[0] = nh;
        route_dev[1] = nm;
    }
}

// out = sum_k hit_w[k] * yh[k] + cpu + shexp * sigmoid(gate)
__global__ void k_moe_combine(float* out, const float* yh, const float* hit_w, int K, const float* cpu, const float* shexp,
                              const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = cpu[i];
    for (int k = 0; k < K; ++k) acc += hit_w[k] * yh[size_t(k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[0]));
    out[i] = acc + shexp[i] * g;
}

__global__ void k_swiglu_1(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

}  // namespace

ExpertCache alloc_expert_cache(const Spec& s, int n_slots) {
    ExpertCache c;
    c.n_slots = n_slots;
    c.slot_bytes = q2_0::expert_bytes({s.d_model, s.d_ff_expert});
    ck(cudaMalloc(&c.slots, size_t(n_slots) * c.slot_bytes + gemv::kWeightTailPad), "cudaMalloc expert cache");
    ck(cudaMemset(c.slots, 0, size_t(n_slots) * c.slot_bytes + gemv::kWeightTailPad), "memset expert cache");
    const size_t tn = size_t(s.n_layer) * s.n_expert;
    ck(cudaMalloc(&c.table_dev, tn * 4), "cudaMalloc slot table");
    c.table.assign(tn, -1);
    ck(cudaMemcpy(c.table_dev, c.table.data(), tn * 4, cudaMemcpyHostToDevice), "slot table");
    c.owner.assign(n_slots, -1);
    ck(cudaMalloc(&c.staging, c.slot_bytes), "cudaMalloc staging");
    return c;
}

void free_expert_cache(ExpertCache& c) {
    if (c.slots) cudaFree(c.slots);
    if (c.table_dev) cudaFree(c.table_dev);
    if (c.staging) cudaFree(c.staging);
    c = ExpertCache{};
}

void expert_cache_fill(const Spec& s, ExpertCache& cache, const ExpertArena& arena,
                       const std::vector<std::pair<int, int>>& order, cudaStream_t stream) {
    const q2_0::ExpertShape es{s.d_model, s.d_ff_expert};
    const size_t mats[3] = {q2_0::mat_bytes(es.d_ff, es.d_model), q2_0::mat_bytes(es.d_ff, es.d_model),
                            q2_0::mat_bytes(es.d_model, es.d_ff)};
    int slot = 0;
    for (const auto& [l, e] : order) {
        while (slot < cache.n_slots && cache.owner[slot] >= 0) ++slot;
        if (slot >= cache.n_slots) break;
        if (cache.table[size_t(l) * s.n_expert + e] >= 0) continue;
        ck(cudaMemcpyAsync(cache.staging, arena.blob(l, e), cache.slot_bytes, cudaMemcpyHostToDevice, stream), "upload expert");
        size_t off = 0;
        for (size_t m : mats) {
            const size_t nblk = m / 18;
            k_unrepack<<<unsigned((nblk + 255) / 256), 256, 0, stream>>>(cache.staging + off,
                                                                      cache.slots + size_t(slot) * cache.slot_bytes + off, nblk);
            off += m;
        }
        cache.owner[slot] = l * s.n_expert + e;
        cache.table[size_t(l) * s.n_expert + e] = slot;
        ++slot;
    }
    ck(cudaMemcpyAsync(cache.table_dev, cache.table.data(), cache.table.size() * 4, cudaMemcpyHostToDevice, stream), "slot table");
    ck(cudaStreamSynchronize(stream), "expert_cache_fill");
}

MoeFastHost alloc_moe_fast_host(const Spec& s) {
    MoeFastHost h;
    ck(cudaHostAlloc(&h.route_host, size_t(2 + 2 * kMaxK) * 4, cudaHostAllocDefault), "cudaHostAlloc route");
    ck(cudaHostAlloc(&h.x_host, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc x");
    ck(cudaHostAlloc(&h.cpu_out, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc cpu out");
    ck(cudaEventCreateWithFlags(&h.routed, cudaEventDisableTiming), "cudaEventCreate");
    return h;
}

void free_moe_fast_host(MoeFastHost& h) {
    if (h.route_host) cudaFreeHost(h.route_host);
    if (h.x_host) cudaFreeHost(h.x_host);
    if (h.cpu_out) cudaFreeHost(h.cpu_out);
    if (h.routed) cudaEventDestroy(h.routed);
    h = MoeFastHost{};
}

void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out) {
    const Spec& s = c.s;
    const int n = s.d_model, E = s.n_expert, K = s.top_k, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (K > kMaxK || E > 1024) throw std::runtime_error("moe_block_fast: unsupported routing shape");
    // device scratch layout
    float* logits = c.scratch.f32;                 // [E]
    float* sg = logits + E;                        // [ffs]
    float* su = sg + ffs;                          // [ffs]
    float* sh = su + ffs;                          // [n]
    float* gate = sh + n;                          // [1] (padded to 4)
    float* hbuf = gate + 4;                        // [K][ff]
    float* yh = hbuf + size_t(K) * ff;             // [K][n]
    float* cpu_dev = yh + size_t(K) * n;           // [n]
    float* hit_w = cpu_dev + n;                    // [K]
    int32_t* hit_slot = reinterpret_cast<int32_t*>(hit_w + kMaxK);     // [K]
    int32_t* route_dev = hit_slot + kMaxK;                              // [2 + 2K]
    uint8_t* xq = static_cast<uint8_t*>(c.scratch.q8);                  // Q8_1 of x
    uint8_t* hq = xq + gemv::q8_1_bytes(n, 1);                          // Q8_1 of the K hidden rows
    if (size_t(reinterpret_cast<float*>(route_dev + 2 + 2 * kMaxK) - c.scratch.f32) > c.scratch.f32_elems ||
        gemv::q8_1_bytes(n, 1) + gemv::q8_1_bytes(ff, K) > c.scratch.q8_bytes)
        throw std::runtime_error("moe_block_fast: scratch too small");

    // 1. routing, and the input for the CPU
    linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, logits, 1);
    k_route<<<1, ((E + 31) / 32) * 32, 0, c.stream>>>(logits, cache.table_dev + size_t(il) * E, E, K, hit_slot, hit_w, route_dev);
    ck(cudaMemcpyAsync(h.route_host, route_dev, size_t(2 + 2 * K) * 4, cudaMemcpyDeviceToHost, c.stream), "route to host");
    ck(cudaMemcpyAsync(h.x_host, x, size_t(n) * 4, cudaMemcpyDeviceToHost, c.stream), "x to host");
    ck(cudaEventRecord(h.routed, c.stream), "event");

    // 2. GPU: cache hits (fused gate+up, then down) and the shared expert
    const uint8_t* slots = cache.slots;
    const size_t gu = q2_0::mat_bytes(ff, n);
    gemv::quantize_q8_1(x, n, 1, xq, c.stream);
    gemv::moe_q2_0(slots + gu, slots, xq, hit_slot, hbuf, K, n, ff, int64_t(cache.slot_bytes), false, c.stream);
    gemv::quantize_q8_1(hbuf, ff, K, hq, c.stream);
    gemv::moe_q2_0(slots + 2 * gu, nullptr, hq, hit_slot, yh, K, ff, n, int64_t(cache.slot_bytes), true, c.stream);
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, sg, 1);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, su, 1);
    k_swiglu_1<<<(ffs + 255) / 256, 256, 0, c.stream>>>(sg, su, ffs);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), sg, sh, 1);
    linear(c, c.w.layer(il, "ffn_gate_inp_shexp.weight"), x, gate, 1);

    // 3. CPU: the misses, while the GPU works
    ck(cudaEventSynchronize(h.routed), "wait routing");
    const int nh = h.route_host[0], nm = h.route_host[1];
    h.hits += nh;
    h.misses += nm;
    if (nm > 0) {
        const q2_0::ExpertShape es{n, ff};
        const size_t qb = q2_0::q8_bytes(n);
        h.act_mem.resize(qb + 64);
        q2_0::Q8Act act = q2_0::q8_view(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(h.act_mem.data()) + 63) & ~uintptr_t(63)), n);
        q2_0::quantize_q8(h.x_host, act);
        std::vector<q2_0::Miss> miss(nm);
        const float* mw = reinterpret_cast<const float*>(h.route_host + 2 + K);
        for (int i = 0; i < nm; ++i) miss[i] = q2_0::Miss{h.arena->blob(il, h.route_host[2 + i]), 1, {0}, {mw[i]}};
        h.scratch.resize(q2_0::moe_cpu_scratch_bytes(es, nm, h.pool->size()));
        q2_0::moe_cpu(*h.pool, es, miss.data(), nm, &act, 1, h.cpu_out, n, h.scratch.data());
    } else {
        std::memset(h.cpu_out, 0, size_t(n) * 4);
    }
    ck(cudaMemcpyAsync(cpu_dev, h.cpu_out, size_t(n) * 4, cudaMemcpyHostToDevice, c.stream), "cpu result to device");
    k_moe_combine<<<(n + 255) / 256, 256, 0, c.stream>>>(out, yh, hit_w, K, cpu_dev, sh, gate, n);
    ck(cudaGetLastError(), "moe_block_fast");
}

}  // namespace flashrt::qwen4exp
