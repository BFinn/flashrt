// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/moe_cpu.hpp"

#include "core/platform.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace flashrt::qwen4exp {

namespace {

constexpr int kMaxK = 16;   // top-k upper bound for the fixed launch shapes
constexpr int kRouteSel = 2 + 2 * kMaxK;   // route record: offset of the selected experts
constexpr int kRouteInts = kRouteSel + kMaxK;

// mailbox layout (one per layer, doorbell mode)
constexpr size_t kMbRouted = 0;     // uint32, GPU -> host: routing and x are in place for token seq
constexpr size_t kMbDone = 64;      // uint32, host -> GPU: out is in place for token seq
constexpr size_t kMbErr = 96;       // uint32, GPU -> host: the combine gave up waiting for token seq
constexpr size_t kMbRoute = 128;    // int32 [2 + 2K], as route_dev
constexpr size_t kMbX = 512;        // float [d_model]
size_t mb_out_off(int n) { return (kMbX + size_t(n) * 4 + 63) & ~size_t(63); }   // float [d_model]

__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Exclusive prefix count of flag over the block (blockDim.x a multiple of 32); *total = all.
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
        if (lane < nw) warp_tot[lane] = v;
    }
    __syncthreads();
    const int before = warp > 0 ? warp_tot[warp - 1] : 0;
    *total = warp_tot[nw - 1];
    __syncthreads();
    return before + in_warp;
}

// ---- cache-hit experts: Q2_0 in the arena's planar layout (quant/q2_0/q2_0.hpp: per matrix,
// codes [rows][nb][16] with element j of a block in byte j%16 at bits 2*(j/16), then fp16
// scales [rows][nb]), int8 activations per 64-element block (scale amax/127), dp4a.
//
// An int8 activation block is stored as 16 words in "planar word order": word w = 4p + ic holds
// elements 16p + 4ic .. +3, so (code word ic >> 2p) & 0x03030303 lines up with it. Words are
// laid out [w][nb] (block index fastest) so lanes on consecutive blocks read consecutive words.
constexpr int kQB = 64;

// Quantizes one 64-element block (lanes hold elements lane and lane + 32) into the planar word
// layout at words[w * ld + b]; returns (scale, sum of codes) via the pointers, from lane 0.
__device__ __forceinline__ void quant_block64(float v0, float v1, int8_t* words_bytes, int ld, int b, float* scale, int* sum) {
    const int lane = threadIdx.x & 31;
    float am = fmaxf(fabsf(v0), fabsf(v1));
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    const float sc = am / 127.0f, inv = am > 0.0f ? 127.0f / am : 0.0f;
    const int q0 = __float2int_rn(v0 * inv), q1 = __float2int_rn(v1 * inv);
    // element j -> word (j / 16) * 4 + (j % 16) / 4, byte j % 4
    const int j0 = lane, j1 = lane + 32;
    words_bytes[((j0 / 16 * 4 + (j0 % 16) / 4) * ld + b) * 4 + j0 % 4] = int8_t(q0);
    words_bytes[((j1 / 16 * 4 + (j1 % 16) / 4) * ld + b) * 4 + j1 % 4] = int8_t(q1);
    int sm = q0 + q1;
    for (int o = 16; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffff, sm, o);
    if (lane == 0) {
        *scale = sc;
        *sum = sm;
    }
}

// sum over a 64-element block of code * activation, codes c in 0..3 (the -1 offset is applied
// by the caller through the activation sum)
__device__ __forceinline__ int dot_block64(uint4 c, const uint32_t* xw, int ld, int b) {
    const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
    int acc = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int ic = 0; ic < 4; ++ic) acc = __dp4a(int((cw[ic] >> (2 * p)) & 0x03030303u), int(xw[(p * 4 + ic) * ld + b]), acc);
    return acc;
}

// Gate and up of cache-hit expert k (blockIdx.y < *hit_n) for 64 rows (blockIdx.x), SwiGLU, and
// the hidden block quantized for the down kernel: hq words [k][16][ff/64], hscale/hsum [k][ff/64].
// 512 threads: 16 warps x 4 rows, 8 lanes per row. Needs n % 512 == 0 and ff % 64 == 0.
__global__ void k_moe_gate_up(const uint8_t* slots, size_t slot_bytes, const int32_t* hit_slot, const int32_t* hit_n,
                              const float* x, int n, int ff, uint32_t* hq, float* hscale, int32_t* hsum) {
    const int k = blockIdx.y;
    if (k >= *hit_n) return;
    extern __shared__ __align__(16) uint32_t xw[];   // [16][nb] words, then scale [nb], sum [nb]
    __shared__ float hrow[kQB];
    const int nb = n / kQB, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* xscale = reinterpret_cast<float*>(xw + 16 * nb);
    int* xsum = reinterpret_cast<int*>(xscale + nb);
    for (int b = warp; b < nb; b += blockDim.x >> 5)
        quant_block64(x[b * kQB + lane], x[b * kQB + 32 + lane], reinterpret_cast<int8_t*>(xw), nb, b, xscale + b, xsum + b);
    __syncthreads();
    const int rsub = lane >> 3, l8 = lane & 7, rl = warp * 4 + rsub, r = blockIdx.x * kQB + rl;
    const uint8_t* base = slots + size_t(hit_slot[k]) * slot_bytes;
    const size_t mb = size_t(ff) * nb * 18;   // one gate/up matrix
    const uint4* cg = reinterpret_cast<const uint4*>(base) + size_t(r) * nb;
    const __half* sg = reinterpret_cast<const __half*>(base + size_t(ff) * nb * 16) + size_t(r) * nb;
    const uint4* cu = reinterpret_cast<const uint4*>(base + mb) + size_t(r) * nb;
    const __half* su = reinterpret_cast<const __half*>(base + mb + size_t(ff) * nb * 16) + size_t(r) * nb;
    float ag = 0.0f, au = 0.0f;
    for (int b = l8; b < nb; b += 8) {
        const uint4 wg = cg[b], wu = cu[b];
        const float xs = xscale[b];
        const int xm = xsum[b];
        ag += __half2float(sg[b]) * xs * float(dot_block64(wg, xw, nb, b) - xm);
        au += __half2float(su[b]) * xs * float(dot_block64(wu, xw, nb, b) - xm);
    }
    for (int o = 4; o > 0; o >>= 1) {
        ag += __shfl_xor_sync(0xffffffff, ag, o);
        au += __shfl_xor_sync(0xffffffff, au, o);
    }
    if (l8 == 0) hrow[rl] = ag / (1.0f + __expf(-ag)) * au;
    __syncthreads();
    if (warp == 0) {
        const int nbh = ff / kQB;
        quant_block64(hrow[lane], hrow[lane + 32], reinterpret_cast<int8_t*>(hq + size_t(k) * 16 * nbh), nbh, blockIdx.x,
                      hscale + k * nbh + blockIdx.x, hsum + k * nbh + blockIdx.x);
    }
}

// Down of cache-hit expert k for 128 rows: yh[k][r]. 512 threads: 16 warps x 8 rows, 4 lanes
// per row. Needs n % 128 == 0, ff % 64 == 0, ff / 64 <= 64.
__global__ void k_moe_down(const uint8_t* slots, size_t slot_bytes, const int32_t* hit_slot, const int32_t* hit_n,
                           const uint32_t* hq, const float* hscale, const int32_t* hsum, float* yh, int n, int ff) {
    const int k = blockIdx.y;
    if (k >= *hit_n) return;
    __shared__ uint32_t hw[16 * 64];
    __shared__ float hs[64];
    __shared__ int hm[64];
    const int nbh = ff / kQB;
    for (int i = threadIdx.x; i < 16 * nbh; i += blockDim.x) hw[i] = hq[size_t(k) * 16 * nbh + i];
    for (int i = threadIdx.x; i < nbh; i += blockDim.x) {
        hs[i] = hscale[k * nbh + i];
        hm[i] = hsum[k * nbh + i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l4 = lane & 3;
    const int r = blockIdx.x * 128 + warp * 8 + (lane >> 2);
    const uint8_t* base = slots + size_t(hit_slot[k]) * slot_bytes + 2 * size_t(ff) * (n / kQB) * 18;
    const uint4* cd = reinterpret_cast<const uint4*>(base) + size_t(r) * nbh;
    const __half* sd = reinterpret_cast<const __half*>(base + size_t(n) * nbh * 16) + size_t(r) * nbh;
    float acc = 0.0f;
    for (int b = l4; b < nbh; b += 4) acc += __half2float(sd[b]) * hs[b] * float(dot_block64(cd[b], hw, nbh, b) - hm[b]);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    if (l4 == 0) yh[size_t(k) * n + r] = acc;
}

// One block of E threads (E <= 1024): softmax over the router logits, top-k by probability,
// weights renormalised (sum clamped at 6.1e-5, as llama.cpp). Hits get their slot; the hit list
// is padded to k with a real slot (or slot 0) at weight 0 so the grouped launches have a fixed
// shape. route_host gets [n_hits, n_miss, miss experts[k], miss weights[k] as float bits,
// then (at kRouteSel) all k selected experts in rank order].
// Doorbell mode (mb != nullptr): the record goes to the mailbox instead, followed by x [n], and
// then the routed flag is raised to seq.
__global__ void k_route(const float* logits, const int32_t* table, int E, int K, int32_t* hit_slot, float* hit_w,
                        int32_t* hit_n, int32_t* route_dev, const float* x, int n, uint8_t* mb, uint32_t seq) {
    if (mb) route_dev = reinterpret_cast<int32_t*>(mb + kMbRoute);
    __shared__ float p[1024];
    __shared__ float red_v[32];
    __shared__ int sel[kMaxK];
    __shared__ float selp[kMaxK];
    __shared__ int sel_slot[kMaxK];
    __shared__ int cand_i[1024];
    __shared__ float cand_p[1024];
    const int e = threadIdx.x;
    const float lg = e < E ? logits[e] : -INFINITY;
    const int my_slot = e < E ? table[e] : -1;   // loaded early, used if e is selected
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
    // top-k by rank: expert e's rank is the number of experts ahead of it (higher probability,
    // or equal and a lower index), so the order is that of k rounds of argmax, lowest index first.
    // Only candidates can be in the top k: p >= the k-th largest warp maximum (k warps have an
    // element at least that large). They are compacted, and ranked among themselves.
    const float pe = p[e];
    {
        float wm = pe;
        for (int o = 16; o > 0; o >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffff, wm, o));
        if ((e & 31) == 0) red_v[e >> 5] = wm;
    }
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float thr = -1.0f;
    if (K <= nw) {   // the K-th largest of the warp maxima
        for (int w = 0; w < nw; ++w) {
            const float v = red_v[w];
            int ahead = 0;
            for (int u = 0; u < nw; ++u) ahead += (red_v[u] > v) | ((red_v[u] == v) & (u < w));
            if (ahead == K - 1) thr = v;
        }
    }
    const bool cand = e < E && pe >= thr;
    int n_cand;
    const int ci = block_scan_flags(cand, &n_cand);
    if (cand) {
        cand_i[ci] = e;
        cand_p[ci] = pe;
    }
    __syncthreads();
    if (cand) {
        int rank = 0;
        for (int j = 0; j < n_cand; ++j) {
            const float pj = cand_p[j];
            rank += (pj > pe) | ((pj == pe) & (cand_i[j] < e));
        }
        if (rank < K) {
            sel[rank] = e;
            selp[rank] = pe;
            sel_slot[rank] = my_slot;
        }
    }
    __syncthreads();
    if (e == 0) {
        float ws = 0.0f;
        for (int k = 0; k < K; ++k) ws += selp[k];
        ws = fmaxf(ws, 6.103515625e-5f);
        int nh = 0, nm = 0;
        int pad = 0;
        for (int k = 0; k < K; ++k) {
            const int slot = sel_slot[k];
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
        for (int k = 0; k < K; ++k) route_dev[kRouteSel + k] = sel[k];
        route_dev[0] = nh;
        route_dev[1] = nm;
        *hit_n = nh;
    }
    if (mb) {
        float* xh = reinterpret_cast<float*>(mb + kMbX);
        for (int i = e; i < n; i += blockDim.x) xh[i] = x[i];
        __threadfence_system();
        __syncthreads();
        if (e == 0) *reinterpret_cast<volatile uint32_t*>(mb + kMbRouted) = seq;
    }
}

// out = sum_k hit_w[k] * yh[k] + cpu + shexp * sigmoid(gate)
__global__ void k_moe_combine(float* out, const float* yh, const float* hit_w, const int32_t* hit_n, const float* cpu,
                              const float* shexp, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = cpu[i];
    const int K = *hit_n;
    for (int k = 0; k < K; ++k) acc += hit_w[k] * yh[size_t(k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[0]));
    out[i] = acc + shexp[i] * g;
}

// Doorbell mode: as k_moe_combine, with the CPU part read from the mailbox once the host has
// raised its done flag to seq. After 10 s (and at least 10M polls, so a timer glitch cannot
// fire it) it gives up, records seq in the mailbox's error word, and carries on with whatever
// is there; the host reports the error after the token.
__global__ void k_moe_combine_db(float* out, const float* yh, const float* hit_w, const int32_t* hit_n, uint8_t* mb,
                                 size_t out_off, uint32_t seq, const float* shexp, const float* gate, int n) {
    if (threadIdx.x == 0) {
        const volatile uint32_t* done = reinterpret_cast<const volatile uint32_t*>(mb + kMbDone);
        const int64_t t0 = int64_t(global_ns());
        for (uint32_t polls = 0; *done != seq; ++polls) {
            __nanosleep(128);
            if (polls > 10000000u && int64_t(global_ns()) - t0 > 10000000000ll) {
                *reinterpret_cast<volatile uint32_t*>(mb + kMbErr) = seq;
                break;
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = reinterpret_cast<const volatile float*>(mb + out_off)[i];
    const int K = *hit_n;
    for (int k = 0; k < K; ++k) acc += hit_w[k] * yh[size_t(k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[0]));
    out[i] = acc + shexp[i] * g;
}

__global__ void k_swiglu_1(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

}  // namespace

size_t moe_hits_scratch_bytes(int K, int ff) { return size_t(K) * (ff / kQB) * (8 + 64) + 256; }

void moe_hits(const uint8_t* slots, size_t slot_bytes, const int32_t* hit_slot, const int32_t* hit_n, int K, const float* x, int n,
              int ff, void* scratch, float* yh, cudaStream_t stream) {
    if (n % 512 || ff % kQB || ff / kQB > 64 || K > kMaxK) throw std::runtime_error("moe_hits: unsupported expert shape");
    const int nbh = ff / kQB;
    float* hscale = static_cast<float*>(scratch);                               // [K][ff/64]
    int32_t* hsum = reinterpret_cast<int32_t*>(hscale + size_t(K) * nbh);       // [K][ff/64]
    uint32_t* hq = reinterpret_cast<uint32_t*>(hsum + size_t(K) * nbh);         // [K][16][ff/64]
    const size_t smem_gu = size_t(n / kQB) * (16 * 4 + 8);
    k_moe_gate_up<<<dim3(ff / kQB, K), 512, smem_gu, stream>>>(slots, slot_bytes, hit_slot, hit_n, x, n, ff, hq, hscale, hsum);
    k_moe_down<<<dim3(n / 128, K), 512, 0, stream>>>(slots, slot_bytes, hit_slot, hit_n, hq, hscale, hsum, yh, n, ff);
    ck(cudaGetLastError(), "moe_hits");
}

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
    return c;
}

void free_expert_cache(ExpertCache& c) {
    if (c.slots) cudaFree(c.slots);
    if (c.table_dev) cudaFree(c.table_dev);
    c = ExpertCache{};
}

void expert_cache_fill(const Spec& s, ExpertCache& cache, const ExpertArena& arena,
                       const std::vector<std::pair<int, int>>& order, cudaStream_t stream) {
    int slot = 0;
    for (const auto& [l, e] : order) {
        while (slot < cache.n_slots && cache.owner[slot] >= 0) ++slot;
        if (slot >= cache.n_slots) break;
        if (cache.table[size_t(l) * s.n_expert + e] >= 0) continue;
        ck(cudaMemcpyAsync(cache.slots + size_t(slot) * cache.slot_bytes, arena.blob(l, e), cache.slot_bytes, cudaMemcpyHostToDevice,
                           stream),
           "upload expert");
        cache.owner[slot] = l * s.n_expert + e;
        cache.table[size_t(l) * s.n_expert + e] = slot;
        ++slot;
    }
    ck(cudaMemcpyAsync(cache.table_dev, cache.table.data(), cache.table.size() * 4, cudaMemcpyHostToDevice, stream), "slot table");
    ck(cudaStreamSynchronize(stream), "expert_cache_fill");
}

MoeFastHost alloc_moe_fast_host(const Spec& s) {
    MoeFastHost h;
    ck(cudaHostAlloc(&h.route_host, size_t(kRouteInts) * 4, cudaHostAllocDefault), "cudaHostAlloc route");
    h.access.assign(size_t(s.n_layer) * s.top_k, -1);
    ck(cudaHostAlloc(&h.x_host, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc x");
    ck(cudaHostAlloc(&h.cpu_out, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc cpu out");
    ck(cudaEventCreateWithFlags(&h.routed, cudaEventDisableTiming), "cudaEventCreate");
    return h;
}

// ---- adaptive expert cache

namespace {
struct TableUpdates {
    int n;
    int32_t idx[64];
    int32_t val[64];
};
__global__ void k_table_update(int32_t* table, TableUpdates u) {
    if (int(threadIdx.x) < u.n) table[u.idx[threadIdx.x]] = u.val[threadIdx.x];
}
}  // namespace

struct CacheManager {
    const Spec* s = nullptr;
    ExpertCache* cache = nullptr;
    const ExpertArena* arena = nullptr;
    CachePolicyConfig cfg;
    cudaStream_t copy = nullptr;
    cudaEvent_t tok_done = nullptr;
    std::vector<float> count;                   // inflated: real count = count / w
    double w = 1.0;                             // weight of an access now
    long token = 0;
    std::set<std::pair<float, int>> resident;   // (count, key) of the cached experts
    struct Pending {
        int key, slot;
        cudaEvent_t ev;
    };
    std::vector<Pending> pending;
    std::vector<cudaEvent_t> events_free;
    std::vector<char> is_pending;               // per key
    bool registered = false;
    CacheStats stats;

    void bump(int key, float add) {
        const int slot = cache->table[key];
        if (slot >= 0) resident.erase({count[key], key});
        count[key] += add;
        if (slot >= 0) resident.insert({count[key], key});
    }
    void renormalise() {
        for (float& c : count) c = float(c / w);
        w = 1.0;
        resident.clear();
        for (int key = 0; key < int(count.size()); ++key)
            if (cache->table[key] >= 0) resident.insert({count[key], key});
    }
};

CacheManager* create_cache_manager(const Spec& s, ExpertCache& cache, const ExpertArena& arena, const CachePolicyConfig& cfg,
                                   const std::vector<uint32_t>& prior) {
    auto* m = new CacheManager;
    m->s = &s;
    m->cache = &cache;
    m->arena = &arena;
    m->cfg = cfg;
    const size_t n = size_t(s.n_layer) * s.n_expert;
    m->count.assign(n, 0.0f);
    for (size_t k = 0; k < n && k < prior.size(); ++k) m->count[k] = float(prior[k]);
    m->is_pending.assign(n, 0);
    for (size_t key = 0; key < n; ++key)
        if (cache.table[key] >= 0) m->resident.insert({m->count[key], int(key)});
    ck(cudaHostRegister(arena.buf.ptr, arena.total_bytes(), cudaHostRegisterDefault), "cudaHostRegister arena");
    m->registered = true;
    ck(cudaStreamCreateWithFlags(&m->copy, cudaStreamNonBlocking), "cudaStreamCreate copy");
    ck(cudaEventCreateWithFlags(&m->tok_done, cudaEventDisableTiming), "cudaEventCreate");
    for (int i = 0; i < cfg.budget; ++i) {
        cudaEvent_t ev;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        m->events_free.push_back(ev);
    }
    return m;
}

void destroy_cache_manager(CacheManager* m) {
    if (!m) return;
    if (m->copy) cudaStreamSynchronize(m->copy);
    for (auto& p : m->pending) m->events_free.push_back(p.ev);
    for (cudaEvent_t ev : m->events_free) cudaEventDestroy(ev);
    if (m->tok_done) cudaEventDestroy(m->tok_done);
    if (m->copy) cudaStreamDestroy(m->copy);
    if (m->registered) cudaHostUnregister(m->arena->buf.ptr);
    delete m;
}

CacheStats cache_manager_stats(const CacheManager* m) { return m->stats; }

void cache_manager_step(CacheManager* m, const MoeFastHost& h, cudaStream_t stream) {
    const Spec& s = *m->s;
    ExpertCache& c = *m->cache;
    const int E = s.n_expert, K = s.top_k;
    TableUpdates upd{};
    // 1. learn from the previous token's routing
    std::vector<int> missed;
    for (int il = 0; il < s.n_layer; ++il)
        for (int k = 0; k < K; ++k) {
            const int e = h.access_prev[size_t(il) * K + k];
            if (e < 0) continue;
            const int key = il * E + e;
            m->bump(key, float(m->w));
            if (c.table[key] < 0 && !m->is_pending[key]) missed.push_back(key);
        }
    if (++m->token % m->cfg.decay_every == 0) {
        m->w /= m->cfg.decay;
        if (m->w > 1e18) m->renormalise();
    }
    // 2. commit finished uploads (the table entry goes live from the next token on)
    for (size_t i = 0; i < m->pending.size();) {
        auto& p = m->pending[i];
        if (cudaEventQuery(p.ev) != cudaSuccess) { ++i; continue; }
        c.table[p.key] = p.slot;
        c.owner[p.slot] = p.key;
        m->resident.insert({m->count[p.key], p.key});
        m->is_pending[p.key] = 0;
        if (upd.n < 64) { upd.idx[upd.n] = p.key; upd.val[upd.n] = p.slot; ++upd.n; }
        m->events_free.push_back(p.ev);
        ++m->stats.committed;
        p = m->pending.back();
        m->pending.pop_back();
    }
    // 3. admit the strongest misses against the weakest residents
    std::sort(missed.begin(), missed.end(), [&](int a, int b) { return m->count[a] > m->count[b]; });
    missed.erase(std::unique(missed.begin(), missed.end()), missed.end());
    std::vector<std::pair<int, int>> uploads;   // (key, slot)
    for (int key : missed) {
        if (int(m->pending.size() + uploads.size()) >= m->cfg.budget || m->resident.empty() || upd.n >= 62) break;
        if (m->count[key] < m->cfg.admit * m->w) break;   // sorted: no later miss qualifies either
        const auto [vcount, victim] = *m->resident.begin();
        if (m->count[key] < m->cfg.margin * vcount) break;
        m->resident.erase(m->resident.begin());
        const int slot = c.table[victim];
        c.table[victim] = -1;
        c.owner[slot] = -1;
        upd.idx[upd.n] = victim;
        upd.val[upd.n] = -1;
        ++upd.n;
        m->is_pending[key] = 1;
        uploads.push_back({key, slot});
    }
    // 4. evictions and commits apply after this token; uploads start after it
    if (upd.n) k_table_update<<<1, 64, 0, stream>>>(c.table_dev, upd);
    if (uploads.empty()) return;
    ck(cudaEventRecord(m->tok_done, stream), "event token done");
    ck(cudaStreamWaitEvent(m->copy, m->tok_done, 0), "wait token done");
    for (const auto& [key, slot] : uploads) {   // the slot takes the arena's planar blob as is
        cudaEvent_t ev = m->events_free.back();
        m->events_free.pop_back();
        ck(cudaMemcpyAsync(c.slots + size_t(slot) * c.slot_bytes, m->arena->blob(key / E, key % E), c.slot_bytes,
                           cudaMemcpyHostToDevice, m->copy),
           "swap upload");
        ck(cudaEventRecord(ev, m->copy), "swap event");
        m->pending.push_back({key, slot, ev});
        ++m->stats.swaps;
    }
}

// ---- doorbell mode: the miss server

struct MissServer {
    const Spec* s = nullptr;
    MoeFastHost* h = nullptr;
    int cpu = -1;
    std::thread th;
    alignas(64) std::atomic<uint32_t> post{0};   // latest token to serve
    alignas(64) std::atomic<uint32_t> served{0}; // latest token fully served
    std::atomic<int> cur_layer{-1};              // diagnostics: the layer being waited for or served
    std::atomic<int> cur_phase{0};               // 0 idle, 1 waiting for routing, 2 running misses
    std::atomic<bool> quit{false};
    std::atomic<bool> failed{false};
    std::string error;

    void run() {
        if (cpu >= 0) pin_current_thread(cpu);
        const Spec& sp = *s;
        const int n = sp.d_model, K = sp.top_k;
        const q2_0::ExpertShape es{n, sp.d_ff_expert};
        std::vector<uint8_t> act_mem(q2_0::q8_bytes(n) + 64);
        q2_0::Q8Act act =
            q2_0::q8_view(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(act_mem.data()) + 63) & ~uintptr_t(63)), n);
        std::vector<uint8_t> scratch(q2_0::moe_cpu_scratch_bytes(es, kMaxK, h->pool->size()));
        q2_0::Miss miss[kMaxK];
        const size_t out_off = mb_out_off(n);
        uint32_t seen = post.load();
        for (;;) {
            uint32_t seq;
            while ((seq = post.load(std::memory_order_acquire)) == seen) {
                if (quit.load()) return;
                post.wait(seen, std::memory_order_acquire);
            }
            seen = seq;
            for (int il = 0; il < sp.n_layer; ++il) {
                uint8_t* mb = h->mbox + size_t(il) * h->mbox_stride;
                cur_layer.store(il, std::memory_order_relaxed);
                cur_phase.store(1, std::memory_order_relaxed);
                const auto t0 = std::chrono::steady_clock::now();
                int k = 0;
                while (__atomic_load_n(reinterpret_cast<uint32_t*>(mb + kMbRouted), __ATOMIC_ACQUIRE) != seq) {
#if defined(__x86_64__)
                    _mm_pause();
#endif
                    if (++k == 4096) {
                        k = 0;
                        if (quit.load()) return;
                        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
                            error = "miss server: no routing from the GPU for layer " + std::to_string(il) + " in 10 s";
                            failed.store(true);
                            return;
                        }
                    }
                }
                const auto t1 = std::chrono::steady_clock::now();
                cur_phase.store(2, std::memory_order_relaxed);
                const int32_t* route = reinterpret_cast<const int32_t*>(mb + kMbRoute);
                const int nh = route[0], nm = route[1];
                for (int k = 0; k < K; ++k) h->access[size_t(il) * K + k] = route[kRouteSel + k];
                float* out = reinterpret_cast<float*>(mb + out_off);
                if (nm > 0) {
                    q2_0::quantize_q8(reinterpret_cast<const float*>(mb + kMbX), act);
                    const float* mw = reinterpret_cast<const float*>(route + 2 + K);
                    for (int i = 0; i < nm; ++i) miss[i] = q2_0::Miss{h->arena->blob(il, route[2 + i]), 1, {0}, {mw[i]}};
                    q2_0::moe_cpu(*h->pool, es, miss, nm, &act, 1, out, n, scratch.data());
                } else {
                    std::memset(out, 0, size_t(n) * 4);
                }
                const auto t2 = std::chrono::steady_clock::now();
                h->hits += nh;
                h->misses += nm;
                h->wait_s += std::chrono::duration<double>(t1 - t0).count();
                const double dt = std::chrono::duration<double>(t2 - t1).count();
                h->cpu_s += dt;
                h->cpu_by_nm[std::min(nm, 16)] += dt;
                ++h->layers_by_nm[std::min(nm, 16)];
                __atomic_store_n(reinterpret_cast<uint32_t*>(mb + kMbDone), seq, __ATOMIC_RELEASE);
            }
            cur_phase.store(0, std::memory_order_relaxed);
            served.store(seq, std::memory_order_release);   // h->access and the statistics are complete
        }
    }
};

void start_doorbell(const Spec& s, MoeFastHost& h, int cpu) {
    if (s.top_k > kMaxK) throw std::runtime_error("start_doorbell: top-k too large");
    h.mbox_stride = (mb_out_off(s.d_model) + size_t(s.d_model) * 4 + 4095) & ~size_t(4095);
    const size_t bytes = size_t(s.n_layer) * h.mbox_stride;
    ck(cudaHostAlloc(&h.mbox, bytes, cudaHostAllocMapped), "cudaHostAlloc mailboxes");
    std::memset(h.mbox, 0, bytes);
    ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h.mbox_dev), h.mbox, 0), "mailbox device pointer");
    h.seq = 0;
    h.server = new MissServer;
    h.server->s = &s;
    h.server->h = &h;
    h.server->cpu = cpu;
    h.server->th = std::thread([srv = h.server] { srv->run(); });
    h.doorbell = true;
}

void doorbell_begin_token(MoeFastHost& h) {
    ++h.seq;
    h.server->post.store(h.seq, std::memory_order_release);
    h.server->post.notify_one();
}

void doorbell_end_token(MoeFastHost& h, const Spec& s) {
    for (int il = 0; il < s.n_layer; ++il) {
        const uint32_t err = __atomic_load_n(reinterpret_cast<uint32_t*>(h.mbox + size_t(il) * h.mbox_stride + kMbErr), __ATOMIC_ACQUIRE);
        if (err)
            throw std::runtime_error("doorbell: the GPU gave up waiting for layer " + std::to_string(il) + " of token " +
                                     std::to_string(err) + "; miss server at layer " + std::to_string(h.server->cur_layer.load()) +
                                     ", phase " + std::to_string(h.server->cur_phase.load()) + ", served " +
                                     std::to_string(h.server->served.load()) + ", posted " + std::to_string(h.server->post.load()) +
                                     (h.server->failed.load() ? ", server failed: " + h.server->error : std::string()));
    }
    // the GPU has consumed every done flag; the server's bookkeeping after the last one is brief
    while (h.server->served.load(std::memory_order_acquire) != h.seq)
        if (h.server->failed.load()) throw std::runtime_error(h.server->error);
}

void free_moe_fast_host(MoeFastHost& h) {
    if (h.server) {
        h.server->quit.store(true);
        h.server->post.fetch_add(0x80000000u);
        h.server->post.notify_one();
        h.server->th.join();
        delete h.server;
    }
    if (h.mbox) cudaFreeHost(h.mbox);
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
    float* yh = gate + 4;                          // [K][n]
    float* cpu_dev = yh + size_t(K) * n;           // [n]
    float* hit_w = cpu_dev + n;                    // [K]
    int32_t* hit_slot = reinterpret_cast<int32_t*>(hit_w + kMaxK);     // [K]
    int32_t* route_dev = hit_slot + kMaxK;                              // [kRouteInts]
    int32_t* hit_n = route_dev + kRouteInts;                            // [1] (padded to 4)
    float* hits_scratch = reinterpret_cast<float*>(hit_n + 4);          // moe_hits_scratch_bytes
    if (size_t(hits_scratch - c.scratch.f32) * 4 + moe_hits_scratch_bytes(K, ff) > c.scratch.f32_elems * 4)
        throw std::runtime_error("moe_block_fast: scratch too small");

    // 1. routing, and the input for the CPU
    linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, logits, 1);
    uint8_t* mb = h.doorbell ? h.mbox_dev + size_t(il) * h.mbox_stride : nullptr;
    k_route<<<1, ((E + 31) / 32) * 32, 0, c.stream>>>(logits, cache.table_dev + size_t(il) * E, E, K, hit_slot, hit_w, hit_n,
                                                     route_dev, x, n, mb, h.seq);
    if (!h.doorbell) {
        ck(cudaMemcpyAsync(h.route_host, route_dev, size_t(kRouteInts) * 4, cudaMemcpyDeviceToHost, c.stream), "route to host");
        ck(cudaMemcpyAsync(h.x_host, x, size_t(n) * 4, cudaMemcpyDeviceToHost, c.stream), "x to host");
        ck(cudaEventRecord(h.routed, c.stream), "event");
    }

    // 2. GPU: cache hits (fused gate+up, then down) and the shared expert
    moe_hits(cache.slots, cache.slot_bytes, hit_slot, hit_n, K, x, n, ff, hits_scratch, yh, c.stream);
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, sg, 1);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, su, 1);
    k_swiglu_1<<<(ffs + 255) / 256, 256, 0, c.stream>>>(sg, su, ffs);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), sg, sh, 1);
    linear(c, c.w.layer(il, "ffn_gate_inp_shexp.weight"), x, gate, 1);

    if (h.doorbell) {   // 3'. the miss server fills the mailbox; the combine waits for it on the GPU
        k_moe_combine_db<<<(n + 255) / 256, 256, 0, c.stream>>>(out, yh, hit_w, hit_n, mb, mb_out_off(n), h.seq, sh, gate, n);
        ck(cudaGetLastError(), "moe_block_fast");
        return;
    }

    // 3. CPU: the misses, while the GPU works
    const auto t0 = std::chrono::steady_clock::now();
    ck(cudaEventSynchronize(h.routed), "wait routing");
    const auto t1 = std::chrono::steady_clock::now();
    const int nh = h.route_host[0], nm = h.route_host[1];
    for (int k = 0; k < K; ++k) h.access[size_t(il) * K + k] = h.route_host[kRouteSel + k];
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
    h.wait_s += std::chrono::duration<double>(t1 - t0).count();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    h.cpu_s += dt;
    h.cpu_by_nm[std::min(nm, 16)] += dt;
    ++h.layers_by_nm[std::min(nm, 16)];
    ck(cudaMemcpyAsync(cpu_dev, h.cpu_out, size_t(n) * 4, cudaMemcpyHostToDevice, c.stream), "cpu result to device");
    k_moe_combine<<<(n + 255) / 256, 256, 0, c.stream>>>(out, yh, hit_w, hit_n, cpu_dev, sh, gate, n);
    ck(cudaGetLastError(), "moe_block_fast");
}

}  // namespace flashrt::qwen4exp
