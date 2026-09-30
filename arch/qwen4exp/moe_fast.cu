// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/moe_cpu.hpp"

#include "core/platform.hpp"

#include <cuda/atomic>
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

// The mailbox flags (routed, done, err) between the GPU and the miss server, as system-scope
// atomics: release after writing what they announce, acquire before reading it.
using SysFlag = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

constexpr int kMaxK = 16;   // top-k upper bound for the fixed launch shapes
constexpr int kRouteSel = 2 + 2 * kMaxK;   // route record: offset of the selected experts
constexpr int kRouteGpu = kRouteSel + kMaxK;   // route record: misses the GPU reads from host memory
constexpr int kRouteInts = kRouteGpu + 1;

// mailbox layout (one per layer, doorbell mode), for windows of up to W tokens
constexpr int kMaxWindow = 8;
constexpr size_t kMbRouted = 0;     // uint32 [W], GPU -> host: token t's routing and x are in place for step seq
constexpr size_t kMbDone = 64;      // uint32, host -> GPU: out is in place for step seq
constexpr size_t kMbErr = 96;       // uint32, GPU -> host: the combine gave up waiting for step seq
constexpr size_t kMbRoute = 128;    // int32 [W][kRouteInts], as route_dev
size_t mb_x_off(int W) { return (kMbRoute + size_t(W) * kRouteInts * 4 + 63) & ~size_t(63); }   // float [W][d_model]
size_t mb_out_off(int n, int W) { return (mb_x_off(W) + size_t(W) * n * 4 + 63) & ~size_t(63); }   // float [W][d_model]

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

// Gate and up of GPU expert k (blockIdx.y < *hit_n; weights at hit_ptr[k], a cache slot or
// mapped host memory) for 64 rows (blockIdx.x), SwiGLU, and
// the hidden block quantized for the down kernel: hq words [k][16][ff/64], hscale/hsum [k][ff/64].
// 512 threads: 16 warps x 4 rows, 8 lanes per row. Needs n % 512 == 0 and ff % 64 == 0.
// Windows: blockIdx.y = t * K + k for token t (x row t, hit_n[t], hit_ptr[t * K + k]).
__global__ void k_moe_gate_up(const uint8_t* const* hit_ptr, const int32_t* hit_n, const float* x, int n, int ff, uint32_t* hq,
                              float* hscale, int32_t* hsum, int K) {
    const int k = blockIdx.y, tok = k / K;
    if (k % K >= hit_n[tok]) return;
    x += size_t(tok) * n;
    extern __shared__ __align__(16) uint32_t xw[];   // [16][nb] words, then scale [nb], sum [nb]
    __shared__ float hrow[kQB];
    const int nb = n / kQB, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* xscale = reinterpret_cast<float*>(xw + 16 * nb);
    int* xsum = reinterpret_cast<int*>(xscale + nb);
    for (int b = warp; b < nb; b += blockDim.x >> 5)
        quant_block64(x[b * kQB + lane], x[b * kQB + 32 + lane], reinterpret_cast<int8_t*>(xw), nb, b, xscale + b, xsum + b);
    __syncthreads();
    const int rsub = lane >> 3, l8 = lane & 7, rl = warp * 4 + rsub, r = blockIdx.x * kQB + rl;
    const uint8_t* base = hit_ptr[k];
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
__global__ void k_moe_down(const uint8_t* const* hit_ptr, const int32_t* hit_n, const uint32_t* hq, const float* hscale,
                           const int32_t* hsum, float* yh, int n, int ff, int K) {
    const int k = blockIdx.y;
    if (k % K >= hit_n[k / K]) return;
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
    const uint8_t* base = hit_ptr[k] + 2 * size_t(ff) * (n / kQB) * 18;
    const uint4* cd = reinterpret_cast<const uint4*>(base) + size_t(r) * nbh;
    const __half* sd = reinterpret_cast<const __half*>(base + size_t(n) * nbh * 16) + size_t(r) * nbh;
    float acc = 0.0f;
    for (int b = l4; b < nbh; b += 4) acc += __half2float(sd[b]) * hs[b] * float(dot_block64(cd[b], hw, nbh, b) - hm[b]);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    if (l4 == 0) yh[size_t(k) * n + r] = acc;
}

// ---- windows: the hits grouped by expert, so an expert several tokens hit is read once
constexpr int kGroupTok = 4;   // tokens per window the grouped kernels take

// One block of 64 threads, one per (token, k) entry of the window's hit lists (at most 64):
// the distinct experts g_ptr[u] (u < *g_n, in order of first appearance) and, per expert and
// token, the entry's k (g_slot[u][t], -1 if the token did not select it).
__global__ void k_moe_group(const uint8_t* const* hit_ptr, const int32_t* hit_n, int K, int T, const uint8_t** g_ptr, int8_t* g_slot,
                            int32_t* g_n) {
    __shared__ const uint8_t* ptr[64];
    __shared__ int u_of[64];
    const int i = threadIdx.x;
    int t = -1, k = -1, N = 0;
    for (int tt = 0; tt < T; ++tt) {
        const int h = hit_n[tt];
        if (i >= N && i < N + h) {
            t = tt;
            k = i - N;
        }
        N += h;
    }
    const bool live = i < N;
    ptr[i] = live ? hit_ptr[t * K + k] : nullptr;
    __syncthreads();
    int f = i;   // the first entry with this expert
    if (live)
        for (int j = 0; j < i; ++j)
            if (ptr[j] == ptr[i]) {
                f = j;
                break;
            }
    int total;
    const int u = block_scan_flags(live && f == i, &total);
    if (live && f == i) {
        u_of[i] = u;
        g_ptr[u] = ptr[i];
        for (int tt = 0; tt < kGroupTok; ++tt) g_slot[u * kGroupTok + tt] = -1;
    }
    __syncthreads();
    if (live) g_slot[u_of[f] * kGroupTok + t] = int8_t(k);
    if (i == 0) *g_n = total;
}

// Gate and up of grouped expert u (blockIdx.y < *g_n) for 64 rows (blockIdx.x), for every token
// of the window that selected it; the hidden rows go where k_moe_gate_up puts them (entry
// t * K + k), so k_moe_down's layout and the combine are unchanged.
__global__ void k_moe_gate_up_g(const uint8_t* const* g_ptr, const int8_t* g_slot, const int32_t* g_n, const float* x, int T, int n,
                                int ff, int K, uint32_t* hq, float* hscale, int32_t* hsum) {
    const int u = blockIdx.y;
    if (u >= *g_n) return;
    extern __shared__ __align__(16) uint32_t xw[];   // [T][16][nb] words, then scale [T][nb], sum [T][nb]
    __shared__ float hrow[kGroupTok][kQB];
    const int nb = n / kQB, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* xscale = reinterpret_cast<float*>(xw + size_t(T) * 16 * nb);
    int* xsum = reinterpret_cast<int*>(xscale + size_t(T) * nb);
    int8_t slot[kGroupTok];
#pragma unroll
    for (int t = 0; t < kGroupTok; ++t) slot[t] = t < T ? g_slot[u * kGroupTok + t] : int8_t(-1);
    for (int t = 0; t < T; ++t) {
        if (slot[t] < 0) continue;
        for (int b = warp; b < nb; b += blockDim.x >> 5)
            quant_block64(x[size_t(t) * n + b * kQB + lane], x[size_t(t) * n + b * kQB + 32 + lane],
                          reinterpret_cast<int8_t*>(xw + size_t(t) * 16 * nb), nb, b, xscale + t * nb + b, xsum + t * nb + b);
    }
    __syncthreads();
    const int rsub = lane >> 3, l8 = lane & 7, rl = warp * 4 + rsub, r = blockIdx.x * kQB + rl;
    const uint8_t* base = g_ptr[u];
    const size_t mb = size_t(ff) * nb * 18;
    const uint4* cg = reinterpret_cast<const uint4*>(base) + size_t(r) * nb;
    const __half* sg = reinterpret_cast<const __half*>(base + size_t(ff) * nb * 16) + size_t(r) * nb;
    const uint4* cu = reinterpret_cast<const uint4*>(base + mb) + size_t(r) * nb;
    const __half* su = reinterpret_cast<const __half*>(base + mb + size_t(ff) * nb * 16) + size_t(r) * nb;
    float ag[kGroupTok] = {}, au[kGroupTok] = {};
    for (int b = l8; b < nb; b += 8) {
        const uint4 wg = cg[b], wu = cu[b];
        const float dg = __half2float(sg[b]), du = __half2float(su[b]);
#pragma unroll
        for (int t = 0; t < kGroupTok; ++t) {
            if (slot[t] < 0) continue;
            const uint32_t* xt = xw + size_t(t) * 16 * nb;
            const float xs = xscale[t * nb + b];
            const int xm = xsum[t * nb + b];
            ag[t] += dg * xs * float(dot_block64(wg, xt, nb, b) - xm);
            au[t] += du * xs * float(dot_block64(wu, xt, nb, b) - xm);
        }
    }
#pragma unroll
    for (int t = 0; t < kGroupTok; ++t) {
        if (slot[t] < 0) continue;
        float a = ag[t], c = au[t];
        for (int o = 4; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, o);
            c += __shfl_xor_sync(0xffffffff, c, o);
        }
        if (l8 == 0) hrow[t][rl] = a / (1.0f + __expf(-a)) * c;
    }
    __syncthreads();
    if (warp < T && slot[warp] >= 0) {
        const int nbh = ff / kQB, e = warp * K + slot[warp];
        quant_block64(hrow[warp][lane], hrow[warp][lane + 32], reinterpret_cast<int8_t*>(hq + size_t(e) * 16 * nbh), nbh, blockIdx.x,
                      hscale + e * nbh + blockIdx.x, hsum + e * nbh + blockIdx.x);
    }
}

// Down of grouped expert u for 128 rows, for every token that selected it: yh[t * K + k][r].
__global__ void k_moe_down_g(const uint8_t* const* g_ptr, const int8_t* g_slot, const int32_t* g_n, const uint32_t* hq,
                             const float* hscale, const int32_t* hsum, float* yh, int T, int n, int ff, int K) {
    const int u = blockIdx.y;
    if (u >= *g_n) return;
    __shared__ uint32_t hw[kGroupTok][16 * 64];
    __shared__ float hs[kGroupTok][64];
    __shared__ int hm[kGroupTok][64];
    const int nbh = ff / kQB;
    int8_t slot[kGroupTok];
#pragma unroll
    for (int t = 0; t < kGroupTok; ++t) slot[t] = t < T ? g_slot[u * kGroupTok + t] : int8_t(-1);
    for (int t = 0; t < T; ++t) {
        if (slot[t] < 0) continue;
        const int e = t * K + slot[t];
        for (int i = threadIdx.x; i < 16 * nbh; i += blockDim.x) hw[t][i] = hq[size_t(e) * 16 * nbh + i];
        for (int i = threadIdx.x; i < nbh; i += blockDim.x) {
            hs[t][i] = hscale[e * nbh + i];
            hm[t][i] = hsum[e * nbh + i];
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l4 = lane & 3;
    const int r = blockIdx.x * 128 + warp * 8 + (lane >> 2);
    const uint8_t* base = g_ptr[u] + 2 * size_t(ff) * (n / kQB) * 18;
    const uint4* cd = reinterpret_cast<const uint4*>(base) + size_t(r) * nbh;
    const __half* sd = reinterpret_cast<const __half*>(base + size_t(n) * nbh * 16) + size_t(r) * nbh;
    float acc[kGroupTok] = {};
    for (int b = l4; b < nbh; b += 4) {
        const uint4 w = cd[b];
        const float d = __half2float(sd[b]);
#pragma unroll
        for (int t = 0; t < kGroupTok; ++t)
            if (slot[t] >= 0) acc[t] += d * hs[t][b] * float(dot_block64(w, hw[t], nbh, b) - hm[t][b]);
    }
#pragma unroll
    for (int t = 0; t < kGroupTok; ++t) {
        if (slot[t] < 0) continue;
        float a = acc[t];
        a += __shfl_xor_sync(0xffffffff, a, 2);
        a += __shfl_xor_sync(0xffffffff, a, 1);
        if (l4 == 0) yh[size_t(t * K + slot[t]) * n + r] = a;
    }
}

// One block of E threads (E <= 1024): softmax over the router logits, top-k by probability,
// weights renormalised (sum clamped at 6.1e-5, as llama.cpp). Hits get their slot; the hit list
// is padded to k with a real slot (or slot 0) at weight 0 so the grouped launches have a fixed
// shape. route_host gets [n_hits, n_miss, miss experts[k], miss weights[k] as float bits,
// then (at kRouteSel) all k selected experts in rank order].
// Doorbell mode (mb != nullptr): the record goes to the mailbox instead, followed by x [n], and
// then the routed flag is raised to seq.
// GPU experts: hits point at their cache slot (slots + slot * slot_bytes); with arena_dev set,
// floor(misses * pcie_frac) (at most pcie_max) of the misses point at their blob in mapped host
// memory and are read over PCIe by the hit kernels, and only the rest go to the CPU.
// Windows: one block per token t (blockIdx.x), each with its own logits, x, hit list, route
// record and routed flag.
// miss_n[t] (device) gets the token's CPU miss count: a token without misses skips the x copy
// here and the mailbox wait and read in k_moe_combine_db (sw77).
__global__ void k_route(const float* logits, const int32_t* table, int E, int K, const uint8_t* slots, size_t slot_bytes,
                        const uint8_t** hit_ptr, float* hit_w, int32_t* hit_n, const uint8_t* arena_dev, size_t arena_stride,
                        int layer, float pcie_frac, int pcie_max, int32_t* route_dev, const float* x, int n, uint8_t* mb,
                        size_t x_off, uint32_t seq, const int32_t* dp, int32_t* miss_n) {
    if (dp) seq = uint32_t(dp[2]);
    const int tok = blockIdx.x;
    logits += size_t(tok) * E;
    hit_ptr += size_t(tok) * K;
    hit_w += size_t(tok) * K;
    hit_n += tok;
    x += size_t(tok) * n;
    route_dev = mb ? reinterpret_cast<int32_t*>(mb + kMbRoute) + size_t(tok) * kRouteInts : route_dev + size_t(tok) * kRouteInts;
    __shared__ float p[1024];
    __shared__ float red_v[32];
    __shared__ int sel[kMaxK];
    __shared__ float selp[kMaxK];
    __shared__ int sel_slot[kMaxK];
    __shared__ int cand_i[1024];
    __shared__ float cand_p[1024];
    __shared__ int s_nm;
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
        int misses = 0;
        for (int k = 0; k < K; ++k) misses += sel_slot[k] < 0;
        const int n_gpu = arena_dev ? min(pcie_max, int(float(misses) * pcie_frac)) : 0;
        int nh = 0, ng = 0, nm = 0;
        const uint8_t* pad = slots;
        for (int k = 0; k < K; ++k) {
            const int slot = sel_slot[k];
            const float w = selp[k] / ws;
            if (slot >= 0) {
                hit_ptr[nh + ng] = pad = slots + size_t(slot) * slot_bytes;
                hit_w[nh + ng] = w;
                ++nh;
            } else if (ng < n_gpu) {
                hit_ptr[nh + ng] = pad = arena_dev + (size_t(layer) * E + sel[k]) * arena_stride;
                hit_w[nh + ng] = w;
                ++ng;
            } else {
                route_dev[2 + nm] = sel[k];
                reinterpret_cast<float*>(route_dev)[2 + K + nm] = w;
                ++nm;
            }
        }
        for (int k = nh + ng; k < K; ++k) {
            hit_ptr[k] = pad;
            hit_w[k] = 0.0f;
        }
        for (int k = 0; k < K; ++k) route_dev[kRouteSel + k] = sel[k];
        route_dev[0] = nh;
        route_dev[1] = nm;
        route_dev[kRouteGpu] = ng;
        *hit_n = nh + ng;
        miss_n[tok] = nm;
        s_nm = nm;
    }
    __syncthreads();
    if (mb) {
        if (s_nm > 0) {   // the CPU reads x only for its misses
            float* xh = reinterpret_cast<float*>(mb + x_off) + size_t(tok) * n;
            for (int i = e; i < n; i += blockDim.x) xh[i] = x[i];
        }
        __syncthreads();   // the block's x and route record, then one release publishes them
        if (e == 0) SysFlag(reinterpret_cast<uint32_t*>(mb + kMbRouted)[tok]).store(seq, cuda::memory_order_release);
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
// Windows: blockIdx.y = token t.
// A token without CPU misses (miss_n[t] == 0) neither waits nor reads the mailbox: its CPU part
// is zero. The host still serves the layer (statistics), and doorbell_end_token waits for it.
__global__ void k_moe_combine_db(float* out, const float* yh, const float* hit_w, const int32_t* hit_n, uint8_t* mb,
                                 size_t out_off, uint32_t seq, const float* shexp, const float* gate, int n, int Kmax, const int32_t* dp,
                                 const int32_t* miss_n) {
    if (dp) seq = uint32_t(dp[2]);
    const bool cpu = miss_n[blockIdx.y] != 0;
    if (cpu && threadIdx.x == 0) {
        SysFlag done(*reinterpret_cast<uint32_t*>(mb + kMbDone));
        const int64_t t0 = int64_t(global_ns());
        for (uint32_t polls = 0; done.load(cuda::memory_order_relaxed) != seq; ++polls) {
            __nanosleep(128);
            if (polls > 10000000u && int64_t(global_ns()) - t0 > 10000000000ll) {
                SysFlag(*reinterpret_cast<uint32_t*>(mb + kMbErr)).store(seq, cuda::memory_order_relaxed);
                break;
            }
        }
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = cpu ? reinterpret_cast<const volatile float*>(mb + out_off)[size_t(t) * n + i] : 0.0f;
    const int K = hit_n[t];
    for (int k = 0; k < K; ++k) acc += hit_w[t * Kmax + k] * yh[(size_t(t) * Kmax + k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[t]));
    out[size_t(t) * n + i] = acc + shexp[size_t(t) * n + i] * g;
}

__global__ void k_swiglu_1(float* g, const float* u, int n) {   // any number of rows
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

}  // namespace

size_t moe_hits_scratch_bytes(int K, int ff, int T) {
    return size_t(T) * K * (ff / kQB) * (8 + 64) + 256 + size_t(T) * K * (8 + kGroupTok) + 64;   // + the grouped lists
}

void moe_hits(const uint8_t* const* hit_ptr, const int32_t* hit_n, int K, const float* x, int n, int ff, void* scratch, float* yh,
              cudaStream_t stream, int T) {
    if (n % 512 || ff % kQB || ff / kQB > 64 || K > kMaxK || T < 1) throw std::runtime_error("moe_hits: unsupported expert shape");
    const int nbh = ff / kQB, TK = T * K;
    float* hscale = static_cast<float*>(scratch);                               // [T*K][ff/64]
    int32_t* hsum = reinterpret_cast<int32_t*>(hscale + size_t(TK) * nbh);      // [T*K][ff/64]
    uint32_t* hq = reinterpret_cast<uint32_t*>(hsum + size_t(TK) * nbh);        // [T*K][16][ff/64]
    const size_t smem_gu = size_t(n / kQB) * (16 * 4 + 8);
    if (T > 1 && T <= kGroupTok && TK <= 64) {   // a window: each distinct expert read once
        const uint8_t** g_ptr = reinterpret_cast<const uint8_t**>((reinterpret_cast<uintptr_t>(hq + size_t(TK) * 16 * nbh) + 15) & ~uintptr_t(15));
        int8_t* g_slot = reinterpret_cast<int8_t*>(g_ptr + TK);
        int32_t* g_n = reinterpret_cast<int32_t*>((reinterpret_cast<uintptr_t>(g_slot + size_t(TK) * kGroupTok) + 3) & ~uintptr_t(3));
        k_moe_group<<<1, 64, 0, stream>>>(hit_ptr, hit_n, K, T, g_ptr, g_slot, g_n);
        k_moe_gate_up_g<<<dim3(ff / kQB, TK), 512, smem_gu * T, stream>>>(g_ptr, g_slot, g_n, x, T, n, ff, K, hq, hscale, hsum);
        k_moe_down_g<<<dim3(n / 128, TK), 512, 0, stream>>>(g_ptr, g_slot, g_n, hq, hscale, hsum, yh, T, n, ff, K);
        ck(cudaGetLastError(), "moe_hits (grouped)");
        return;
    }
    k_moe_gate_up<<<dim3(ff / kQB, TK), 512, smem_gu, stream>>>(hit_ptr, hit_n, x, n, ff, hq, hscale, hsum, K);
    k_moe_down<<<dim3(n / 128, TK), 512, 0, stream>>>(hit_ptr, hit_n, hq, hscale, hsum, yh, n, ff, K);
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

MoeFastHost alloc_moe_fast_host(const Spec& s, int max_window) {
    MoeFastHost h;
    if (max_window < 1 || max_window > kMaxWindow) throw std::runtime_error("alloc_moe_fast_host: window must be 1..8");
    h.max_window = max_window;
    ck(cudaHostAlloc(&h.route_host, size_t(kRouteInts) * 4, cudaHostAllocDefault), "cudaHostAlloc route");
    h.access.assign(size_t(max_window) * s.n_layer * s.top_k, -1);
    ck(cudaHostAlloc(&h.x_host, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc x");
    ck(cudaHostAlloc(&h.cpu_out, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc cpu out");
    ck(cudaEventCreateWithFlags(&h.routed, cudaEventDisableTiming), "cudaEventCreate");
    return h;
}

// ---- adaptive expert cache

namespace {
constexpr int kMaxTableUpdates = 256;
constexpr long kCommitLag = 1;   // steps from an upload's issue to its commit (2 lost on short agent turns: sw108, sw109)   // per token: evictions and commits (a budget of 64 needs more than 64)
struct TableUpdates {
    int n;
    int32_t idx[kMaxTableUpdates];
    int32_t val[kMaxTableUpdates];
};
__global__ void k_table_update(int32_t* table, TableUpdates u) {
    if (int(threadIdx.x) < u.n) table[u.idx[threadIdx.x]] = u.val[threadIdx.x];
}
}  // namespace

// Registers the arena with CUDA once (mapped), whoever asks first; returns its device address.
const uint8_t* arena_register(const ExpertArena& arena) {
    unsigned flags = 0;
    if (cudaHostGetFlags(&flags, arena.buf.ptr) != cudaSuccess) {
        cudaGetLastError();   // not registered yet
        ck(cudaHostRegister(arena.buf.ptr, arena.total_bytes(), cudaHostRegisterMapped), "cudaHostRegister arena");
    }
    void* dev = nullptr;
    ck(cudaHostGetDevicePointer(&dev, arena.buf.ptr, 0), "arena device pointer");
    return static_cast<const uint8_t*>(dev);
}

void enable_pcie_misses(MoeFastHost& h, const ExpertArena& arena, float frac, int max_per_layer) {
    h.arena_dev = arena_register(arena);
    h.arena_stride = arena.stride;
    h.pcie_frac = frac;
    h.pcie_max = max_per_layer;
}

struct CacheManager {
    const Spec* s = nullptr;
    ExpertCache* cache = nullptr;
    const ExpertArena* arena = nullptr;
    CachePolicyConfig cfg;
    cudaStream_t copy = nullptr;
    cudaEvent_t tok_done = nullptr;
    std::vector<float> count;                   // inflated: real count = count / w
    double w = 1.0;                             // weight of an access now
    long token = 0;    // steps
    std::set<std::pair<float, int>> resident;   // (count, key) of the cached experts
    struct Pending {
        int key, slot;
        cudaEvent_t ev;
        long step;   // the step that issued it
    };
    std::vector<Pending> pending;
    std::vector<cudaEvent_t> events_free;
    std::vector<char> is_pending;               // per key
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
    for (size_t k = 0; k < n && k < prior.size(); ++k) m->count[k] = float(prior[k]) * cfg.seed_scale;
    m->is_pending.assign(n, 0);
    for (size_t key = 0; key < n; ++key)
        if (cache.table[key] >= 0) m->resident.insert({m->count[key], int(key)});
    arena_register(arena);
    ck(cudaStreamCreateWithFlags(&m->copy, cudaStreamNonBlocking), "cudaStreamCreate copy");
    ck(cudaEventCreateWithFlags(&m->tok_done, cudaEventDisableTiming), "cudaEventCreate");
    for (int i = 0; i < 2 * cfg.budget; ++i) {   // this step's uploads and the previous step's, pending
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
    delete m;
}

CacheStats cache_manager_stats(const CacheManager* m) { return m->stats; }

void cache_manager_step(CacheManager* m, const MoeFastHost& h, cudaStream_t stream) {
    const Spec& s = *m->s;
    ExpertCache& c = *m->cache;
    const int E = s.n_expert, K = s.top_k;
    TableUpdates upd{};
    // 1. learn from the previous step's routing (every token of it)
    std::vector<int> missed;
    for (int t = 0; t < h.access_prev_T; ++t)
        for (int il = 0; il < s.n_layer; ++il)
            for (int k = 0; k < K; ++k) {
                const int e = h.access_prev[(size_t(t) * s.n_layer + il) * K + k];
                if (e < 0) continue;
                const int key = il * E + e;
                m->bump(key, float(m->w));
                if (c.table[key] < 0 && !m->is_pending[key]) missed.push_back(key);
            }
    const int budget = m->cfg.budget;
    if (++m->token % m->cfg.decay_every == 0) {
        m->w /= m->cfg.decay;
        if (m->w > 1e18) m->renormalise();
    }
    // 2. commit the uploads issued at the previous step (the table entry goes live from the next
    // token on). A fixed step, not whichever uploads a query finds finished: with 64 in flight
    // that depended on timing, and so did the cache's content, the hit/miss arithmetic and the
    // outputs (sw106: the fast-path KLD varied 0.00876-0.00920 between runs). The wait overlaps
    // the token just enqueued: this thread synchronises on it next anyway, and CPU misses run on
    // the miss server's thread. (Two steps later cost 1-2 points of hits on short turns, sw108.)
    for (size_t i = 0; i < m->pending.size();) {
        auto& p = m->pending[i];
        if (p.step > m->token - kCommitLag) { ++i; continue; }
        ck(cudaEventSynchronize(p.ev), "swap commit");
        c.table[p.key] = p.slot;
        c.owner[p.slot] = p.key;
        m->resident.insert({m->count[p.key], p.key});
        m->is_pending[p.key] = 0;
        if (upd.n < kMaxTableUpdates) { upd.idx[upd.n] = p.key; upd.val[upd.n] = p.slot; ++upd.n; }
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
        // the budget is per step: counting the pending ones too would halve it, as each stays two
        // steps (sw107: with the MTP head 108.5 -> 96.9 tok/s)
        if (int(uploads.size()) >= budget || m->resident.empty() || upd.n >= kMaxTableUpdates - 2) break;
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
    if (upd.n) k_table_update<<<1, kMaxTableUpdates, 0, stream>>>(c.table_dev, upd);
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
        m->pending.push_back({key, slot, ev, m->token});
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
        const int W = h->max_window;
        const size_t qb = (q2_0::q8_bytes(n) + 63) & ~size_t(63);
        std::vector<uint8_t> act_mem(qb * W + 64);
        std::vector<q2_0::Q8Act> act(W);
        for (int t = 0; t < W; ++t)
            act[t] = q2_0::q8_view(
                reinterpret_cast<void*>(((reinterpret_cast<uintptr_t>(act_mem.data()) + 63) & ~uintptr_t(63)) + size_t(t) * qb), n);
        // a window's misses grouped by expert, up to 4 tokens per entry
        const int max_miss = W * kMaxK;
        std::vector<uint8_t> scratch(q2_0::moe_cpu_scratch_bytes(es, max_miss, h->pool->size()));
        std::vector<q2_0::Miss> miss(max_miss);
        std::vector<int> slot_of(sp.n_expert, -1), entry_expert(max_miss);
        const size_t out_off = mb_out_off(n, W), x_off = mb_x_off(W);
        uint32_t seen = post.load();
        for (;;) {
            uint32_t seq;
            while ((seq = post.load(std::memory_order_acquire)) == seen) {
                if (quit.load()) return;
                post.wait(seen, std::memory_order_acquire);
            }
            seen = seq;
            const int T = h->window_T;
            for (int il = 0; il < sp.n_layer; ++il) {
                uint8_t* mb = h->mbox + size_t(il) * h->mbox_stride;
                cur_layer.store(il, std::memory_order_relaxed);
                cur_phase.store(1, std::memory_order_relaxed);
                const auto t0 = std::chrono::steady_clock::now();
                int k = 0;
                for (int t = 0; t < T; ++t)
                    while (std::atomic_ref<uint32_t>(reinterpret_cast<uint32_t*>(mb + kMbRouted)[t]).load(std::memory_order_acquire) != seq) {
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
                int nh = 0, nm = 0, n_entries = 0;
                for (int t = 0; t < T; ++t) {
                    const int32_t* route = reinterpret_cast<const int32_t*>(mb + kMbRoute) + size_t(t) * kRouteInts;
                    const float* mw = reinterpret_cast<const float*>(route + 2 + K);
                    nh += route[0];
                    h->gpu_misses += route[kRouteGpu];
                    for (int k = 0; k < K; ++k) h->access[(size_t(t) * sp.n_layer + il) * K + k] = route[kRouteSel + k];
                    for (int i = 0; i < route[1]; ++i) {
                        const int e = route[2 + i];
                        int& sl = slot_of[e];
                        if (sl < 0 || miss[sl].n_tok == 4) {   // a new entry for this expert
                            sl = n_entries++;
                            miss[sl] = q2_0::Miss{h->arena->blob(il, e), 0, {0}, {0}};
                            entry_expert[sl] = e;
                        }
                        q2_0::Miss& m = miss[sl];
                        m.tok[m.n_tok] = t;
                        m.w[m.n_tok] = mw[i];
                        ++m.n_tok;
                        ++nm;
                    }
                }
                float* out = reinterpret_cast<float*>(mb + out_off);
                if (n_entries > 0) {
                    for (int t = 0; t < T; ++t) q2_0::quantize_q8(reinterpret_cast<const float*>(mb + x_off) + size_t(t) * n, act[t]);
                    q2_0::moe_cpu(*h->pool, es, miss.data(), n_entries, act.data(), T, out, n, scratch.data());
                    for (int i = 0; i < n_entries; ++i) slot_of[entry_expert[i]] = -1;
                } else {
                    std::memset(out, 0, size_t(T) * n * 4);
                }
                const auto t2 = std::chrono::steady_clock::now();
                h->hits += nh;
                h->misses += nm;
                h->wait_s += std::chrono::duration<double>(t1 - t0).count();
                const double dt = std::chrono::duration<double>(t2 - t1).count();
                h->cpu_s += dt;
                h->cpu_by_nm[std::min(n_entries, 16)] += dt;
                ++h->layers_by_nm[std::min(n_entries, 16)];
                std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(mb + kMbDone)).store(seq, std::memory_order_release);
            }
            cur_phase.store(0, std::memory_order_relaxed);
            served.store(seq, std::memory_order_release);   // h->access and the statistics are complete
        }
    }
};

void start_doorbell(const Spec& s, MoeFastHost& h, int cpu) {
    if (s.top_k > kMaxK) throw std::runtime_error("start_doorbell: top-k too large");
    h.mbox_stride = (mb_out_off(s.d_model, h.max_window) + size_t(h.max_window) * s.d_model * 4 + 4095) & ~size_t(4095);
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

void doorbell_begin_token(MoeFastHost& h, int T) {
    if (T < 1 || T > h.max_window) throw std::runtime_error("doorbell_begin_token: window longer than the mailboxes");
    h.window_T = T;
    ++h.seq;
    h.server->post.store(h.seq, std::memory_order_release);
    h.server->post.notify_one();
}

void doorbell_end_token(MoeFastHost& h, const Spec& s) {
    for (int il = 0; il < s.n_layer; ++il) {
        const uint32_t err =
            std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(h.mbox + size_t(il) * h.mbox_stride + kMbErr)).load(std::memory_order_acquire);
        if (err) {
            h.db_failed = true;
            throw std::runtime_error("doorbell: the GPU gave up waiting for layer " + std::to_string(il) + " of token " +
                                     std::to_string(err) + "; miss server at layer " + std::to_string(h.server->cur_layer.load()) +
                                     ", phase " + std::to_string(h.server->cur_phase.load()) + ", served " +
                                     std::to_string(h.server->served.load()) + ", posted " + std::to_string(h.server->post.load()) +
                                     (h.server->failed.load() ? ", server failed: " + h.server->error : std::string()));
        }
    }
    // the GPU has consumed every done flag; the server's bookkeeping after the last one is brief
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; h.server->served.load(std::memory_order_acquire) != h.seq; ++k) {
        if (h.server->failed.load()) {
            h.db_failed = true;
            throw std::runtime_error(h.server->error);
        }
        if ((k & 4095) == 4095 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
            h.db_failed = true;
            throw std::runtime_error("doorbell: the miss server did not finish token " + std::to_string(h.seq) + " in 10 s (layer " +
                                     std::to_string(h.server->cur_layer.load()) + ", phase " + std::to_string(h.server->cur_phase.load()) +
                                     ", served " + std::to_string(h.server->served.load()) + ")");
        }
    }
}

bool doorbell_failed(const MoeFastHost& h) { return h.db_failed || (h.server && h.server->failed.load()); }

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

void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out, int T) {
    const Spec& s = c.s;
    const int n = s.d_model, E = s.n_expert, K = s.top_k, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (K > kMaxK || E > 1024) throw std::runtime_error("moe_block_fast: unsupported routing shape");
    if (T < 1 || T > h.max_window || (T > 1 && !h.doorbell)) throw std::runtime_error("moe_block_fast: window needs doorbell mode and fitting mailboxes");
    // device scratch layout (per-token blocks back to back)
    float* logits = c.scratch.f32;                 // [T][E]
    float* sg = logits + size_t(T) * E;            // [T][ffs]
    float* su = sg + size_t(T) * ffs;              // [T][ffs]
    float* sh = su + size_t(T) * ffs;              // [T][n]
    float* gate = sh + size_t(T) * n;              // [T] (padded to 8)
    float* yh = gate + 8;                          // [T][K][n]
    float* cpu_dev = yh + size_t(T) * K * n;       // [n]
    float* hit_w = cpu_dev + n;                    // [T][K]
    const uint8_t** hit_ptr = reinterpret_cast<const uint8_t**>(hit_w + size_t(T) * kMaxK);   // [T][K] (8-byte aligned: n even)
    int32_t* route_dev = reinterpret_cast<int32_t*>(hit_ptr + size_t(T) * kMaxK);             // [T][kRouteInts]
    if (reinterpret_cast<uintptr_t>(hit_ptr) % 8) throw std::runtime_error("moe_block_fast: misaligned scratch");
    int32_t* hit_n = route_dev + size_t(T) * kRouteInts;                // [T] (padded to 8)
    int32_t* miss_n = hit_n + 8;                                        // [T] (padded to 8)
    float* hits_scratch = reinterpret_cast<float*>(miss_n + 8);         // moe_hits_scratch_bytes
    if (size_t(hits_scratch - c.scratch.f32) * 4 + moe_hits_scratch_bytes(K, ff, T) > c.scratch.f32_elems * 4)
        throw std::runtime_error("moe_block_fast: scratch too small");

    // 1. routing (with the shared expert's gate logit when both are BF16), and the input for the CPU
    const LinearOut rg[2] = {{&c.w.layer(il, "ffn_gate_inp.weight"), logits}, {&c.w.layer(il, "ffn_gate_inp_shexp.weight"), gate}};
    const bool rg_fused = linear_multi_ok(rg, 2, T);
    if (rg_fused) linear_multi(c, rg, 2, x, T);
    else linear(c, *rg[0].W, x, logits, T);
    uint8_t* mb = h.doorbell ? h.mbox_dev + size_t(il) * h.mbox_stride : nullptr;
    k_route<<<T, ((E + 31) / 32) * 32, 0, c.stream>>>(logits, cache.table_dev + size_t(il) * E, E, K, cache.slots, cache.slot_bytes,
                                                     hit_ptr, hit_w, hit_n, h.arena_dev, h.arena_stride, il, h.pcie_frac, h.pcie_max,
                                                     route_dev, x, n, mb, mb_x_off(h.max_window), h.seq, c.dparams, miss_n);
    if (!h.doorbell) {
        ck(cudaMemcpyAsync(h.route_host, route_dev, size_t(kRouteInts) * 4, cudaMemcpyDeviceToHost, c.stream), "route to host");
        ck(cudaMemcpyAsync(h.x_host, x, size_t(n) * 4, cudaMemcpyDeviceToHost, c.stream), "x to host");
        ck(cudaEventRecord(h.routed, c.stream), "event");
    }

    // 2. GPU: cache hits (fused gate+up, then down) and the shared expert
    moe_hits(hit_ptr, hit_n, K, x, n, ff, hits_scratch, yh, c.stream, T);
    {
        const GpuTensor* ws[2] = {&c.w.layer(il, "ffn_gate_shexp.weight"), &c.w.layer(il, "ffn_up_shexp.weight")};
        float* ys[2] = {sg, su};
        linear_shared(c, ws, ys, 2, x, T);
    }
    const GpuTensor& w_down = c.w.layer(il, "ffn_down_shexp.weight");
    if (q8_act(w_down, T)) {   // SwiGLU writes the down projection's activations
        gemv::swiglu_q8_1(sg, su, ffs, T, c.scratch.q8, c.stream);
        gemv::matvec_q(w_down.type, w_down.dev, c.scratch.q8, sh, ffs, n, T, c.stream);
    } else {
        k_swiglu_1<<<(T * ffs + 255) / 256, 256, 0, c.stream>>>(sg, su, T * ffs);
        linear(c, w_down, sg, sh, T);
    }
    if (!rg_fused) linear(c, *rg[1].W, x, gate, T);

    if (h.doorbell) {   // 3'. the miss server fills the mailbox; the combine waits for it on the GPU
        k_moe_combine_db<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(out, yh, hit_w, hit_n, mb, mb_out_off(n, h.max_window), h.seq,
                                                                        sh, gate, n, K, c.dparams, miss_n);
        ck(cudaGetLastError(), "moe_block_fast");
        return;
    }

    // 3. CPU: the misses, while the GPU works
    const auto t0 = std::chrono::steady_clock::now();
    ck(cudaEventSynchronize(h.routed), "wait routing");
    const auto t1 = std::chrono::steady_clock::now();
    const int nh = h.route_host[0], nm = h.route_host[1];
    h.gpu_misses += h.route_host[kRouteGpu];
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
