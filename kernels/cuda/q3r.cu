// SPDX-License-Identifier: Apache-2.0
#include "kernels/cuda/q3r.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace flashrt::q3r {

namespace {

constexpr int kQK = 256;
constexpr int kBlockBytes = 110;   // hmask[32] qs[64] scales[12] d (fp16)

size_t up256(size_t x) { return (x + 255) & ~size_t(255); }

struct Planes {
    uint32_t* low;
    uint16_t* high;
    int8_t* sc;
    __half* d;
};

__host__ __device__ inline Planes planes(void* base, int64_t rows, int64_t K) {
    uint8_t* p = static_cast<uint8_t*>(base);
    Planes q;
    q.low = reinterpret_cast<uint32_t*>(p);
    p += (size_t(rows) * K / 4 + 255) & ~size_t(255);
    q.high = reinterpret_cast<uint16_t*>(p);
    p += (size_t(rows) * K / 8 + 255) & ~size_t(255);
    q.sc = reinterpret_cast<int8_t*>(p);
    p += (size_t(rows) * K / 16 + 255) & ~size_t(255);
    q.d = reinterpret_cast<__half*>(p);
    return q;
}

// one thread per 16-element group (group gi of super-block sb of row r)
__global__ void k_repack(const uint8_t* src, Planes q, int64_t rows, int64_t K) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;   // global group
    const int64_t groups_per_row = K / 16;
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gr = g % groups_per_row;
    const int64_t sb = gr / 16;
    const int gi = int(gr % 16);
    const uint8_t* b = src + (r * (K / kQK) + sb) * kBlockBytes;
    const uint8_t* hmask = b;
    const uint8_t* qs = b + 32;
    const uint8_t* scales = b + 96;
    const int is = gi;   // the group's scale index (8n + 2j + l/16 == gi for e = 16 gi)
    const int us = is < 4 ? (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
                 : is < 8 ? (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
                 : is < 12 ? (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4)
                           : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    uint32_t low = 0;
    uint32_t high = 0;
    for (int i = 0; i < 16; ++i) {
        const int e = 16 * gi + i, n = e / 128, j = (e % 128) / 32, l = e % 32;
        low |= uint32_t((qs[32 * n + l] >> (2 * j)) & 3) << (2 * i);
        high |= uint32_t((hmask[l] >> (4 * n + j)) & 1) << i;
    }
    q.low[g] = low;
    q.high[g] = uint16_t(high);
    q.sc[g] = int8_t(us - 32);
    if (gi == 0) q.d[r * (K / kQK) + sb] = *reinterpret_cast<const __half*>(b + 108);
}

// 8 warps per block, one row per warp. Lane l takes groups l, l+32, ...; x sits in shared memory
// transposed (xt[i][g] = x[16 g + i]) so a warp's reads are consecutive, with each group's sum.
__global__ void k_matvec(Planes q, const float* x, float* y, int64_t rows, int64_t K) {
    extern __shared__ __align__(16) float xt[];   // [16][K/16] then group sums [K/16]
    const int64_t G = K / 16;
    float* xsum = xt + K;
    for (int64_t gidx = threadIdx.x; gidx < G; gidx += blockDim.x) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = x[16 * gidx + i];
            xt[i * G + gidx] = v;
            s += v;
        }
        xsum[gidx] = s;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t r = int64_t(blockIdx.x) * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const uint32_t* low = q.low + r * G;
    const uint16_t* high = q.high + r * G;
    const int8_t* sc = q.sc + r * G;
    const __half* d = q.d + r * (K / kQK);
    float acc = 0.0f;
#pragma unroll 4
    for (int64_t gidx = lane; gidx < G; gidx += 32) {
        const uint32_t lo = low[gidx];
        const uint32_t hi = high[gidx];
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s += float(((lo >> (2 * i)) & 3) | (((hi >> i) & 1) << 2)) * xt[i * G + gidx];
        acc += __half2float(d[gidx / 16]) * float(sc[gidx]) * (s - 4.0f * xsum[gidx]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) y[r] = acc;
}

// v2: each lane takes 2 consecutive groups (8-byte low, 4-byte high, 2-byte scale loads) of
// RPW rows (x reads shared across rows). Block of 8 warps = 8 * RPW rows. xt from shared memory
// (built per block) or, with XG, from a global copy built once per call by k_prep.
__global__ void k_prep(const float* x, float* xt, int64_t K) {
    const int64_t G = K / 16;
    for (int64_t gidx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; gidx < G; gidx += int64_t(gridDim.x) * blockDim.x) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = x[16 * gidx + i];
            xt[i * G + gidx] = v;
            s += v;
        }
        xt[K + gidx] = s;
    }
}

template <int RPW, bool XG>
__global__ void k_matvec2(Planes q, const float* x, const float* xt_g, float* y, int64_t rows, int64_t K) {
    extern __shared__ __align__(16) float xt_s[];
    const int64_t G = K / 16;
    const float* xt;
    if (XG) {
        xt = xt_g;
    } else {
        for (int64_t gidx = threadIdx.x; gidx < G; gidx += blockDim.x) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float v = x[16 * gidx + i];
                xt_s[i * G + gidx] = v;
                s += v;
            }
            xt_s[K + gidx] = s;
        }
        __syncthreads();
        xt = xt_s;
    }
    const float* xsum = xt + K;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t r0 = (int64_t(blockIdx.x) * (blockDim.x >> 5) + warp) * RPW;
    float acc[RPW];
#pragma unroll
    for (int k = 0; k < RPW; ++k) acc[k] = 0.0f;
    for (int64_t g2 = 2 * lane; g2 < G; g2 += 64) {   // groups g2, g2 + 1 (G is even: K % 256 == 0)
        uint2 lo[RPW];
        uint32_t hi[RPW];
        uint32_t scv[RPW];
        float dv[RPW];
#pragma unroll
        for (int k = 0; k < RPW; ++k) {
            const int64_t r = min(r0 + k, rows - 1);
            lo[k] = *reinterpret_cast<const uint2*>(q.low + r * G + g2);
            hi[k] = *reinterpret_cast<const uint32_t*>(q.high + r * G + g2);
            scv[k] = *reinterpret_cast<const uint16_t*>(q.sc + r * G + g2);
            dv[k] = __half2float(q.d[r * (K / kQK) + g2 / 16]);
        }
        float s0[RPW], s1[RPW];
#pragma unroll
        for (int k = 0; k < RPW; ++k) s0[k] = s1[k] = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float2 xv = *reinterpret_cast<const float2*>(xt + i * G + g2);
#pragma unroll
            for (int k = 0; k < RPW; ++k) {
                s0[k] += float(((lo[k].x >> (2 * i)) & 3) | (((hi[k] >> i) & 1) << 2)) * xv.x;
                s1[k] += float(((lo[k].y >> (2 * i)) & 3) | (((hi[k] >> (16 + i)) & 1) << 2)) * xv.y;
            }
        }
        const float xs0 = xsum[g2], xs1 = xsum[g2 + 1];
#pragma unroll
        for (int k = 0; k < RPW; ++k)
            acc[k] += dv[k] * (float(int8_t(scv[k] & 0xff)) * (s0[k] - 4.0f * xs0) + float(int8_t(scv[k] >> 8)) * (s1[k] - 4.0f * xs1));
    }
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
        float a = acc[k];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (lane == 0 && r0 + k < rows) y[r0 + k] = a;
    }
}

float* g_xt = nullptr;   // k_prep's output, allocated on first use (K <= 11264)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("q3r ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

size_t bytes(int64_t rows, int64_t K) {
    return up256(size_t(rows) * K / 4) + up256(size_t(rows) * K / 8) + up256(size_t(rows) * K / 16) +
           up256(size_t(rows) * (K / kQK) * 2);
}

void repack(const void* q3k, void* q3r, int64_t rows, int64_t K, cudaStream_t stream) {
    if (K % kQK) throw std::runtime_error("q3r::repack: K must be a multiple of 256");
    const int64_t groups = rows * K / 16;
    k_repack<<<unsigned((groups + 255) / 256), 256, 0, stream>>>(static_cast<const uint8_t*>(q3k), planes(q3r, rows, K), rows, K);
    ck(cudaGetLastError(), "repack");
}

void matvec_variant(int v, const void* q3r, const float* x, float* y, int64_t rows, int64_t K, cudaStream_t stream) {
    const Planes pl = planes(const_cast<void*>(q3r), rows, K);
    const size_t smem = size_t(K + K / 16) * 4;
    if (v == 1) {
        k_matvec<<<unsigned((rows + 7) / 8), 256, smem, stream>>>(pl, x, y, rows, K);
    } else if (v == 2) {
        k_matvec2<2, false><<<unsigned((rows + 15) / 16), 256, smem, stream>>>(pl, x, nullptr, y, rows, K);
    } else if (v == 3 || v == 4) {
        if (!g_xt) ck(cudaMalloc(&g_xt, size_t(11264 + 11264 / 16) * 4), "cudaMalloc xt");
        k_prep<<<unsigned((K / 16 + 127) / 128), 128, 0, stream>>>(x, g_xt, K);
        if (v == 3) k_matvec2<2, true><<<unsigned((rows + 15) / 16), 256, 0, stream>>>(pl, x, g_xt, y, rows, K);
        else k_matvec2<4, true><<<unsigned((rows + 31) / 32), 256, 0, stream>>>(pl, x, g_xt, y, rows, K);
    } else if (v == 5) {
        k_matvec2<1, false><<<unsigned((rows + 7) / 8), 256, smem, stream>>>(pl, x, nullptr, y, rows, K);
    }
    ck(cudaGetLastError(), "matvec");
}

void matvec(const void* q3r, const float* x, float* y, int64_t rows, int64_t K, cudaStream_t stream) {
    if (K % kQK || K > 11264) throw std::runtime_error("q3r::matvec: unsupported K");
    const size_t smem = size_t(K + K / 16) * 4;
    k_matvec<<<unsigned((rows + 7) / 8), 256, smem, stream>>>(planes(const_cast<void*>(q3r), rows, K), x, y, rows, K);
    ck(cudaGetLastError(), "matvec");
}

}  // namespace flashrt::q3r
