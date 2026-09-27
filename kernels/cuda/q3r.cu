// SPDX-License-Identifier: Apache-2.0
#include "kernels/cuda/q3r.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace flashrt::q3r {

namespace {

constexpr int kQK = 256;           // Q3_K super-block
constexpr int kBlockBytes = 110;   // hmask[32] qs[64] scales[12] d (fp16)
constexpr int kTok = 8;            // tokens per pass of the mat-vec

size_t up256(size_t x) { return (x + 255) & ~size_t(255); }

struct Planes {
    uint8_t* lo;
    uint8_t* hi;
    int8_t* sc;
    __half* d;
};

__host__ __device__ inline Planes planes(void* base, int64_t rows, int64_t K) {
    uint8_t* p = static_cast<uint8_t*>(base);
    Planes q;
    q.lo = p;
    p += (size_t(rows) * K / 4 + 255) & ~size_t(255);
    q.hi = p;
    p += (size_t(rows) * K / 8 + 255) & ~size_t(255);
    q.sc = reinterpret_cast<int8_t*>(p);
    p += (size_t(rows) * K / 16 + 255) & ~size_t(255);
    q.d = reinterpret_cast<__half*>(p);
    return q;
}

// Q3_K element e (0..255) of a super-block: its 3-bit q and its group scale index
__device__ inline int q3k_q(const uint8_t* b, int e) {
    const uint8_t* hmask = b;
    const uint8_t* qs = b + 32;
    const int n = e / 128, j = (e % 128) / 32, l = e % 32;
    return ((qs[32 * n + l] >> (2 * j)) & 3) | (((hmask[l] >> (4 * n + j)) & 1) << 2);
}

// one thread per 16 bytes of the lo plane (one 64-block) of one row
__global__ void k_repack(const uint8_t* src, Planes q, int64_t rows, int64_t K) {
    const int64_t nb = K / 64;
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;   // (row, 64-block)
    if (g >= rows * nb) return;
    const int64_t r = g / nb, b = g % nb;
    const uint8_t* sb = src + (r * (K / kQK) + b / 4) * kBlockBytes;
    const int e0 = int(b % 4) * 64;
    uint8_t lo[16] = {}, hi[16] = {};
    for (int j = 0; j < 64; ++j) {
        const int q = q3k_q(sb, e0 + j);
        lo[j % 16] |= uint8_t((q & 3) << (2 * (j / 16)));
        hi[j % 16] |= uint8_t(((q >> 2) & 1) << ((b % 2) * 4 + j / 16));
    }
    uint8_t* dlo = q.lo + (r * nb + b) * 16;
    for (int i = 0; i < 16; ++i) dlo[i] = lo[i];
    // the hi plane is shared by the block pair: the odd block ORs into what the even one wrote
    // would race, so the even block's thread writes both halves
    if (b % 2 == 0) {
        uint8_t hi2[16] = {};
        if (b + 1 < nb) {
            const uint8_t* sb2 = src + (r * (K / kQK) + (b + 1) / 4) * kBlockBytes;
            const int e1 = int((b + 1) % 4) * 64;
            for (int j = 0; j < 64; ++j) hi2[j % 16] |= uint8_t(((q3k_q(sb2, e1 + j) >> 2) & 1) << (4 + j / 16));
        }
        uint8_t* dhi = q.hi + (r * (nb / 2) + b / 2) * 16;
        for (int i = 0; i < 16; ++i) dhi[i] = hi[i] | hi2[i];
    }
    // scales of the block's 4 groups, and d once per super-block
    const uint8_t* scales = sb + 96;
    for (int gi = 0; gi < 4; ++gi) {
        const int is = int(b % 4) * 4 + gi;
        const int us = is < 4 ? (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
                     : is < 8 ? (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
                     : is < 12 ? (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4)
                               : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
        q.sc[r * (K / 16) + b * 4 + gi] = int8_t(us - 32);
    }
    if (b % 4 == 0) q.d[r * (K / kQK) + b / 4] = *reinterpret_cast<const __half*>(sb + 108);
}

// y[t][r] for ts (<= kTok) tokens at a time: 16 warps x 4 rows, 8 lanes per row (64 rows per block).
// Activations of the pass's tokens are quantized per 64 values into shared memory, words in
// "planar word order" (word 4p + ic holds elements 16p + 4ic .. +3) laid out [t][w][nb].
__global__ void k_matvec(Planes q, const float* x, float* y, int64_t rows, int64_t K, int T, int ts) {
    extern __shared__ __align__(16) uint32_t smem[];
    const int nb = int(K / 64), nt = min(ts, T - int(blockIdx.y) * ts), t0 = blockIdx.y * ts;
    uint32_t* xw = smem;                                             // [ts][16][nb]
    float* xscale = reinterpret_cast<float*>(xw + ts * 16 * nb);     // [ts][nb]
    int* xsum = reinterpret_cast<int*>(xscale + ts * nb);            // [ts][nb][4] (per 16-group)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;
    for (int item = warp; item < nt * nb; item += nwarp) {
        const int t = item / nb, b = item % nb;
        const float* xb = x + size_t(t0 + t) * K + size_t(b) * 64;
        const float v0 = xb[lane], v1 = xb[lane + 32];
        float am = fmaxf(fabsf(v0), fabsf(v1));
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
        const float inv = am > 0.0f ? 127.0f / am : 0.0f;
        const int q0 = __float2int_rn(v0 * inv), q1 = __float2int_rn(v1 * inv);
        int8_t* wb = reinterpret_cast<int8_t*>(xw + size_t(t) * 16 * nb);
        const int j0 = lane, j1 = lane + 32;
        wb[((j0 / 16 * 4 + (j0 % 16) / 4) * nb + b) * 4 + j0 % 4] = int8_t(q0);
        wb[((j1 / 16 * 4 + (j1 % 16) / 4) * nb + b) * 4 + j1 % 4] = int8_t(q1);
        // per 16-group sums: lanes 0-15 are group 0 (q0) / group 2 (q1), lanes 16-31 group 1 / 3
        int s0 = q0, s1 = q1;
        for (int o = 8; o > 0; o >>= 1) {
            s0 += __shfl_xor_sync(0xffffffff, s0, o);
            s1 += __shfl_xor_sync(0xffffffff, s1, o);
        }
        if (lane == 0) {
            xscale[t * nb + b] = am / 127.0f;
            xsum[(t * nb + b) * 4 + 0] = s0;
            xsum[(t * nb + b) * 4 + 2] = s1;
        }
        if (lane == 16) {
            xsum[(t * nb + b) * 4 + 1] = s0;
            xsum[(t * nb + b) * 4 + 3] = s1;
        }
    }
    __syncthreads();
    const int rsub = lane >> 3, l8 = lane & 7;
    const int64_t r = int64_t(blockIdx.x) * 64 + warp * 4 + rsub;
    if (r >= rows) return;
    const uint4* lo = reinterpret_cast<const uint4*>(q.lo) + r * nb;
    const uint4* hi = reinterpret_cast<const uint4*>(q.hi) + r * (nb / 2);
    const uint32_t* sc = reinterpret_cast<const uint32_t*>(q.sc + r * (K / 16));   // 4 group scales per 64-block
    const __half* d = q.d + r * (K / kQK);
    float acc[kTok];
#pragma unroll
    for (int t = 0; t < kTok; ++t) acc[t] = 0.0f;
    for (int b = l8; b < nb; b += 8) {
        const uint4 wl = lo[b], wh = hi[b / 2];
        const uint32_t scw = sc[b];
        const float db = __half2float(d[b / 4]);
        const int hs = (b % 2) * 4;
        const uint32_t lw[4] = {wl.x, wl.y, wl.z, wl.w}, hw[4] = {wh.x, wh.y, wh.z, wh.w};
        uint32_t v[16];   // q of elements 16p + 4ic .. +3 as bytes, v[4p + ic]
#pragma unroll
        for (int p = 0; p < 4; ++p)
#pragma unroll
            for (int ic = 0; ic < 4; ++ic)
                v[p * 4 + ic] = ((lw[ic] >> (2 * p)) & 0x03030303u) | (((hw[ic] >> (hs + p)) & 0x01010101u) << 2);
        for (int t = 0; t < nt; ++t) {
            const uint32_t* xt = xw + size_t(t) * 16 * nb;
            const int* xs = xsum + (t * nb + b) * 4;
            float blk = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int a = 0;
#pragma unroll
                for (int ic = 0; ic < 4; ++ic) a = __dp4a(int(v[p * 4 + ic]), int(xt[(p * 4 + ic) * nb + b]), a);
                blk += float(int8_t(scw >> (8 * p))) * float(a - 4 * xs[p]);
            }
            acc[t] += db * xscale[t * nb + b] * blk;
        }
    }
#pragma unroll
    for (int t = 0; t < kTok; ++t) {
        float a = acc[t];
        for (int o = 4; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (l8 == 0 && t < nt) y[size_t(t0 + t) * rows + r] = a;
    }
}

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
    const int64_t items = rows * (K / 64);
    k_repack<<<unsigned((items + 127) / 128), 128, 0, stream>>>(static_cast<const uint8_t*>(q3k), planes(q3r, rows, K), rows, K);
    ck(cudaGetLastError(), "repack");
}

void matvec(const void* q3r, const float* x, float* y, int64_t rows, int64_t K, int T, cudaStream_t stream) {
    if (K % kQK || K > 8192 || T < 1) throw std::runtime_error("q3r::matvec: unsupported shape");
    const int nb = int(K / 64), ts = T < kTok ? T : kTok;   // tokens per pass
    const size_t smem = size_t(ts) * nb * (16 * 4 + 4 + 16);
    static bool attr_set = false;
    if (!attr_set) {
        ck(cudaFuncSetAttribute(k_matvec, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024), "smem attribute");
        attr_set = true;
    }
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((T + ts - 1) / ts));
    k_matvec<<<grid, 512, smem, stream>>>(planes(const_cast<void*>(q3r), rows, K), x, y, rows, K, T, ts);
    ck(cudaGetLastError(), "matvec");
}

}  // namespace flashrt::q3r
