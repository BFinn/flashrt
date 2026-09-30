// SPDX-License-Identifier: Apache-2.0
#include "kernels/cuda/moe_q2.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flashrt::moe_q2 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int kJ = 64;                       // tokens per tile
constexpr int kRT = 128;                     // weight rows per CTA
constexpr int kKB = 2;                       // 64-weight blocks per pipeline stage
constexpr int kThreads = 256;                // 8 warps: rows x tokens per warp = 1,024 (AB x 1024 / AB)
constexpr int kActStride = kKB * 64 + 32;    // bytes per token in shared: 32 mod 128, so the 8-byte fragment loads do not conflict
constexpr int kMagic = 0x4B400000;           // bits of 1.5 * 2^23: int_as_float(kMagic + i) = 12582912 + i for |i| < 2^22
constexpr float kMagicF = 12582912.0f;
constexpr int kGroupTokens = 512, kMaxExperts = 4096;

// Activations are int8 in scale blocks of AB (32 or 64) with a float scale d and m = kMagic - (sum
// of the block's codes), stored together as float2 {d, m's bits} (one load per block pair; sw83). Within each 32 elements, element 16w + 4c + i sits at byte 8c + 4w + i,
// so lane c's two B fragment registers (elements 4c.. and 16 + 4c..) are one 8-byte load.
__device__ __forceinline__ int perm32(int j) { return 8 * ((j & 15) >> 2) + 4 * (j >> 4) + (j & 3); }
__device__ __forceinline__ int perm(int j) { return (j & ~31) + perm32(j & 31); }

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void mma_s8_acc(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}

// x [rows][kdim] -> scale blocks of AB (block w = row * kdim / AB + b): one warp per block
template <int AB>
__global__ void k_quant(const float* x, int n_blocks, int8_t* q, float2* dm) {
    constexpr int V = AB / 32;
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= n_blocks) return;
    float v[V], amax = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] = x[size_t(w) * AB + 32 * i + lane];
        amax = fmaxf(amax, fabsf(v[i]));
    }
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const int qi = int(roundf(v[i] * id));
        sum += qi;
        q[size_t(w) * AB + 32 * i + perm32(lane)] = int8_t(qi);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    if (lane == 0) dm[w] = make_float2(dd, __int_as_float(kMagic - sum));
}

// Grouping of the (token, slot) pairs by expert, stable in token order (as in ggml_gemm.cu):
// per-block histograms, one scan, one placement warp per block. row r of the compact order holds
// slot slot_of[r] = t * K + k of token tok_of[r] = t.
__global__ void k_hist(const int32_t* ids, int T, int K, int E, int32_t* cnt) {
    __shared__ int32_t h[kMaxExperts];
    for (int e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
    __syncthreads();
    const int s0 = blockIdx.x * kGroupTokens * K, s1 = min(T, (blockIdx.x + 1) * kGroupTokens) * K;
    for (int sl = s0 + threadIdx.x; sl < s1; sl += blockDim.x) atomicAdd(&h[ids[sl]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[size_t(blockIdx.x) * E + e] = h[e];
}

// block-wide exclusive scan (blockDim a multiple of 32); returns the thread's offset, sets *total
__device__ int block_excl_scan(int v, int* total) {
    __shared__ int ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int incl = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(~0u, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) ws[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < nw ? ws[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(~0u, w, o);
            if (lane >= o) w += n;
        }
        ws[lane] = w;
    }
    __syncthreads();
    const int r = (warp > 0 ? ws[warp - 1] : 0) + incl - v;
    *total = ws[nw - 1];
    __syncthreads();
    return r;
}

// One block of 1024 threads: bounds, each block's first row per expert, and the tile list
// (expert, first token) in expert order.
__global__ void k_scan_tiles(int32_t* cnt, int nb, int E, int32_t* bounds, int2* tiles, int* n_tiles) {
    __shared__ int32_t base[kMaxExperts];
    int run_rows = 0, run_tiles = 0;
    for (int e0 = 0; e0 < E; e0 += blockDim.x) {
        const int e = e0 + threadIdx.x;
        int c = 0;
        if (e < E)
            for (int b = 0; b < nb; ++b) c += cnt[size_t(b) * E + e];
        int tot_rows, tot_tiles;
        const int off = block_excl_scan(c, &tot_rows) + run_rows;
        const int nt = (c + kJ - 1) / kJ;
        const int toff = block_excl_scan(nt, &tot_tiles) + run_tiles;
        if (e < E) {
            bounds[e] = off;
            base[e] = off;
            for (int i = 0; i < nt; ++i) tiles[toff + i] = make_int2(e, i * kJ);
        }
        run_rows += tot_rows;
        run_tiles += tot_tiles;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bounds[E] = run_rows;
        *n_tiles = run_tiles;
    }
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        int run = base[e];
        for (int b = 0; b < nb; ++b) {
            const int c = cnt[size_t(b) * E + e];
            cnt[size_t(b) * E + e] = run;
            run += c;
        }
    }
}

__global__ void k_place(const int32_t* ids, int T, int K, int E, const int32_t* cnt, int32_t* tok_of, int32_t* slot_of) {
    __shared__ int32_t next[kMaxExperts];
    for (int e = threadIdx.x; e < E; e += blockDim.x) next[e] = cnt[size_t(blockIdx.x) * E + e];
    __syncwarp();
    const int lane = threadIdx.x;
    const int s0 = blockIdx.x * kGroupTokens * K, s1 = min(T, (blockIdx.x + 1) * kGroupTokens) * K;
    for (int b = s0; b < s1; b += 32) {
        const int sl = b + lane;
        const int e = sl < s1 ? ids[sl] : -1 - lane;
        const unsigned peers = __match_any_sync(~0u, e);
        if (e >= 0) {
            const int row = next[e] + __popc(peers & ((1u << lane) - 1));
            tok_of[row] = sl / K;
            slot_of[row] = sl;
        }
        __syncwarp();
        if (e >= 0 && lane == __ffs(peers) - 1) next[e] += __popc(peers);
        __syncwarp();
    }
}

// One 64-weight block of a warp's tile: acc[mat][m][n] += weight rows wr * AB + 16m .. (of each
// matrix) x tokens wt * 1024 / AB + 8n .. over the block. w: the block's codes of row 0, rows 16
// bytes apart, matrices wmat bytes apart; ws: each row's fp16 scales of a pair of blocks (the
// high half when kb is odd), matrices kRT apart; a: token 0's 64 bytes of the block, tokens
// astride apart; sdm: token 0's {scale, magic sum} of the block, tokens sstride apart.
// The codes enter the MMA unsigned; with AB == 64 the scales are constant over the block, so both
// k-steps accumulate in int32 and each element takes one conversion and one scaled add; with
// AB == 32, two conversions.
template <int NMAT, int AB>
__device__ __forceinline__ void block_mma(float (&acc)[NMAT][AB / 16][1024 / AB / 8][4], const uint8_t* w, int wmat, const uint32_t* ws,
                                          int kb, const uint8_t* a, int astride, const float2* sdm, int sstride, int wr, int wt, int g,
                                          int c) {
    constexpr int RW = AB, TW = 1024 / AB, MT = RW / 16, NT = TW / 8, SB = 64 / AB;
    uint2 bf[NT][2];
    float da[NT][2][SB];   // [n][token 2c + tc][scale block]
    int ma[NT][2][SB];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const uint8_t* ar = a + (wt * TW + 8 * n + g) * astride + 8 * c;
        bf[n][0] = *reinterpret_cast<const uint2*>(ar);
        bf[n][1] = *reinterpret_cast<const uint2*>(ar + 32);
#pragma unroll
        for (int tc = 0; tc < 2; ++tc) {
            const int j = wt * TW + 8 * n + 2 * c + tc;
            if constexpr (SB == 2) {   // 16-byte aligned: sstride and the offsets are even
                const float4 v = *reinterpret_cast<const float4*>(sdm + j * sstride);
                da[n][tc][0] = v.x;
                ma[n][tc][0] = __float_as_int(v.y);
                da[n][tc][1] = v.z;
                ma[n][tc][1] = __float_as_int(v.w);
            } else {
                const float2 v = sdm[j * sstride];
                da[n][tc][0] = v.x;
                ma[n][tc][0] = __float_as_int(v.y);
            }
        }
    }
#pragma unroll
    for (int mat = 0; mat < NMAT; ++mat)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int rl = wr * RW + 16 * m + g;
            const uint32_t wlo = *reinterpret_cast<const uint32_t*>(w + mat * wmat + rl * 16 + 4 * c);
            const uint32_t whi = *reinterpret_cast<const uint32_t*>(w + mat * wmat + (rl + 8) * 16 + 4 * c);
            constexpr uint32_t M3 = 0x03030303u;
            const uint32_t a0[4] = {wlo & M3, whi & M3, (wlo >> 2) & M3, (whi >> 2) & M3};
            const uint32_t a1[4] = {(wlo >> 4) & M3, (whi >> 4) & M3, (wlo >> 6) & M3, (whi >> 6) & M3};
            const uint32_t s_lo = ws[mat * kRT + rl], s_hi = ws[mat * kRT + rl + 8];
            const float dw0 = __half2float(__ushort_as_half(uint16_t(kb & 1 ? s_lo >> 16 : s_lo & 0xffff)));
            const float dw1 = __half2float(__ushort_as_half(uint16_t(kb & 1 ? s_hi >> 16 : s_hi & 0xffff)));
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if constexpr (AB == 32) {
                    int c0[4], c1[4];
                    mma_s8(c0, a0, bf[n][0].x, bf[n][0].y);
                    mma_s8(c1, a1, bf[n][1].x, bf[n][1].y);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int tc = q & 1;
                        const float f0 = __int_as_float(c0[q] + ma[n][tc][0]) - kMagicF;
                        const float f1 = __int_as_float(c1[q] + ma[n][tc][1]) - kMagicF;
                        const float t = fmaf(f1, da[n][tc][1], f0 * da[n][tc][0]);
                        acc[mat][m][n][q] = fmaf(t, q < 2 ? dw0 : dw1, acc[mat][m][n][q]);
                    }
                } else {
                    int cc[4];
                    mma_s8(cc, a0, bf[n][0].x, bf[n][0].y);
                    mma_s8_acc(cc, a1, bf[n][1].x, bf[n][1].y);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int tc = q & 1;
                        const float f = __int_as_float(cc[q] + ma[n][tc][0]) - kMagicF;
                        acc[mat][m][n][q] = fmaf(f, (q < 2 ? dw0 : dw1) * da[n][tc][0], acc[mat][m][n][q]);
                    }
                }
            }
        }
}

template <int NMAT, int AB>
struct Stage {
    uint8_t w[NMAT][kKB][kRT][16];      // codes, row-major per block
    uint32_t ws[NMAT][kRT];              // the two blocks' fp16 scales
    uint8_t a[kJ][kActStride];           // activations
    float2 adm[kJ][kKB * 64 / AB];       // activation {scale, kMagic - code sum}
};

// Gate and up: one CTA takes kRT rows of d_ff (of both matrices) x kJ tokens of one expert, the
// weights and the tokens' rows of x streaming through NST stages together (dynamic shared memory). The output
// h = silu(gate) * up is quantized for the down product (hq, hd, hm by compact row). AB: the
// activations' scale block; a warp takes AB weight rows (so the epilogue quantizes whole blocks)
// x 1024 / AB tokens.
template <int NMAT, int AB, int NST>
__global__ void __launch_bounds__(kThreads) k_moe_q2(const uint8_t* experts, size_t stride, size_t off0, size_t off1, int rows, int kdim,
                                                     const int8_t* aq, const float2* adm, const int32_t* a_row, const int32_t* bounds,
                                                     const int2* tiles, const int* n_tiles, int8_t* hq, float2* hdm, float* y,
                                                     const int32_t* slot_of) {
    constexpr int RW = AB, TW = 1024 / AB, MT = RW / 16, NT = TW / 8, RG = kRT / RW, SB = 64 / AB;
    extern __shared__ __align__(16) uint8_t smem[];
    Stage<NMAT, AB>* st = reinterpret_cast<Stage<NMAT, AB>*>(smem);
    __shared__ int32_t srow[kJ];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int2 tl = tiles[tile];
    const int e = tl.x, j0 = tl.y, p0 = bounds[e], ne = bounds[e + 1] - p0;
    const int r0 = blockIdx.x * kRT, nb = kdim / 64;
    const uint8_t* blob = experts + size_t(e) * stride;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int wr = warp % RG, wt = warp / RG;
    if (tid < kJ) {
        const int p = p0 + min(j0 + tid, ne - 1);
        srow[tid] = a_row ? a_row[p] : p;
    }
    __syncthreads();
    const int kblk = kdim / AB;
    const int n_it = nb / kKB;
    auto load = [&](int it) {   // always commits a group (empty past the end), so the waits count right
        if (it >= n_it) {
            asm volatile("cp.async.commit_group;\n" ::);
            return;
        }
        const int kb0 = it * kKB;
        Stage<NMAT, AB>& S = st[it % NST];
        for (int i = tid; i < NMAT * kRT * kKB; i += kThreads) {
            const int mat = i / (kRT * kKB), rem = i % (kRT * kKB), r = rem / kKB, kb = rem % kKB;
            const uint8_t* codes = blob + (mat ? off1 : off0);
            cp_async16(&S.w[mat][kb][r][0], codes + (size_t(r0 + r) * nb + kb0 + kb) * 16);
        }
        for (int i = tid; i < NMAT * kRT; i += kThreads) {
            const int mat = i / kRT, r = i % kRT;
            const uint8_t* codes = blob + (mat ? off1 : off0);
            cp_async4(&S.ws[mat][r], codes + size_t(rows) * nb * 16 + (size_t(r0 + r) * nb + kb0) * 2);
        }
        for (int i = tid; i < kJ * kKB * 4; i += kThreads) {
            const int j = i / (kKB * 4), q = i % (kKB * 4);
            cp_async16(&S.a[j][q * 16], aq + size_t(srow[j]) * kdim + kb0 * 64 + q * 16);
        }
        constexpr int DMQ = kKB * SB * 8 / 16;   // 16-byte pieces of {d, m} per token and stage
        for (int i = tid; i < kJ * DMQ; i += kThreads) {
            const int j = i / DMQ, q = i % DMQ;
            cp_async16(reinterpret_cast<uint8_t*>(&S.adm[j][0]) + q * 16,
                       reinterpret_cast<const uint8_t*>(adm + size_t(srow[j]) * kblk + kb0 * SB) + q * 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[NMAT][MT][NT][4];
#pragma unroll
    for (int mat = 0; mat < NMAT; ++mat)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[mat][m][n][0] = acc[mat][m][n][1] = acc[mat][m][n][2] = acc[mat][m][n][3] = 0.0f;
    for (int s = 0; s < NST - 1; ++s) load(s);
    for (int it = 0; it < n_it; ++it) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NST - 2));
        __syncthreads();   // this stage is in; the stage computed last is no longer read
        load(it + NST - 1);
        const Stage<NMAT, AB>& S = st[it % NST];
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb)
            block_mma<NMAT, AB>(acc, &S.w[0][kb][0][0], kKB * kRT * 16, &S.ws[0][0], kb, &S.a[0][kb * 64], kActStride, &S.adm[0][kb * SB],
                                kKB * SB, wr, wt, g, c);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();   // the epilogue reuses the stages

    // element q of acc[.][m][n]: weight row r0 + wr * RW + 16m + g + 8 (q >> 1), token wt * TW + 8n + 2c + (q & 1)
    static_assert(NMAT == 2, "k_moe_q2 is gate and up");
    {
        // h = silu(gate) * up; the warp's RW rows are one scale block of the down input: quantize per token
        uint8_t* sq = reinterpret_cast<uint8_t*>(&st[0]) + warp * 1024;   // [TW tokens][AB bytes]
        const int ffb = rows / AB, blk = (r0 + wr * RW) / AB;
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int tc = 0; tc < 2; ++tc) {
                float h[MT][2];   // [m][row g or g + 8]
                float amax = 0.0f;
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int hr = 0; hr < 2; ++hr) {
                        const float gv = acc[0][m][n][2 * hr + tc], uv = acc[1][m][n][2 * hr + tc];
                        h[m][hr] = gv / (1.0f + __expf(-gv)) * uv;
                        amax = fmaxf(amax, fabsf(h[m][hr]));
                    }
                for (int o = 4; o < 32; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
                const float dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f;
                const int lt = 8 * n + 2 * c + tc;
                int sum = 0;
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int hr = 0; hr < 2; ++hr) {
                        const int qi = int(roundf(h[m][hr] * id));
                        sum += qi;
                        sq[lt * AB + perm(16 * m + 8 * hr + g)] = uint8_t(int8_t(qi));
                    }
                for (int o = 4; o < 32; o <<= 1) sum += __shfl_xor_sync(~0u, sum, o);
                const int j = j0 + wt * TW + lt;
                if (g == 0 && j < ne) hdm[size_t(p0 + j) * ffb + blk] = make_float2(dd, __int_as_float(kMagic - sum));
            }
        __syncwarp();
        constexpr int PARTS = AB / 32;   // 32-byte pieces per token
        const int lt = lane / PARTS, part = lane % PARTS, j = j0 + wt * TW + lt;
        if (j < ne) {
            const uint4* src = reinterpret_cast<const uint4*>(sq + lt * AB + part * 32);
            uint4* dst = reinterpret_cast<uint4*>(hq + size_t(p0 + j) * rows + r0 + wr * RW + part * 32);
            dst[0] = src[0];
            dst[1] = src[1];
        }
    }
}

// Down: one CTA per tile (kJ tokens of one expert) over all of d_model. The tokens' activations
// (all of d_ff, by compact row) load into shared memory once, and the weights stream through
// kDownStages stages, row tile after row tile, so the prologue and the activation loads are paid
// once per tile, not once per 128 rows (d_ff is short: 10 weight blocks). y by slot, as float or
// BF16 (OutT).
constexpr int kDownStages = 3;
size_t down_smem_bytes(int kdim, int AB) {
    return size_t(kJ) * (kdim + 32) + size_t(kJ) * (kdim / AB) * 8 + size_t(kDownStages) * (kKB * kRT * 16 + kRT * 4);
}
template <int AB, typename OutT>
__global__ void __launch_bounds__(kThreads) k_moe_q2_down(const uint8_t* experts, size_t stride, size_t off, int rows, int kdim,
                                                          const int8_t* aq, const float2* adm, const int32_t* bounds, const int2* tiles,
                                                          const int* n_tiles, OutT* y, const int32_t* slot_of) {
    constexpr int RW = AB, TW = 1024 / AB, MT = RW / 16, NT = TW / 8, RG = kRT / RW, SB = 64 / AB;
    constexpr int WSTAGE = kKB * kRT * 16;   // code bytes per stage
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int32_t srow[kJ];
    const int tile = blockIdx.x;
    if (tile >= *n_tiles) return;
    const int2 tl = tiles[tile];
    const int e = tl.x, j0 = tl.y, p0 = bounds[e], ne = bounds[e + 1] - p0;
    const int astride = kdim + 32, kblk = kdim / AB, nb = kdim / 64, n_it = nb / kKB, steps = rows / kRT * n_it;
    uint8_t* sa = smem;                                                        // [kJ][astride]
    float2* sdm = reinterpret_cast<float2*>(sa + kJ * astride);                // [kJ][kblk] {d, m}
    uint8_t* sw = reinterpret_cast<uint8_t*>(sdm + kJ * kblk);                 // [stage][kKB][kRT][16]
    uint32_t* sws = reinterpret_cast<uint32_t*>(sw + kDownStages * WSTAGE);    // [stage][kRT]
    const uint8_t* codes = experts + size_t(e) * stride + off;
    const uint8_t* scales = codes + size_t(rows) * nb * 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int wr = warp % RG, wt = warp / RG;
    if (tid < kJ) srow[tid] = p0 + min(j0 + tid, ne - 1);
    __syncthreads();
    // the activations and their scales join the first weight stage's group
    for (int i = tid; i < kJ * (kdim / 16); i += kThreads) {
        const int j = i / (kdim / 16), q = i % (kdim / 16);
        cp_async16(sa + j * astride + q * 16, aq + size_t(srow[j]) * kdim + q * 16);
    }
    for (int i = tid; i < kJ * (kblk / 2); i += kThreads) {
        const int j = i / (kblk / 2), q = i % (kblk / 2);
        cp_async16(sdm + j * kblk + 2 * q, adm + size_t(srow[j]) * kblk + 2 * q);
    }
    auto load_w = [&](int step) {   // always commits a group (empty past the end), so the waits count right
        if (step < steps) {
            const int r0 = step / n_it * kRT, kb0 = step % n_it * kKB, buf = step % kDownStages;
            for (int i = tid; i < kRT * kKB; i += kThreads) {
                const int r = i / kKB, kb = i % kKB;
                cp_async16(sw + buf * WSTAGE + (kb * kRT + r) * 16, codes + (size_t(r0 + r) * nb + kb0 + kb) * 16);
            }
            for (int r = tid; r < kRT; r += kThreads) cp_async4(sws + buf * kRT + r, scales + (size_t(r0 + r) * nb + kb0) * 2);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    for (int s = 0; s < kDownStages - 1; ++s) load_w(s);
    float acc[1][MT][NT][4];
    auto zero = [&] {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[0][m][n][0] = acc[0][m][n][1] = acc[0][m][n][2] = acc[0][m][n][3] = 0.0f;
    };
    zero();
    for (int step = 0; step < steps; ++step) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kDownStages - 2));
        __syncthreads();   // this stage is in; the stage computed last is no longer read
        load_w(step + kDownStages - 1);
        const int buf = step % kDownStages, it = step % n_it;
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb) {
            const int kg = it * kKB + kb;
            block_mma<1, AB>(acc, sw + buf * WSTAGE + kb * kRT * 16, 0, sws + buf * kRT, kb, sa + kg * 64, astride, sdm + kg * SB, kblk,
                             wr, wt, g, c);
        }
        if (it == n_it - 1) {   // this row tile is done: element q of acc[0][m][n] is row r0 + wr * RW + 16m + g + 8 (q >> 1)
            const int r0 = step / n_it * kRT;
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = j0 + wt * TW + 8 * n + 2 * c + (q & 1);
                    if (j >= ne) continue;
                    OutT* yr = y + size_t(slot_of[p0 + j]) * rows + r0 + wr * RW + g + 8 * (q >> 1);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        if constexpr (std::is_same_v<OutT, float>) yr[16 * m] = acc[0][m][n][q];
                        else yr[16 * m] = __float2bfloat16(acc[0][m][n][q]);
                    }
                }
            zero();
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
}

struct Ws {
    int32_t *cnt, *bounds, *tok_of, *slot_of, *n_tiles;
    int2* tiles;
    int8_t *xq, *hq;
    float2 *xdm, *hdm;
    size_t bytes;
};
size_t up256(size_t x) { return (x + 255) & ~size_t(255); }
int max_tiles(int T, int K, int E) { return (T * K + kJ - 1) / kJ + E; }
Ws carve(void* base, int T, int K, int n, int ff, int E) {
    Ws w{};
    char* p = static_cast<char*>(base);
    auto take = [&](size_t b) {
        char* r = p;
        p += up256(b);
        return r;
    };
    const size_t S = size_t(T) * K, nbk = (T + kGroupTokens - 1) / kGroupTokens;
    w.cnt = reinterpret_cast<int32_t*>(take(nbk * E * 4));
    w.bounds = reinterpret_cast<int32_t*>(take(size_t(E + 1) * 4));
    w.tok_of = reinterpret_cast<int32_t*>(take(S * 4));
    w.slot_of = reinterpret_cast<int32_t*>(take(S * 4));
    w.n_tiles = reinterpret_cast<int32_t*>(take(4));
    w.tiles = reinterpret_cast<int2*>(take(size_t(max_tiles(T, K, E)) * 8));
    w.xq = reinterpret_cast<int8_t*>(take(size_t(T) * n));
    w.xdm = reinterpret_cast<float2*>(take(size_t(T) * n / 32 * 8));
    w.hq = reinterpret_cast<int8_t*>(take(S * ff));
    w.hdm = reinterpret_cast<float2*>(take(S * ff / 32 * 8));
    w.bytes = size_t(p - static_cast<char*>(base));
    return w;
}

}  // namespace

size_t workspace_bytes(int T, int K, int n, int ff, int E) { return carve(nullptr, T, K, n, ff, E).bytes; }

void run(const uint8_t* experts, size_t stride, int E, int n, int ff, const float* x, const int32_t* ids, int T, int K, void* yd,
         void* ws, size_t ws_bytes, cudaStream_t stream, bool block64, bool yd_bf16) {
    if (n % kRT || ff % kRT || n % (64 * kKB) || ff % (64 * kKB) || E > kMaxExperts || T < 1)
        throw std::runtime_error("moe_q2::run: unsupported shape");
    const Ws w = carve(ws, T, K, n, ff, E);
    if (w.bytes > ws_bytes) throw std::runtime_error("moe_q2::run: workspace too small");
    const int nbk = (T + kGroupTokens - 1) / kGroupTokens;
    k_hist<<<nbk, 256, 0, stream>>>(ids, T, K, E, w.cnt);
    k_scan_tiles<<<1, 1024, 0, stream>>>(w.cnt, nbk, E, w.bounds, w.tiles, w.n_tiles);
    k_place<<<nbk, 32, 0, stream>>>(ids, T, K, E, w.cnt, w.tok_of, w.slot_of);
    const size_t gu = size_t(ff) * (n / 64) * 18;   // q2_0::mat_bytes(ff, n)
    const int mt = max_tiles(T, K, E);
    auto go = [&](auto ab) {
        constexpr int AB = decltype(ab)::value;
        const int xb = T * n / AB;
        k_quant<AB><<<(xb + 7) / 8, 256, 0, stream>>>(x, xb, w.xq, w.xdm);
        constexpr int NST = 2;   // gate and up pipeline stages (3 and 4 were no faster: sw55)
        const size_t sm = sizeof(Stage<2, AB>) * NST;
        ck(cudaFuncSetAttribute(k_moe_q2<2, AB, NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sm)), "moe_q2 smem");
        k_moe_q2<2, AB, NST><<<dim3(ff / kRT, mt), kThreads, sm, stream>>>(experts, stride, 0, gu, ff, n, w.xq, w.xdm, w.tok_of, w.bounds,
                                                                          w.tiles, w.n_tiles, w.hq, w.hdm, nullptr, nullptr);
        const size_t smem = down_smem_bytes(ff, AB);   // above the 48 KB default: opt in (cheap; once per layer)
        auto down = [&](auto* y) {
            using OutT = std::remove_pointer_t<decltype(y)>;
            ck(cudaFuncSetAttribute(k_moe_q2_down<AB, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)), "moe_q2 down smem");
            k_moe_q2_down<AB, OutT><<<mt, kThreads, smem, stream>>>(experts, stride, 2 * gu, n, ff, w.hq, w.hdm, w.bounds, w.tiles,
                                                                   w.n_tiles, y, w.slot_of);
        };
        if (yd_bf16) down(static_cast<__nv_bfloat16*>(yd));
        else down(static_cast<float*>(yd));
    };
    if (block64) go(std::integral_constant<int, 64>{});
    else go(std::integral_constant<int, 32>{});
    ck(cudaGetLastError(), "moe_q2::run");
}

}  // namespace flashrt::moe_q2
