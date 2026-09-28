// SPDX-License-Identifier: Apache-2.0
#include "kernels/cuda/moe_q2.h"

#include <cuda_fp16.h>

#include <algorithm>
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
// of the block's codes). Within each 32 elements, element 16w + 4c + i sits at byte 8c + 4w + i,
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
__global__ void k_quant(const float* x, int n_blocks, int8_t* q, float* d, int32_t* m) {
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
    if (lane == 0) {
        d[w] = dd;
        m[w] = kMagic - sum;
    }
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

template <int NMAT, int AB>
struct Stage {
    uint8_t w[NMAT][kKB][kRT][16];      // codes, row-major per block
    uint32_t ws[NMAT][kRT];              // the two blocks' fp16 scales
    uint8_t a[kJ][kActStride];           // activations
    float ad[kJ][kKB * 64 / AB];         // activation scales
    int32_t am[kJ][kKB * 64 / AB];       // kMagic - code sums
};

// One CTA: kRT weight rows x kJ tokens of one expert. NMAT == 2: gate and up (rows of d_ff), the
// output h = silu(gate) * up quantized for the down product (hq, hd, hm by compact row); the
// activations are the tokens' rows of x. NMAT == 1: down (rows of d_model), y by slot; the
// activations are h by compact row. AB: the activations' scale block. A warp takes AB weight
// rows (so the gate/up epilogue quantizes whole blocks) x 1024 / AB tokens. With AB == 64 the
// scales are constant over a weight block, so the two k-steps accumulate in int32 and each
// element takes one conversion and one scaled add per block; with AB == 32, two.
template <int NMAT, int AB>
__global__ void __launch_bounds__(kThreads, NMAT == 1 ? 2 : 1) k_moe_q2(const uint8_t* experts, size_t stride, size_t off0, size_t off1, int rows, int kdim,
                                                     const int8_t* aq, const float* ad, const int32_t* am, const int32_t* a_row,
                                                     const int32_t* bounds, const int2* tiles, const int* n_tiles, int8_t* hq, float* hd,
                                                     int32_t* hm, float* y, const int32_t* slot_of) {
    constexpr int RW = AB, TW = 1024 / AB, MT = RW / 16, NT = TW / 8, RG = kRT / RW, SB = 64 / AB;
    __shared__ __align__(16) Stage<NMAT, AB> st[2];
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
    auto load = [&](int it, int buf) {
        const int kb0 = it * kKB;
        Stage<NMAT, AB>& S = st[buf];
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
        for (int i = tid; i < kJ * 2; i += kThreads) {
            const int j = i >> 1;
            const size_t o = size_t(srow[j]) * kblk + kb0 * SB;
            if constexpr (SB == 2) {
                if (i & 1) cp_async16(&S.am[j][0], am + o);
                else cp_async16(&S.ad[j][0], ad + o);
            } else {
                if (i & 1) cp_async8(&S.am[j][0], am + o);
                else cp_async8(&S.ad[j][0], ad + o);
            }
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
    const int n_it = nb / kKB;
    load(0, 0);
    for (int it = 0; it < n_it; ++it) {
        if (it + 1 < n_it) {
            load(it + 1, (it + 1) & 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const Stage<NMAT, AB>& S = st[it & 1];
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb) {
            uint2 bf[NT][2];
            float da[NT][2][SB];   // [n][token 2c + tc][scale block]
            int ma[NT][2][SB];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const uint8_t* ar = &S.a[wt * TW + 8 * n + g][kb * 64 + 8 * c];
                bf[n][0] = *reinterpret_cast<const uint2*>(ar);
                bf[n][1] = *reinterpret_cast<const uint2*>(ar + 32);
#pragma unroll
                for (int tc = 0; tc < 2; ++tc) {
                    const int j = wt * TW + 8 * n + 2 * c + tc;
#pragma unroll
                    for (int q = 0; q < SB; ++q) {
                        da[n][tc][q] = S.ad[j][kb * SB + q];
                        ma[n][tc][q] = S.am[j][kb * SB + q];
                    }
                }
            }
#pragma unroll
            for (int mat = 0; mat < NMAT; ++mat)
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int rl = wr * RW + 16 * m + g;
                    const uint32_t wlo = *reinterpret_cast<const uint32_t*>(&S.w[mat][kb][rl][4 * c]);
                    const uint32_t whi = *reinterpret_cast<const uint32_t*>(&S.w[mat][kb][rl + 8][4 * c]);
                    constexpr uint32_t M3 = 0x03030303u;
                    const uint32_t a0[4] = {wlo & M3, whi & M3, (wlo >> 2) & M3, (whi >> 2) & M3};
                    const uint32_t a1[4] = {(wlo >> 4) & M3, (whi >> 4) & M3, (wlo >> 6) & M3, (whi >> 6) & M3};
                    const uint32_t s_lo = S.ws[mat][rl], s_hi = S.ws[mat][rl + 8];
                    const float dw0 = __half2float(__ushort_as_half(uint16_t(kb ? s_lo >> 16 : s_lo & 0xffff)));
                    const float dw1 = __half2float(__ushort_as_half(uint16_t(kb ? s_hi >> 16 : s_hi & 0xffff)));
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
        __syncthreads();   // the buffer is refilled next iteration
    }

    // element q of acc[.][m][n]: weight row r0 + wr * RW + 16m + g + 8 (q >> 1), token wt * TW + 8n + 2c + (q & 1)
    if constexpr (NMAT == 2) {
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
                if (g == 0 && j < ne) {
                    hd[size_t(p0 + j) * ffb + blk] = dd;
                    hm[size_t(p0 + j) * ffb + blk] = kMagic - sum;
                }
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
    } else {
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = j0 + wt * TW + 8 * n + 2 * c + (q & 1);
                if (j >= ne) continue;
                const int slot = slot_of[p0 + j];
#pragma unroll
                for (int m = 0; m < MT; ++m) y[size_t(slot) * rows + r0 + wr * RW + 16 * m + g + 8 * (q >> 1)] = acc[0][m][n][q];
            }
    }
}

struct Ws {
    int32_t *cnt, *bounds, *tok_of, *slot_of, *n_tiles;
    int2* tiles;
    int8_t *xq, *hq;
    float *xd, *hd;
    int32_t *xm, *hm;
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
    w.xd = reinterpret_cast<float*>(take(size_t(T) * n / 32 * 4));
    w.xm = reinterpret_cast<int32_t*>(take(size_t(T) * n / 32 * 4));
    w.hq = reinterpret_cast<int8_t*>(take(S * ff));
    w.hd = reinterpret_cast<float*>(take(S * ff / 32 * 4));
    w.hm = reinterpret_cast<int32_t*>(take(S * ff / 32 * 4));
    w.bytes = size_t(p - static_cast<char*>(base));
    return w;
}

}  // namespace

size_t workspace_bytes(int T, int K, int n, int ff, int E) { return carve(nullptr, T, K, n, ff, E).bytes; }

void run(const uint8_t* experts, size_t stride, int E, int n, int ff, const float* x, const int32_t* ids, int T, int K, float* yd,
         void* ws, size_t ws_bytes, cudaStream_t stream, bool block64) {
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
        k_quant<AB><<<(xb + 7) / 8, 256, 0, stream>>>(x, xb, w.xq, w.xd, w.xm);
        k_moe_q2<2, AB><<<dim3(ff / kRT, mt), kThreads, 0, stream>>>(experts, stride, 0, gu, ff, n, w.xq, w.xd, w.xm, w.tok_of, w.bounds,
                                                                    w.tiles, w.n_tiles, w.hq, w.hd, w.hm, nullptr, nullptr);
        k_moe_q2<1, AB><<<dim3(n / kRT, mt), kThreads, 0, stream>>>(experts, stride, 2 * gu, 0, n, ff, w.hq, w.hd, w.hm, nullptr, w.bounds,
                                                                   w.tiles, w.n_tiles, nullptr, nullptr, nullptr, yd, w.slot_of);
    };
    if (block64) go(std::integral_constant<int, 64>{});
    else go(std::integral_constant<int, 32>{});
    ck(cudaGetLastError(), "moe_q2::run");
}

}  // namespace flashrt::moe_q2
