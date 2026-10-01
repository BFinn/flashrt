// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// QSA attention: the KV caches (fp16, q8, the hot set), the indexer and block selection,
// split-K and tensor-core attention.
#include "arch/qwen4exp/blocks_common.cuh"

#include <cooperative_groups.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace flashrt::qwen4exp {

namespace {

// per (token, head): RMS norm over dim with weight w, then NEOX rope on the first n_rot dims
// at position pos0 + token; src rows are `src_stride` floats apart per token and `head_stride`
// per head, dst is [T][heads][dim]

// round_fp16: store values rounded to fp16 (the KV cache format of the parity reference)
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_out(__half* p, float v) { *p = __float2half(v); }

// OutT float: plain (queries); OutT __half: the K cache (round to fp16, as llama.cpp's F16 cache)
// dst + position * dst_pos_stride (0 for plain outputs); the position from dp[1] when dp is set
template <typename OutT>
__global__ void k_norm_rope(const float* src, int src_stride, int head_stride, const float* w, OutT* dst, int heads,
                            int dim, int n_rot, int pos0, float theta_scale, float eps, size_t dst_pos_stride, const int32_t* dp) {
    if (dp) pos0 = dp[1];
    dst += size_t(pos0) * dst_pos_stride;
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const float* x = src + size_t(t) * src_stride + size_t(h) * head_stride;
    OutT* y = dst + (size_t(t) * heads + h) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    const int half = n_rot / 2;
    const float pos = float(pos0 + t);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i < half) {
            const float theta = pos * powf(theta_scale, float(i));
            float sn, cs;
            sincosf(theta, &sn, &cs);
            const float x0 = x[i] * inv * w[i], x1 = x[i + half] * inv * w[i + half];
            store_out(y + i, x0 * cs - x1 * sn);
            store_out(y + i + half, x0 * sn + x1 * cs);
        } else if (i >= n_rot) {
            store_out(y + i, x[i] * inv * w[i]);
        }
    }
}

// dst + position * pos_stride; the position from dp[1] when dp is set (graph mode)
__global__ void k_copy_h(const float* src, __half* dst, int n, int pos0, size_t pos_stride, const int32_t* dp) {
    if (dp) pos0 = dp[1];
    dst += size_t(pos0) * pos_stride;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Q8_0 rows (blocks of 32, d = amax / 127, q = round(x / d), as ggml's quantize_row_q8_0_ref):
// src [n_rows][dim] float -> dst + position * pos_stride_rows rows; one warp per block of 32.
// With a hot set (hdst set): the row goes to the host store, and to the block's GPU slot when
// the block is resident (rows are cell * kvh + head; slots hold r cells).
// With a prefill mirror (mdst set): the row also goes to the full-size VRAM mirror.
__global__ void k_quant_q8(const float* src, int8_t* dst, __half* dsc, int n_rows, int dim, int pos0, size_t pos_stride_rows,
                           const int32_t* dp, int8_t* hdst, __half* hdsc, const int32_t* slot_of_block, int r, int kvh,
                           int8_t* mdst = nullptr, __half* mdsc = nullptr) {
    if (dp) pos0 = dp[1];
    const int lane = threadIdx.x & 31;
    const long g = (long(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;   // block of 32
    if (g >= long(n_rows) * (dim / 32)) return;
    const long row = g / (dim / 32), b = g % (dim / 32);
    const float x = src[row * dim + b * 32 + lane];
    float am = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const size_t orow = size_t(pos0) * pos_stride_rows + size_t(row);
    const int8_t qv = int8_t(roundf(x * id));
    if (!hdst) {
        dst[orow * dim + b * 32 + lane] = qv;
        if (lane == 0) dsc[orow * (dim / 32) + b] = __float2half(d);
        return;
    }
    hdst[orow * dim + b * 32 + lane] = qv;
    if (lane == 0) hdsc[orow * (dim / 32) + b] = __float2half(d);
    if (mdst) {
        mdst[orow * dim + b * 32 + lane] = qv;
        if (lane == 0) mdsc[orow * (dim / 32) + b] = __float2half(d);
    }
    const long cell = long(orow) / kvh, head = long(orow) % kvh;
    const int slot = slot_of_block[cell / r];
    if (slot >= 0) {
        const size_t srow = (size_t(slot) * r + cell % r) * kvh + head;
        dst[srow * dim + b * 32 + lane] = qv;
        if (lane == 0) dsc[srow * (dim / 32) + b] = __float2half(d);
    }
}

// Hot-set upkeep before one attention step (graph-safe, all on the device):
//  k_hot_select (one CTA): the blocks token T-1 selected are marked used and pinned for this step;
//    up to kHotPromote missed blocks get CLOCK victim slots (never a pinned one). The table is
//    updated at once; the data follows in k_hot_copy.
//  k_hot_copy (kHotPromote CTAs): CTA k copies promoted block k from the host store.
// Values are the same in both stores, so where a block sits never changes a result; blocks past
// the promotion limit are read from the host store by the attention kernel.
constexpr int kHotPromote = 1024;
__device__ int block_scan_flags(bool flag, int* total);
__global__ void k_hot_select(int32_t* slot_of_block, int32_t* block_of_slot, uint8_t* refbit, uint32_t* pinned, int32_t* hand,
                             int32_t* promo, const int32_t* cells, const int32_t* counts, int ldc, int T, int pos0,
                             const int32_t* dp, int r, int B) {
    if (dp) pos0 = dp[1];
    __shared__ int miss[kHotPromote];
    __shared__ int nmiss;
    const int t = T - 1, q = pos0 + t;
    const int cnt = counts ? counts[t] : -1;
    const int nblk = cnt < 0 ? q / r + 1 : (cnt + r - 1) / r;
    const uint32_t stamp = uint32_t(hand[1]) + 1;   // hand[1]: step counter
    if (threadIdx.x == 0) nmiss = 0;
    __syncthreads();
    for (int base = 0; base < nblk; base += blockDim.x) {
        const int i = base + threadIdx.x;
        int b = -1;
        if (i < nblk) b = cnt < 0 ? i : cells[size_t(t) * ldc + size_t(i) * r] / r;
        const int slot = b >= 0 ? slot_of_block[b] : 0;
        if (b >= 0 && slot >= 0) {
            refbit[slot] = 1;
            pinned[slot] = stamp;
        }
        const bool m = b >= 0 && slot < 0;
        int total;
        const int rank = block_scan_flags(m, &total);
        if (m && nmiss + rank < kHotPromote) miss[nmiss + rank] = b;
        __syncthreads();
        if (threadIdx.x == 0) nmiss = min(kHotPromote, nmiss + total);
        __syncthreads();
    }
    // CLOCK, a chunk of the ring at a time in parallel: from the hand, a slot not referenced since the
    // hand last passed and not pinned this step is a victim; passing any other slot clears its
    // reference bit. A block scan ranks the chunk's candidates in clock order, the first `need` take
    // the missed blocks in order, and the hand stops after the last one taken: the same victims,
    // pairing and hand as one thread stepping slot by slot. Give up after 2B slots without a victim
    // (every slot pinned: the rest stays in the host store).
    __shared__ int s_h, s_np, s_run, s_stop, s_last;
    if (threadIdx.x == 0) {
        s_h = hand[0];
        s_np = 0;
        s_run = 0;
    }
    __syncthreads();
    while (true) {
        const int np0 = s_np, h0 = s_h, need = nmiss - np0;
        if (need <= 0 || s_run >= 2 * B) break;
        const int ch = min(min(int(blockDim.x), B), 2 * B - s_run);   // no slot twice in a chunk
        const int i = threadIdx.x;
        const int slot = h0 + i < B ? h0 + i : h0 + i - B;
        const bool cand = i < ch && !refbit[slot] && pinned[slot] != stamp;
        int total;
        const int rank = block_scan_flags(cand, &total);
        if (threadIdx.x == 0) s_stop = ch;
        __syncthreads();
        if (cand && rank == need - 1) s_stop = i + 1;   // the last victim needed: the hand stops after it
        if (cand && rank == total - 1) s_last = i;       // the chunk's last candidate
        __syncthreads();
        const int stop = s_stop;
        if (i < stop) {
            if (cand && rank < need) {
                const int k = np0 + rank, blk = miss[k];
                const int old = block_of_slot[slot];
                if (old >= 0) slot_of_block[old] = -1;
                block_of_slot[slot] = blk;
                slot_of_block[blk] = slot;
                refbit[slot] = 1;
                pinned[slot] = stamp;
                promo[1 + 2 * k] = blk;
                promo[2 + 2 * k] = slot;
            } else {
                refbit[slot] = 0;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int taken = min(total, need);
            s_np = np0 + taken;
            s_h = h0 + stop < B ? h0 + stop : h0 + stop - B;
            // non-victims in a row since the last victim (the serial loop's guard)
            if (taken == 0) s_run += stop;
            else s_run = stop - 1 - s_last;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        promo[0] = s_np;
        hand[0] = s_h;
        hand[1] = int32_t(stamp);
    }
}

__global__ void k_hot_copy(int8_t* K, int8_t* V, __half* Ks, __half* Vs, const int8_t* hK, const int8_t* hV, const __half* hKs,
                           const __half* hVs, const int32_t* promo, int r, int kvh) {
    const int k = blockIdx.x;
    if (k >= promo[0]) return;
    const int b = promo[1 + 2 * k], v = promo[2 + 2 * k];
    const int rows = r * kvh, per_row = 17;   // 16 x 16 B of values + 16 B of scales
    for (int it = threadIdx.x; it < rows * per_row * 2; it += blockDim.x) {
        const int which = it % 2, part = (it / 2) % per_row, rr = it / 2 / per_row;
        const size_t src_row = size_t(b) * rows + rr, dst_row = size_t(v) * rows + rr;
        if (part < 16)
            reinterpret_cast<int4*>((which ? V : K) + dst_row * 256)[part] = reinterpret_cast<const int4*>((which ? hV : hK) + src_row * 256)[part];
        else
            *reinterpret_cast<int4*>((which ? Vs : Ks) + dst_row * 8) = *reinterpret_cast<const int4*>((which ? hVs : hKs) + src_row * 8);
    }
}

// fp16 rows -> Q8_0 rows (loading an fp16 state file into a q8 cache)
__global__ void k_h2q8(const __half* src, int8_t* dst, __half* dsc, long n_rows, int dim) {
    const int lane = threadIdx.x & 31;
    const long g = (long(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (g >= n_rows * (dim / 32)) return;
    const long row = g / (dim / 32), b = g % (dim / 32);
    const float x = __half2float(src[row * dim + b * 32 + lane]);
    float am = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    dst[row * dim + b * 32 + lane] = int8_t(roundf(x * id));
    if (lane == 0) dsc[row * (dim / 32) + b] = __float2half(d);
}

// K/V cell readers for the attention kernel: fp16, or Q8_0 (value * block scale)
// resolve(cell, head) gives the row the k4/v accessors take: cell * kv_heads + head, or with a
// hot set, a GPU slot row (>= 0) or -(host row) - 1 for a block that is not resident.
struct KvF16 {
    const __half* K;
    const __half* V;
    int kvh;
    __device__ long resolve(int cell, int hk) const { return long(cell) * kvh + hk; }
    __device__ float4 k4(long row, int d0) const {   // 4 values from dim d0 (d0 % 4 == 0)
        const uint2 u = *reinterpret_cast<const uint2*>(K + row * 256 + d0);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
    __device__ float v(long row, int d) const { return __half2float(V[row * 256 + d]); }
};
struct KvQ8 {
    const int8_t* K;
    const int8_t* V;
    const __half* Ks;
    const __half* Vs;
    int kvh;
    __device__ long resolve(int cell, int hk) const { return long(cell) * kvh + hk; }
    __device__ float4 k4(long row, int d0) const {
        const char4 c = *reinterpret_cast<const char4*>(K + row * 256 + d0);
        const float sc = __half2float(Ks[row * 8 + d0 / 32]);
        return make_float4(sc * c.x, sc * c.y, sc * c.z, sc * c.w);
    }
    __device__ float v(long row, int d) const { return __half2float(Vs[row * 8 + d / 32]) * float(V[row * 256 + d]); }
};
// q8 with a hot set: resident blocks in the GPU slots (g), the rest read from the mapped host store (h)
struct KvQ8Hot {
    KvQ8 g, h;
    const int32_t* slot_of_block;
    int r, kvh;
    __device__ long resolve(int cell, int hk) const {
        const int slot = slot_of_block[cell / r];
        return slot >= 0 ? (long(slot) * r + cell % r) * kvh + hk : -(long(cell) * kvh + hk) - 1;
    }
    __device__ float4 k4(long row, int d0) const { return row >= 0 ? g.k4(row, d0) : h.k4(-row - 1, d0); }
    __device__ float v(long row, int d) const { return row >= 0 ? g.v(row, d) : h.v(-row - 1, d); }
};

// ---- split-K flash-decode attention
constexpr int kAttnSplit = 64;     // cells per partial
constexpr int kAttnMaxGroup = 16;  // query heads per KV head

// One block per (split, kv head, token), 256 threads, head dim D == 256: the partial softmax of
// the group's G query heads over up to kAttnSplit cells. Cells come from the token's list, or
// are 0..pos when counts[t] < 0. Writes, per (token, head, split): max, sum, acc[D].
template <int G, typename KV>
__global__ void k_attn_part(const float* q, KV kvr, int heads, int kv_heads, int pos0, float scale,
                            const int32_t* cells, const int32_t* counts, int ldc, int n_splits, float* part, const int32_t* dp) {
    if (dp) pos0 = dp[1];
    constexpr int D = 256;
    __shared__ __align__(16) float qs[G][D];
    __shared__ float sc[G][kAttnSplit];
    __shared__ long cell_rows[kAttnSplit];
    const int split = blockIdx.x, hk = blockIdx.y, t = blockIdx.z;
    const int cnt = counts ? counts[t] : -1;
    const int n = cnt >= 0 ? cnt : pos0 + t + 1;
    const int j0 = split * kAttnSplit, nj = min(kAttnSplit, n - j0);
    float* pt = part + ((size_t(t) * heads + hk * G) * n_splits + split) * (D + 2);
    if (nj <= 0) {   // empty split: neutral partial
        for (int h = 0; h < G; ++h)
            if (threadIdx.x == 0) {
                pt[size_t(h) * n_splits * (D + 2) + 0] = -INFINITY;
                pt[size_t(h) * n_splits * (D + 2) + 1] = 0.0f;
            }
        return;
    }
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) qs[i / D][i % D] = q[(size_t(t) * heads + hk * G + i / D) * D + i % D];
    for (int j = threadIdx.x; j < nj; j += blockDim.x) cell_rows[j] = kvr.resolve(cnt >= 0 ? cells[size_t(t) * ldc + j0 + j] : j0 + j, hk);
    __syncthreads();
    // scores: warp w takes cells w, w + 8, ...; lane covers dims [lane*4, +4) and [128 + lane*4, +4)
    // (coalesced K rows, conflict-free shared q)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int j = warp; j < nj; j += blockDim.x >> 5) {
        const long krow = cell_rows[j];
        const float4 k0 = kvr.k4(krow, lane * 4), k1 = kvr.k4(krow, 128 + lane * 4);
        float d[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 q0 = reinterpret_cast<const float4*>(qs[h])[lane], q1 = reinterpret_cast<const float4*>(qs[h])[32 + lane];
            d[h] = q0.x * k0.x + q0.y * k0.y + q0.z * k0.z + q0.w * k0.w + q1.x * k1.x + q1.y * k1.y + q1.z * k1.z + q1.w * k1.w;
        }
#pragma unroll
        for (int h = 0; h < G; ++h)
            for (int o = 16; o > 0; o >>= 1) d[h] += __shfl_xor_sync(0xffffffff, d[h], o);
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < G; ++h) sc[h][j] = d[h] * scale;
    }
    __syncthreads();
    // per head: max and exp over the split
    __shared__ float mx[G], sm[G];
    if (threadIdx.x < G) {
        const int h = threadIdx.x;
        float m = -INFINITY;
        for (int j = 0; j < nj; ++j) m = fmaxf(m, sc[h][j]);
        float s = 0.0f;
        for (int j = 0; j < nj; ++j) {
            const float e = expf(sc[h][j] - m);
            sc[h][j] = e;
            s += e;
        }
        mx[h] = m;
        sm[h] = s;
    }
    __syncthreads();
    // V: thread d accumulates dimension d for every head
    const int dd = threadIdx.x;
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int j = 0; j < nj; ++j) {
        const float v = kvr.v(cell_rows[j], dd);
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] += sc[h][j] * v;
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        float* ph = pt + size_t(h) * n_splits * (D + 2);
        ph[2 + dd] = acc[h];
        if (dd == 0) {
            ph[0] = mx[h];
            ph[1] = sm[h];
        }
    }
}

// One block per (token, head): combine the splits, then the sigmoid output gate.
__global__ void k_attn_combine(const float* part, int n_splits, const float* qfull, float* o, int heads, int dim,
                               int gate_stride, int gate_head_stride, int gate_off) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const float* ph = part + (size_t(t) * heads + h) * n_splits * (dim + 2);
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, ph[size_t(s) * (dim + 2)]);
    float denom = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float ms = ph[size_t(s) * (dim + 2)];
        if (ms > -INFINITY) denom += ph[size_t(s) * (dim + 2) + 1] * expf(ms - m);
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float ms = ph[size_t(s) * (dim + 2)];
            if (ms > -INFINITY) acc += ph[size_t(s) * (dim + 2) + 2 + i] * expf(ms - m);
        }
        const float g = qfull[size_t(t) * gate_stride + size_t(h) * gate_head_stride + gate_off + i];
        o[(size_t(t) * heads + h) * dim + i] = acc / denom / (1.0f + expf(-g));
    }
}

// ---- tensor-core attention for prefill sub-batches
// One CTA per (kv head, token), W warps. The group's G <= 16 query heads are the M = 16 rows of
// mma.m16n8k16 (fp16 in, fp32 accumulate). The token's cells go by in steps of 16, step k to
// warp k % W; each warp keeps its own online softmax (base 2, the max refreshed only when a row
// grows by more than 8: the final division uses the same max, so this is exact) and its own
// O [16][256] in registers. K and V go from global memory straight into fragments, with no shared
// staging: the dot product's k order and the output's column order are permuted so that each
// lane reads contiguous bytes. For K that is 64 dims of one cell; for V, 32 dims of four cells,
// which is one Q8_0 block. The warps then combine in a fixed order, and the output gate follows.
// Dims in the fragments:
//   QK k-step s, lane (g, c): k slots 2c, 2c+1 -> dims 64c + 4s + {0, 1}; 2c+8, 2c+9 -> + {2, 3}
//   PV n-tile i, column slot n -> dim 32n + i
constexpr int kAttnTcWarps = 4;
constexpr int kAttnTcMinTokens = 16;   // sub-batches at least this large take the tensor-core kernel

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
// bytes (sel) of x ^ 0x80808080 (signed int8 made unsigned) -> half2 of the signed values, exact
__device__ __forceinline__ __half2 i8x2(uint32_t xs, uint32_t sel) {
    return __hsub2(u2h(__byte_perm(xs, 0x64646464u, sel)), __float2half2_rn(1152.0f));   // (1024 + x + 128) - 1152
}

template <int G, bool Q8, int W>
__global__ void __launch_bounds__(32 * W) k_attn_tc(const float* q, const void* Kp, const void* Vp, const __half* Ks, const __half* Vs,
                                                     int heads, int kvh, int pos0, float scale, const int32_t* cells, const int32_t* counts,
                                                     int ldc, const float* qfull, float* o) {
    constexpr int D = 256;
    __shared__ uint4 qf[16][32];     // Q fragments per k-step, pre-scaled
    __shared__ float4 ob[32][32];    // the combined O, in fragment order
    __shared__ float mw[W][16], lw[W][16], ms[16], ls[16];
    const int hk = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int cnt = counts ? counts[t] : -1;
    const int n = cnt >= 0 ? cnt : pos0 + t + 1;
    const int32_t* cl = cnt >= 0 ? cells + size_t(t) * ldc : nullptr;
    {   // a0: row g dims {0,1}, a1: row g+8 {0,1}, a2: row g {2,3}, a3: row g+8 {2,3} (of 64c + 4s)
        const float qs = scale * 1.4426950408889634f;
        const float* qt = q + (size_t(t) * heads + size_t(hk) * G) * D;
        for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
            const int s = i / 32, l = i % 32, d = 64 * (l & 3) + 4 * s;
            uint32_t r[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int row = (l >> 2) + 8 * (k & 1), dd = d + 2 * (k >> 1);
                const float2 v = row < G ? *reinterpret_cast<const float2*>(qt + row * D + dd) : make_float2(0.0f, 0.0f);
                r[k] = h2u(__floats2half2_rn(v.x * qs, v.y * qs));
            }
            qf[s][l] = make_uint4(r[0], r[1], r[2], r[3]);
        }
    }
    __syncthreads();

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // rows g, g + 8 (l: this lane's share)
    const int n_steps = (n + 15) / 16;
    for (int st = warp; st < n_steps; st += W) {
        const int j0 = st * 16;
        // the step's 16 cells as K/V rows (cells past the end repeat the first, and are masked)
        const int jj = j0 + (lane & 15);
        const int cell = cl ? cl[jj < n ? jj : j0] : (jj < n ? jj : j0);
        const int row = cell * kvh + hk;
        const int rk0 = __shfl_sync(~0u, row, g), rk1 = __shfl_sync(~0u, row, 8 + g);
        const int rv0 = __shfl_sync(~0u, row, 2 * c), rv1 = __shfl_sync(~0u, row, 2 * c + 1);
        const int rv2 = __shfl_sync(~0u, row, 2 * c + 8), rv3 = __shfl_sync(~0u, row, 2 * c + 9);

        // V first (it is needed last): cells 2c, 2c+1, 2c+8, 2c+9, dims [32g, 32g + 32)
        constexpr int VW = Q8 ? 8 : 16;   // words per cell
        uint32_t va[VW], vb[VW], vc[VW], vd[VW];
        __half2 sab, scd;
        {
            const size_t esz = Q8 ? 1 : 2;
            const char* V = static_cast<const char*>(Vp);
            const size_t off = size_t(32 * g) * esz;
#pragma unroll
            for (int k = 0; k < VW / 4; ++k) {
                reinterpret_cast<uint4*>(va)[k] = reinterpret_cast<const uint4*>(V + size_t(rv0) * D * esz + off)[k];
                reinterpret_cast<uint4*>(vb)[k] = reinterpret_cast<const uint4*>(V + size_t(rv1) * D * esz + off)[k];
                reinterpret_cast<uint4*>(vc)[k] = reinterpret_cast<const uint4*>(V + size_t(rv2) * D * esz + off)[k];
                reinterpret_cast<uint4*>(vd)[k] = reinterpret_cast<const uint4*>(V + size_t(rv3) * D * esz + off)[k];
            }
            if constexpr (Q8) {
                sab = __halves2half2(Vs[size_t(rv0) * 8 + g], Vs[size_t(rv1) * 8 + g]);
                scd = __halves2half2(Vs[size_t(rv2) * 8 + g], Vs[size_t(rv3) * 8 + g]);
            }
        }
        // K: n-tile i, lane (g, c): cell 8i + g, dims [64c, 64c + 64)
        constexpr int KW = Q8 ? 16 : 32;
        uint32_t k0w[KW], k1w[KW];
        __half2 ksc[2][2];
        {
            const size_t esz = Q8 ? 1 : 2;
            const char* K = static_cast<const char*>(Kp);
#pragma unroll
            for (int k = 0; k < KW / 4; ++k) {
                reinterpret_cast<uint4*>(k0w)[k] = reinterpret_cast<const uint4*>(K + (size_t(rk0) * D + 64 * c) * esz)[k];
                reinterpret_cast<uint4*>(k1w)[k] = reinterpret_cast<const uint4*>(K + (size_t(rk1) * D + 64 * c) * esz)[k];
            }
            if constexpr (Q8) {
                ksc[0][0] = __half2half2(Ks[size_t(rk0) * 8 + 2 * c]);
                ksc[0][1] = __half2half2(Ks[size_t(rk0) * 8 + 2 * c + 1]);
                ksc[1][0] = __half2half2(Ks[size_t(rk1) * 8 + 2 * c]);
                ksc[1][1] = __half2half2(Ks[size_t(rk1) * 8 + 2 * c + 1]);
            }
        }
        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const uint4 a4 = qf[s][lane];
            const uint32_t a[4] = {a4.x, a4.y, a4.z, a4.w};
            uint32_t b00, b01, b10, b11;
            if constexpr (Q8) {
                const uint32_t x0 = k0w[s] ^ 0x80808080u, x1 = k1w[s] ^ 0x80808080u;
                b00 = h2u(__hmul2(i8x2(x0, 0x4140), ksc[0][s / 8]));
                b01 = h2u(__hmul2(i8x2(x0, 0x4342), ksc[0][s / 8]));
                b10 = h2u(__hmul2(i8x2(x1, 0x4140), ksc[1][s / 8]));
                b11 = h2u(__hmul2(i8x2(x1, 0x4342), ksc[1][s / 8]));
            } else {
                b00 = k0w[2 * s];
                b01 = k0w[2 * s + 1];
                b10 = k1w[2 * s];
                b11 = k1w[2 * s + 1];
            }
            mma16816(s0, a, b00, b01);
            mma16816(s1, a, b10, b11);
        }
        // s0: cells j0 + 2c + {0, 1}, s1: j0 + 8 + 2c + {0, 1}; rows g (e = 0, 1) and g + 8 (e = 2, 3)
        if (j0 + 16 > n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (j0 + 2 * c + (e & 1) >= n) s0[e] = -INFINITY;
                if (j0 + 8 + 2 * c + (e & 1) >= n) s1[e] = -INFINITY;
            }
        }
        float r0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
        float r1 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
        r0 = fmaxf(r0, __shfl_xor_sync(~0u, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(~0u, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(~0u, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(~0u, r1, 2));
        if (__any_sync(~0u, r0 > m0 + 8.0f || r1 > m1 + 8.0f)) {
            const float n0 = fmaxf(m0, r0), n1 = fmaxf(m1, r1);
            const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);
            m0 = n0;
            m1 = n1;
            l0 *= a0;
            l1 *= a1;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                acc[i][0] *= a0;
                acc[i][1] *= a0;
                acc[i][2] *= a1;
                acc[i][3] *= a1;
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float mm = e < 2 ? m0 : m1;
            s0[e] = exp2f(s0[e] - mm);
            s1[e] = exp2f(s1[e] - mm);
        }
        l0 += s0[0] + s0[1] + s1[0] + s1[1];
        l1 += s0[2] + s0[3] + s1[2] + s1[3];
        const uint32_t pa[4] = {h2u(__floats2half2_rn(s0[0], s0[1])), h2u(__floats2half2_rn(s0[2], s0[3])),
                                h2u(__floats2half2_rn(s1[0], s1[1])), h2u(__floats2half2_rn(s1[2], s1[3]))};
        // PV: n-tile i, b0 = (V[2c], V[2c+1]) and b1 = (V[2c+8], V[2c+9]) at dim 32g + i
        if constexpr (Q8) {
#pragma unroll
            for (int w = 0; w < 8; ++w) {   // word w: dims 32g + 4w .. + 3, n-tiles 4w .. 4w + 3
                const uint32_t xa = va[w] ^ 0x80808080u, xb = vb[w] ^ 0x80808080u, xc = vc[w] ^ 0x80808080u, xd = vd[w] ^ 0x80808080u;
                const uint32_t t0 = __byte_perm(xa, xb, 0x5140), t1 = __byte_perm(xa, xb, 0x7362);
                const uint32_t u0 = __byte_perm(xc, xd, 0x5140), u1 = __byte_perm(xc, xd, 0x7362);
                mma16816(acc[4 * w + 0], pa, h2u(__hmul2(i8x2(t0, 0x4140), sab)), h2u(__hmul2(i8x2(u0, 0x4140), scd)));
                mma16816(acc[4 * w + 1], pa, h2u(__hmul2(i8x2(t0, 0x4342), sab)), h2u(__hmul2(i8x2(u0, 0x4342), scd)));
                mma16816(acc[4 * w + 2], pa, h2u(__hmul2(i8x2(t1, 0x4140), sab)), h2u(__hmul2(i8x2(u1, 0x4140), scd)));
                mma16816(acc[4 * w + 3], pa, h2u(__hmul2(i8x2(t1, 0x4342), sab)), h2u(__hmul2(i8x2(u1, 0x4342), scd)));
            }
        } else {
#pragma unroll
            for (int w = 0; w < 16; ++w) {   // word w: dims 32g + 2w, + 1: n-tiles 2w, 2w + 1
                mma16816(acc[2 * w + 0], pa, __byte_perm(va[w], vb[w], 0x5410), __byte_perm(vc[w], vd[w], 0x5410));
                mma16816(acc[2 * w + 1], pa, __byte_perm(va[w], vb[w], 0x7632), __byte_perm(vc[w], vd[w], 0x7632));
            }
        }
    }

    // combine the warps: common max per row, then the scaled sums in warp order (deterministic)
    l0 += __shfl_xor_sync(~0u, l0, 1);
    l0 += __shfl_xor_sync(~0u, l0, 2);
    l1 += __shfl_xor_sync(~0u, l1, 1);
    l1 += __shfl_xor_sync(~0u, l1, 2);
    if (c == 0) {
        mw[warp][g] = m0;
        mw[warp][g + 8] = m1;
        lw[warp][g] = l0;
        lw[warp][g + 8] = l1;
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float M = -INFINITY, L = 0.0f;
        for (int w = 0; w < W; ++w) M = fmaxf(M, mw[w][threadIdx.x]);
        for (int w = 0; w < W; ++w) L += lw[w][threadIdx.x] * exp2f(mw[w][threadIdx.x] - M);
        ms[threadIdx.x] = M;
        ls[threadIdx.x] = L;
    }
    __syncthreads();
    const float f0 = exp2f(m0 - ms[g]), f1 = exp2f(m1 - ms[g + 8]);   // 0 for a warp without cells
    for (int w = 0; w < W; ++w) {
        if (warp == w) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                float4 v = make_float4(acc[i][0] * f0, acc[i][1] * f0, acc[i][2] * f1, acc[i][3] * f1);
                if (w > 0) {
                    const float4 u = ob[i][lane];
                    v = make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
                }
                ob[i][lane] = v;
            }
        }
        __syncthreads();
    }
    // out [t][h][d] = O / L * sigmoid(gate); O[r][d] sits at n-tile d % 32, lane (r % 8) * 4 + d / 64,
    // component (d / 32) % 2 + 2 * (r / 8)
    for (int idx = threadIdx.x; idx < G * D; idx += blockDim.x) {
        const int r = idx / D, d = idx % D, h = hk * G + r;
        const float v = reinterpret_cast<const float*>(&ob[d % 32][(r % 8) * 4 + d / 64])[(d / 32) % 2 + 2 * (r / 8)];
        const float gt = qfull[(size_t(t) * heads + h) * 2 * D + D + d];
        o[(size_t(t) * heads + h) * D + d] = v / ls[r] / (1.0f + expf(-gt));
    }
}

// Pool every block completed by a token of this call: mean of the block's raw keys (from this
// call's keys, or the ring for earlier positions), RMS norm, NEOX rope at the block's first
// position. One CUDA block per token; tokens that complete no block exit.
__global__ void k_idx_pool(const float* kraw, const float* ring, const float* w, __half* pooled, int pos0, int r, int dim,
                           int n_rot, float theta_scale, float eps, const int32_t* dp) {
    if (dp) pos0 = dp[1];
    const int t = blockIdx.x, p = pos0 + t;
    if (p % r != r - 1) return;
    const int b = p / r;
    __shared__ float m[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < r; ++k) {
            const int pk = p - (r - 1) + k;
            acc += pk >= pos0 ? kraw[size_t(pk - pos0) * dim + i] : ring[size_t(pk % (2 * r)) * dim + i];
        }
        acc /= float(r);
        m[i] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    __syncthreads();
    const int half = n_rot / 2;
    __half* y = pooled + size_t(b) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i < half) {
            float sn, cs;
            sincosf(float(b * r) * powf(theta_scale, float(i)), &sn, &cs);
            const float x0 = m[i] * inv * w[i], x1 = m[i + half] * inv * w[i + half];
            y[i] = __float2half_rn(x0 * cs - x1 * sn);
            y[i + half] = __float2half_rn(x0 * sn + x1 * cs);
        } else if (i >= n_rot) {
            y[i] = __float2half_rn(m[i] * inv * w[i]);
        }
    }
}

// the last 2r raw keys of this call go to ring slot position % 2r
__global__ void k_idx_ring(const float* kraw, float* ring, int pos0, int T, int r, int dim, const int32_t* dp) {
    if (dp) pos0 = dp[1];
    const int t = T - 1 - int(blockIdx.x);   // the last min(T, 2r) tokens
    const int p = pos0 + t;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ring[size_t(p % (2 * r)) * dim + i] = kraw[size_t(t) * dim + i];
}

// score[t][b] = sum over heads of relu(q[t][h] . pooled[b]), for the blocks complete at token t
__global__ void k_idx_scores(const float* qi, const __half* pooled, float* scores, int ld, int pos0, int r, int heads,
                             int dim) {
    const int t = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = (pos0 + t + 1) / r;
    if (b >= nb) return;
    const __half* kb = pooled + size_t(b) * dim;
    float sum = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const float* qh = qi + (size_t(t) * heads + h) * dim;
        float d = 0.0f;
        for (int i = 0; i < dim; ++i) d += qh[i] * __half2float(kb[i]);
        sum += fmaxf(d, 0.0f);
    }
    scores[size_t(t) * ld + b] = sum;
}

// As k_idx_scores for dim == 128: one warp per pooled key (a coalesced 256-byte read of fp16, the
// warp's keys loaded before the dot products), the token's queries in shared memory; each block
// covers 8 warps x kIdxKeysPerWarp keys. Bound by the keys' bytes: fp32 keys took 37 us per call at
// 245K (sw122), fp16 halves them.
constexpr int kIdxKeysPerWarp = 4;
__global__ void k_idx_scores128(const float* qi, const __half* pooled, float* scores, int ld, int pos0, int r, int heads,
                                const int32_t* dp) {
    if (dp) pos0 = dp[1];
    extern __shared__ __align__(16) float qs[];   // [heads][128]
    const int t = blockIdx.y;
    const int nb = (pos0 + t + 1) / r;
    const int b0 = blockIdx.x * (blockDim.x >> 5) * kIdxKeysPerWarp;
    if (b0 >= nb) return;
    for (int i = threadIdx.x; i < heads * 128; i += blockDim.x) qs[i] = qi[size_t(t) * heads * 128 + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint2 kr[kIdxKeysPerWarp] = {};
#pragma unroll
    for (int j = 0; j < kIdxKeysPerWarp; ++j) {
        const int b = b0 + warp * kIdxKeysPerWarp + j;
        if (b < nb) kr[j] = reinterpret_cast<const uint2*>(pooled + size_t(b) * 128)[lane];
    }
#pragma unroll
    for (int j = 0; j < kIdxKeysPerWarp; ++j) {
        const int b = b0 + warp * kIdxKeysPerWarp + j;
        if (b >= nb) break;
        const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&kr[j].x));
        const float2 k23 = __half22float2(*reinterpret_cast<const __half2*>(&kr[j].y));
        const float4 kv = make_float4(k01.x, k01.y, k23.x, k23.y);
        float sum = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const float4 q = reinterpret_cast<const float4*>(qs + h * 128)[lane];
            float d = q.x * kv.x + q.y * kv.y + q.z * kv.z + q.w * kv.w;
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
            sum += fmaxf(d, 0.0f);
        }
        if (lane == 0) scores[size_t(t) * ld + b] = sum;
    }
}

// Indexer scores for prefill sub-batches (4 heads, dim 128) on tensor cores: a CTA takes 32
// tokens, whose 128 (token, head) query rows are the M side of mma.m16n8k16 (fp16 in, fp32
// accumulate; each warp holds 8 tokens' fragments in registers). It walks tiles of 64 pooled
// keys (fp16, as stored), and fuses relu and the sum over the heads (two
// shuffles) into the epilogue. Tiles past the tile's last token are skipped. Scores of blocks a
// token can not see yet are written too, as k_idx_select reads only (pos + 1) / r of each row.
constexpr int kIdxTcTokens = 32, kIdxTcKeys = 64;
__global__ void __launch_bounds__(128) k_idx_scores_tc(const float* qi, const __half* pooled, float* scores, int ld, int pos0, int r, int Ts) {
    constexpr int D = 128, KP = D + 8;   // key row stride in shared (halves): conflict-free fragment loads
    __shared__ __align__(16) __half ks[kIdxTcKeys][KP];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int t0 = blockIdx.y * kIdxTcTokens;
    const int t_last = min(t0 + kIdxTcTokens, Ts) - 1;
    const int nb_last = (pos0 + t_last + 1) / r;   // the tile's largest block count
    // A fragments: m-tile mt, step s; rows g, g + 8 -> token 8 * warp + 4 * mt + row / 4, head row % 4
    uint32_t qa[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int row = g + 8 * hr, t = t0 + 8 * warp + 4 * mt + row / 4;
            const float* qr = qi + (size_t(t) * 4 + row % 4) * D;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                float2 lo = make_float2(0.0f, 0.0f), hi = make_float2(0.0f, 0.0f);
                if (t < Ts) {
                    lo = *reinterpret_cast<const float2*>(qr + 16 * s + 2 * c);
                    hi = *reinterpret_cast<const float2*>(qr + 16 * s + 2 * c + 8);
                }
                qa[mt][s][hr] = h2u(__floats2half2_rn(lo.x, lo.y));
                qa[mt][s][2 + hr] = h2u(__floats2half2_rn(hi.x, hi.y));
            }
        }
    const int n_tiles = (nb_last + kIdxTcKeys - 1) / kIdxTcKeys;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int b0 = tile * kIdxTcKeys;
        __syncthreads();   // the previous tile's keys are no longer read
        for (int i = threadIdx.x; i < kIdxTcKeys * D / 4; i += blockDim.x) {
            const int kb = i / (D / 4), d = (i % (D / 4)) * 4;
            uint2 u = make_uint2(0u, 0u);
            if (b0 + kb < nb_last) u = *reinterpret_cast<const uint2*>(pooled + size_t(b0 + kb) * D + d);
            *reinterpret_cast<uint2*>(&ks[kb][d]) = u;
        }
        __syncthreads();
        float acc[2][8][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t b0f = *reinterpret_cast<const uint32_t*>(&ks[8 * nt + g][16 * s + 2 * c]);
                const uint32_t b1f = *reinterpret_cast<const uint32_t*>(&ks[8 * nt + g][16 * s + 2 * c + 8]);
                mma16816(acc[0][nt], qa[0][s], b0f, b1f);
                mma16816(acc[1][nt], qa[1][s], b0f, b1f);
            }
        // relu, sum over the 4 heads (lanes 4 and 8 apart), write: row g -> token 4 * mt + g / 4,
        // row g + 8 -> + 2; columns 2c, 2c + 1 -> blocks b0 + 8 nt + 2c + {0, 1}
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = fmaxf(acc[mt][nt][e], 0.0f);
                    v[e] += __shfl_xor_sync(~0u, v[e], 4);
                    v[e] += __shfl_xor_sync(~0u, v[e], 8);
                }
                if ((g & 3) == 0) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int t = t0 + 8 * warp + 4 * mt + g / 4 + 2 * (e >> 1), b = b0 + 8 * nt + 2 * c + (e & 1);
                        if (t < Ts && b < ld) scores[size_t(t) * ld + b] = v[e];
                    }
                }
            }
    }
}

__device__ __forceinline__ unsigned ordered_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// block-wide exclusive scan of 0/1 flags (blockDim a multiple of 32, <= 1024); returns the
// thread's rank and sets *total
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
        if (lane < nw) warp_tot[lane] = v;   // inclusive
    }
    __syncthreads();
    const int before = warp > 0 ? warp_tot[warp - 1] : 0;
    *total = warp_tot[nw - 1];
    __syncthreads();
    return before + in_warp;
}

// block-wide exclusive scan of ints (blockDim a multiple of 32, <= 1024); sets *total
__device__ int block_scan_int(int v, int* total) {
    __shared__ int warp_tot[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int incl = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffff, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) warp_tot[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < nw ? warp_tot[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) w += n;
        }
        if (lane < nw) warp_tot[lane] = w;   // inclusive
    }
    __syncthreads();
    const int before = (warp > 0 ? warp_tot[warp - 1] : 0) + incl - v;
    *total = warp_tot[nw - 1];
    __syncthreads();
    return before;
}

// Per token (one cluster of kSelCluster CTAs): the cells to attend to. Dense (counts = -1) while
// q + 1 <= width; otherwise the top M blocks by score, M = nsel minus one when the incomplete tail
// exists, then the tail cells. The M-th largest score is found by a 32-bit radix select, 8 bits a
// pass; ties at it are taken in block order. CTA k of the cluster owns blocks [k * per, (k + 1) *
// per): each builds its digit histogram, and every CTA sums all of them through distributed shared
// memory, so all take the same digit. The final pass offsets each CTA's blocks by the counts of the
// CTAs before it, so the list is written in block order, as a single CTA would (P-5: one CTA took
// 103 us per call at 245K, 61K blocks). Prefill sub-batches (128 tokens) fill the GPU with tokens
// already, and 8 CTAs per token cost them 1.9-8.3x (test_idx_select, sw126): they take 1 CTA per
// token below kSelDeepBlocks blocks and 4 above (33 us at 32K, 213 us at 245K, against 275 and 398).
constexpr int kSelCluster = 8;
constexpr int kSelSingleMin = 64;        // sub-batches of this many tokens or more: 1 or 4 CTAs per token
constexpr int kSelDeepBlocks = 24576;    // (98K positions) the measured crossover between 1 and 4
constexpr int kSelKeys = 9216;      // a CTA's keys held in shared memory (65,536 blocks over 8 CTAs: 262K positions)
template <int C>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(1024)
    k_idx_select(const float* scores, int ld, int32_t* cells, int32_t* counts, int ldc, int pos0, int r, int nsel, int width,
                 const int32_t* dp) {
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int rank = int(cl.block_rank());
    if (dp) pos0 = dp[1];
    const int t = blockIdx.y, q = pos0 + t;
    if (q + 1 <= width) {   // the whole cluster takes this branch (one token per cluster)
        if (rank == 0 && threadIdx.x == 0) counts[t] = -1;
        return;
    }
    const int nb = (q + 1) / r, tail = (q + 1) - nb * r, M = nsel - (tail > 0 ? 1 : 0);
    const float* sc = scores + size_t(t) * ld;
    int32_t* out = cells + size_t(t) * ldc;
    const int per = (nb + C - 1) / C;
    const int lo = min(nb, rank * per), hi = min(nb, lo + per);
    int filled;
    if (nb <= M) {
        for (int b = lo + int(threadIdx.x); b < hi; b += blockDim.x)
            for (int k = 0; k < r; ++k) out[b * r + k] = b * r + k;
        filled = nb;
    } else {
        // tau = the M-th largest key, 8 bits at a time: this CTA's histogram of the next digit over
        // its keys that match the digits found so far (integer counts: order-independent), summed
        // over the cluster. Two histogram buffers, alternating by pass: a CTA clears the one it
        // writes next only after the next pass's cluster barrier, by when every CTA has read it.
        __shared__ int hist2[2][256], tot[256];
        __shared__ unsigned sh_prefix;
        __shared__ int sh_need;
        // the CTA's keys, loaded once (independent loads) for all five passes; a longer range reads
        // them from global memory each pass
        __shared__ unsigned keys[kSelKeys];
        const bool held = hi - lo <= kSelKeys;
        if (held)
            for (int i = threadIdx.x; i < hi - lo; i += blockDim.x) keys[i] = ordered_key(sc[lo + i]);
        auto key_of = [&](int b) { return held ? keys[b - lo] : ordered_key(sc[b]); };
        if (threadIdx.x == 0) {
            sh_prefix = 0;
            sh_need = M;
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            int* hist = hist2[(shift >> 3) & 1];
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            const unsigned prefix = sh_prefix, hmask = shift == 24 ? 0u : ~0u << (shift + 8);
            // warp-aggregated: lanes with the same digit add once (scores share their top bits, so
            // plain atomics would serialise on a few bins)
            for (int b0 = lo; b0 < hi; b0 += blockDim.x) {
                const int b = b0 + threadIdx.x;
                int bin = -1;
                if (b < hi) {
                    const unsigned key = key_of(b);
                    if ((key & hmask) == prefix) bin = int((key >> shift) & 255);
                }
                const unsigned same = __match_any_sync(0xffffffff, bin);
                if (bin >= 0 && int(threadIdx.x & 31) == __ffs(same) - 1) atomicAdd(&hist[bin], __popc(same));
            }
            cl.sync();   // every CTA's histogram is complete (and last pass's buffer read by all)
            for (int d = threadIdx.x; d < 256; d += blockDim.x) {
                int v = 0;
                for (int k = 0; k < C; ++k) v += cl.map_shared_rank(hist, k)[d];
                tot[d] = v;
            }
            __syncthreads();
            if (threadIdx.x < 32) {   // warp 0: the digit holding the need-th largest key
                const int lane = threadIdx.x, need = sh_need;
                int part = 0;   // lane owns digits 255 - 8 lane .. 248 - 8 lane (descending)
                for (int k = 0; k < 8; ++k) part += tot[255 - 8 * lane - k];
                int incl = part;
                for (int o = 1; o < 32; o <<= 1) {
                    const int v = __shfl_up_sync(0xffffffff, incl, o);
                    if (lane >= o) incl += v;
                }
                const int excl = incl - part;
                const unsigned hit = __ballot_sync(0xffffffff, excl < need && incl >= need);
                if (lane == __ffs(hit) - 1) {
                    int acc = excl, d = 255 - 8 * lane;
                    for (int k = 0; k < 7; ++k, --d) {
                        if (acc + tot[d] >= need) break;
                        acc += tot[d];
                    }
                    sh_need = need - acc;
                    sh_prefix = prefix | (unsigned(d) << shift);
                }
            }
            __syncthreads();
        }
        const unsigned tau = sh_prefix;
        // blocks above tau, plus the lowest-index blocks equal to tau up to M, written in block
        // order (deterministic, so the attention sums in a fixed order). Each thread owns a
        // contiguous segment of the CTA's range: count, scan for the offsets within the CTA, add
        // the counts of the CTAs before this one, then write in order.
        const int seg = (hi - lo + int(blockDim.x) - 1) / int(blockDim.x);
        const int b0 = min(hi, lo + int(threadIdx.x) * seg), b1 = min(hi, b0 + seg);
        int g = 0, e = 0;
        for (int b = b0; b < b1; ++b) {
            const unsigned key = key_of(b);
            g += key > tau;
            e += key == tau;
        }
        __shared__ int cta_ge[2];
        int g_cta, e_cta;
        const int g_before = block_scan_int(g, &g_cta);
        const int e_before = block_scan_int(e, &e_cta);
        if (threadIdx.x == 0) {
            cta_ge[0] = g_cta;
            cta_ge[1] = e_cta;
        }
        cl.sync();
        int g_off = 0, e_off = 0, g_total = 0;
        for (int k = 0; k < C; ++k) {
            const int* o = cl.map_shared_rank(cta_ge, k);
            if (k < rank) {
                g_off += o[0];
                e_off += o[1];
            }
            g_total += o[0];
        }
        cl.sync();   // no CTA exits while another still reads its counts
        const int tie_budget = M - g_total;
        int pos = g_off + g_before + min(e_off + e_before, tie_budget), tie_idx = e_off + e_before;
        for (int b = b0; b < b1; ++b) {
            const unsigned key = key_of(b);
            bool sel = key > tau;
            if (key == tau) sel = tie_idx++ < tie_budget;
            if (sel) {
                for (int k = 0; k < r; ++k) out[pos * r + k] = b * r + k;
                ++pos;
            }
        }
        filled = M;
    }
    if (rank != 0) return;
    for (int k = threadIdx.x; k < tail; k += blockDim.x) out[filled * r + k] = nb * r + k;
    if (threadIdx.x == 0) counts[t] = filled * r + tail;
}

// cluster: CTAs per token (1, 2, 4 or 8, the same output); 0 chooses: 8 in graphs and for short
// windows, otherwise 1 or 4 by depth (sw127)
void select_launch(const float* scores, int ld, int32_t* cells, int32_t* counts, int ldc, int pos0, int T, int r, int nsel, int width,
                   const int32_t* dp, int cluster, cudaStream_t stream) {
    if (cluster == 0) cluster = !dp && T >= kSelSingleMin ? ((pos0 + T) / r >= kSelDeepBlocks ? 4 : 1) : kSelCluster;
    switch (cluster) {
        case 1: k_idx_select<1><<<dim3(1, T), 1024, 0, stream>>>(scores, ld, cells, counts, ldc, pos0, r, nsel, width, dp); break;
        case 2: k_idx_select<2><<<dim3(2, T), 1024, 0, stream>>>(scores, ld, cells, counts, ldc, pos0, r, nsel, width, dp); break;
        case 4: k_idx_select<4><<<dim3(4, T), 1024, 0, stream>>>(scores, ld, cells, counts, ldc, pos0, r, nsel, width, dp); break;
        default: k_idx_select<8><<<dim3(8, T), 1024, 0, stream>>>(scores, ld, cells, counts, ldc, pos0, r, nsel, width, dp); break;
    }
    ck(cudaGetLastError(), "qsa_select");
}

}  // namespace

void qsa_select(const float* scores, int ld, int32_t* cells, int32_t* counts, int ldc, int pos0, int T, int r, int nsel, int width,
                cudaStream_t stream, int cluster) {
    select_launch(scores, ld, cells, counts, ldc, pos0, T, r, nsel, width, nullptr, cluster, stream);
}

void qsa_h2q8_rows(const void* src_f16, void* dst_q8, void* dst_scales, long n_rows, int dim, cudaStream_t stream) {
    const long groups = n_rows * (dim / 32);
    k_h2q8<<<unsigned((groups * 32 + 255) / 256), 256, 0, stream>>>(static_cast<const __half*>(src_f16), static_cast<int8_t*>(dst_q8),
                                                                   static_cast<__half*>(dst_scales), n_rows, dim);
    ck(cudaGetLastError(), "h2q8");
}

namespace {
// one device buffer to or from the file, through a host bounce buffer
void state_bytes(FILE* f, void* dev, size_t bytes, bool save, std::vector<uint8_t>& bounce) {
    bounce.resize(bytes);
    if (save) {
        ck(cudaMemcpy(bounce.data(), dev, bytes, cudaMemcpyDefault), "state to host");
        if (std::fwrite(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file write failed");
    } else {
        if (std::fread(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file truncated");
        ck(cudaMemcpy(dev, bounce.data(), bytes, cudaMemcpyDefault), "state to device");
    }
}
}  // namespace

void qsa_state_io(FILE* f, const Spec& s, QsaCache& kv, int pos, bool save, bool file_q8, cudaStream_t stream) {
    std::vector<uint8_t> bounce;
    const size_t kvn = size_t(s.n_head_kv) * s.head_dim_k;   // K or V values of one cell
    const bool q8 = kv.q8;
    if (file_q8 && !q8) throw std::runtime_error("state file has a q8 KV cache; this cache is fp16");
    void* K = kv.hot_blocks ? kv.hK : kv.K;
    void* V = kv.hot_blocks ? kv.hV : kv.V;
    uint16_t* Ks = kv.hot_blocks ? kv.hKs : kv.Ks;
    uint16_t* Vs = kv.hot_blocks ? kv.hVs : kv.Vs;
    if (kv.hot_blocks) reset_qsa_hot(s, kv, stream);
    if (!q8) {
        state_bytes(f, kv.K, size_t(pos) * kvn * 2, save, bounce);
        state_bytes(f, kv.V, size_t(pos) * kvn * 2, save, bounce);
    } else if (file_q8 || save) {
        state_bytes(f, K, size_t(pos) * kvn, save, bounce);
        state_bytes(f, Ks, size_t(pos) * kvn / 32 * 2, save, bounce);
        state_bytes(f, V, size_t(pos) * kvn, save, bounce);
        state_bytes(f, Vs, size_t(pos) * kvn / 32 * 2, save, bounce);
    } else {   // fp16 file into a q8 cache: convert in chunks on the GPU
        const long rows = long(pos) * s.n_head_kv, chunk = 1L << 16;
        void* tmp = nullptr;
        ck(cudaMalloc(&tmp, size_t(chunk) * s.head_dim_k * 2), "cudaMalloc state conversion");
        for (int which = 0; which < 2; ++which) {
            int8_t* dst = static_cast<int8_t*>(which ? V : K);
            uint16_t* dsc = which ? Vs : Ks;
            for (long r0 = 0; r0 < rows; r0 += chunk) {
                const long nr = std::min(chunk, rows - r0);
                state_bytes(f, tmp, size_t(nr) * s.head_dim_k * 2, false, bounce);
                qsa_h2q8_rows(tmp, dst + size_t(r0) * s.head_dim_k, dsc + size_t(r0) * (s.head_dim_k / 32), nr, s.head_dim_k, stream);
                ck(cudaStreamSynchronize(stream), "state conversion");
            }
        }
        cudaFree(tmp);
    }
    {   // the file keeps the pooled keys in fp32 (its format before they were stored in fp16)
        const size_t np = size_t(pos / s.qsa_block + 1) * s.idx_dim;
        std::vector<__half> h(np);
        if (save) {
            ck(cudaMemcpy(h.data(), kv.idx_pooled, np * 2, cudaMemcpyDeviceToHost), "pooled keys to host");
            bounce.resize(np * 4);
            float* fl = reinterpret_cast<float*>(bounce.data());
            for (size_t i = 0; i < np; ++i) fl[i] = __half2float(h[i]);
            if (std::fwrite(bounce.data(), 1, np * 4, f) != np * 4) throw std::runtime_error("state file write failed");
        } else {
            bounce.resize(np * 4);
            if (std::fread(bounce.data(), 1, np * 4, f) != np * 4) throw std::runtime_error("state file truncated");
            const float* fl = reinterpret_cast<const float*>(bounce.data());
            for (size_t i = 0; i < np; ++i) h[i] = __float2half_rn(fl[i]);
            ck(cudaMemcpy(kv.idx_pooled, h.data(), np * 2, cudaMemcpyHostToDevice), "pooled keys to device");
        }
    }
    // the file keeps the ring of the last `block` positions at slot position % block (its format
    // before the ring grew to qsa_ring_slots); positions before 0 stay zero
    const int r = s.qsa_block, R = qsa_ring_slots(s);
    const size_t row = size_t(s.idx_dim) * 4;
    std::vector<uint8_t> ring(size_t(R) * row, 0), file(size_t(r) * row, 0);
    if (save) ck(cudaMemcpy(ring.data(), kv.idx_ring, ring.size(), cudaMemcpyDeviceToHost), "ring to host");
    for (int p = std::max(0, pos - r); p < pos && save; ++p) std::memcpy(&file[size_t(p % r) * row], &ring[size_t(p % R) * row], row);
    if (save ? std::fwrite(file.data(), 1, file.size(), f) != file.size() : std::fread(file.data(), 1, file.size(), f) != file.size())
        throw std::runtime_error("state file ring i/o failed");
    if (!save) {
        for (int p = std::max(0, pos - r); p < pos; ++p) std::memcpy(&ring[size_t(p % R) * row], &file[size_t(p % r) * row], row);
        ck(cudaMemcpy(kv.idx_ring, ring.data(), ring.size(), cudaMemcpyHostToDevice), "ring to device");
    }
}

size_t qsa_cell_bytes(const Spec& s, bool q8) {
    const size_t n = size_t(s.n_head_kv) * s.head_dim_k;
    return q8 ? n + n / 32 * 2 : n * 2;
}

void qsa_mirror_begin(const Spec& s, QsaCache& kv, int pos, int cells, cudaStream_t stream) {
    if (!kv.hot_blocks || kv.mK) return;
    cells = std::min(kv.capacity, (std::max(cells, pos) + s.qsa_block - 1) / s.qsa_block * s.qsa_block);
    kv.mcap = cells;
    const size_t n = size_t(cells) * s.n_head_kv * s.head_dim_k;
    ck(cudaMalloc(&kv.mK, n), "cudaMalloc KV mirror");
    ck(cudaMalloc(&kv.mV, n), "cudaMalloc KV mirror");
    ck(cudaMalloc(&kv.mKs, n / 32 * 2), "cudaMalloc KV mirror");
    ck(cudaMalloc(&kv.mVs, n / 32 * 2), "cudaMalloc KV mirror");
    const size_t have = size_t(pos) * s.n_head_kv * s.head_dim_k;   // what the host store holds so far
    if (have) {
        ck(cudaMemcpyAsync(kv.mK, kv.hK, have, cudaMemcpyDefault, stream), "KV mirror fill");
        ck(cudaMemcpyAsync(kv.mV, kv.hV, have, cudaMemcpyDefault, stream), "KV mirror fill");
        ck(cudaMemcpyAsync(kv.mKs, kv.hKs, have / 32 * 2, cudaMemcpyDefault, stream), "KV mirror fill");
        ck(cudaMemcpyAsync(kv.mVs, kv.hVs, have / 32 * 2, cudaMemcpyDefault, stream), "KV mirror fill");
    }
}

void qsa_mirror_end(QsaCache& kv) {
    for (void* p : {kv.mK, kv.mV, static_cast<void*>(kv.mKs), static_cast<void*>(kv.mVs)})
        if (p) cudaFree(p);
    kv.mK = kv.mV = nullptr;
    kv.mKs = kv.mVs = nullptr;
    kv.mcap = 0;
}

void reset_qsa_hot(const Spec& s, QsaCache& kv, cudaStream_t stream) {
    if (!kv.hot_blocks) return;
    ck(cudaMemsetAsync(kv.slot_of_block, 0xff, size_t(kv.capacity / s.qsa_block) * 4, stream), "reset hot table");
    ck(cudaMemsetAsync(kv.block_of_slot, 0xff, size_t(kv.hot_blocks) * 4, stream), "reset hot slots");
    ck(cudaMemsetAsync(kv.refbit, 0, size_t(kv.hot_blocks), stream), "reset hot refbits");
    ck(cudaMemsetAsync(kv.clock_hand, 0, 8, stream), "reset clock");
    ck(cudaMemsetAsync(kv.pinned, 0, size_t(kv.hot_blocks) * 4, stream), "reset hot pins");
}

QsaCache alloc_qsa_cache(const Spec& s, int capacity, bool q8, int hot_blocks) {
    QsaCache kv;
    const int r = s.qsa_block;
    capacity = (capacity + r - 1) / r * r;
    kv.capacity = capacity;
    kv.q8 = q8;
    if (q8 && s.head_dim_k % 32) throw std::runtime_error("alloc_qsa_cache: q8 needs head_dim % 32 == 0");
    if (hot_blocks > 0 && size_t(hot_blocks) * r < size_t(capacity)) {
        if (!q8 || s.head_dim_k != 256) throw std::runtime_error("alloc_qsa_cache: the hot set needs a q8 cache with head_dim 256");
        kv.hot_blocks = hot_blocks;
        const size_t hn = size_t(capacity) * s.n_head_kv * s.head_dim_k;
        auto host = [&](void** hp, void** dp, size_t bytes) {
            ck(cudaHostAlloc(hp, bytes, cudaHostAllocMapped), "cudaHostAlloc KV host store");
            ck(cudaHostGetDevicePointer(dp, *hp, 0), "KV host store device pointer");
        };
        host(&kv.hK_host, &kv.hK, hn);
        host(&kv.hV_host, &kv.hV, hn);
        host(&kv.hKs_host, reinterpret_cast<void**>(&kv.hKs), hn / 32 * 2);
        host(&kv.hVs_host, reinterpret_cast<void**>(&kv.hVs), hn / 32 * 2);
        ck(cudaMalloc(&kv.slot_of_block, size_t(capacity / r) * 4), "cudaMalloc hot table");
        ck(cudaMalloc(&kv.block_of_slot, size_t(hot_blocks) * 4), "cudaMalloc hot slots");
        ck(cudaMalloc(&kv.refbit, size_t(hot_blocks)), "cudaMalloc hot refbits");
        ck(cudaMalloc(&kv.clock_hand, 8), "cudaMalloc clock");
        ck(cudaMalloc(&kv.pinned, size_t(hot_blocks) * 4), "cudaMalloc hot pins");
        ck(cudaMalloc(&kv.promo, size_t(1 + 2 * kHotPromote) * 4), "cudaMalloc hot promotions");
        reset_qsa_hot(s, kv, nullptr);
        ck(cudaDeviceSynchronize(), "hot set init");
    }
    const size_t n = size_t(kv.hot_blocks ? size_t(kv.hot_blocks) * r : size_t(capacity)) * s.n_head_kv * s.head_dim_k;
    ck(cudaMalloc(&kv.K, n * (q8 ? 1 : 2)), "cudaMalloc K cache");
    ck(cudaMalloc(&kv.V, n * (q8 ? 1 : 2)), "cudaMalloc V cache");
    if (q8) {
        ck(cudaMalloc(&kv.Ks, n / 32 * 2), "cudaMalloc K scales");
        ck(cudaMalloc(&kv.Vs, n / 32 * 2), "cudaMalloc V scales");
    }
    ck(cudaMalloc(&kv.idx_pooled, size_t(capacity / s.qsa_block + 1) * s.idx_dim * 2), "cudaMalloc pooled keys");
    ck(cudaMalloc(&kv.idx_ring, size_t(qsa_ring_slots(s)) * s.idx_dim * 4), "cudaMalloc key ring");
    ck(cudaMemset(kv.idx_ring, 0, size_t(qsa_ring_slots(s)) * s.idx_dim * 4), "memset key ring");
    return kv;
}

void free_qsa_cache(QsaCache& kv) {
    qsa_mirror_end(kv);
    if (kv.K) cudaFree(kv.K);
    if (kv.V) cudaFree(kv.V);
    if (kv.Ks) cudaFree(kv.Ks);
    if (kv.Vs) cudaFree(kv.Vs);
    for (void* h : {kv.hK_host, kv.hV_host, kv.hKs_host, kv.hVs_host})
        if (h) cudaFreeHost(h);
    for (void* d : {static_cast<void*>(kv.slot_of_block), static_cast<void*>(kv.block_of_slot), static_cast<void*>(kv.refbit),
                    static_cast<void*>(kv.clock_hand), static_cast<void*>(kv.pinned), static_cast<void*>(kv.promo)})
        if (d) cudaFree(d);
    if (kv.idx_pooled) cudaFree(kv.idx_pooled);
    if (kv.idx_ring) cudaFree(kv.idx_ring);
    kv = QsaCache{};
}

void qsa_scratch_reserve(const Spec& s, BlockScratch& bs, int T, int max_nb, int n_splits) {
    const int r = s.qsa_block, width = s.idx_top_k + r - 1, ldc = ((width + r - 1) / r) * r;
    if (n_splits == 0) n_splits = (std::max(width, ldc) + kAttnSplit - 1) / kAttnSplit;
    if (max_nb > 0 && bs.idx_scores_elems < size_t(T) * max_nb) {
        if (bs.idx_scores) cudaFree(bs.idx_scores);
        bs.idx_scores_elems = size_t(T) * max_nb * 2;
        ck(cudaMalloc(&bs.idx_scores, bs.idx_scores_elems * 4), "cudaMalloc idx scores");
        bs.version = new_scratch_version();
    }
    if (bs.idx_cells_elems < size_t(T) * ldc) {
        if (bs.idx_cells) cudaFree(bs.idx_cells);
        if (bs.idx_counts) cudaFree(bs.idx_counts);
        bs.idx_cells_elems = size_t(T) * ldc * 2;
        ck(cudaMalloc(&bs.idx_cells, bs.idx_cells_elems * 4), "cudaMalloc idx cells");
        ck(cudaMalloc(&bs.idx_counts, bs.idx_cells_elems / ldc * 4), "cudaMalloc idx counts");
        bs.version = new_scratch_version();
    }
    const size_t need = size_t(T) * s.n_head * std::max(n_splits, 0) * (s.head_dim_k + 2);
    if (bs.attn_part_elems < need) {
        if (bs.attn_part) cudaFree(bs.attn_part);
        bs.attn_part_elems = need;
        ck(cudaMalloc(&bs.attn_part, need * 4), "cudaMalloc attention partials");
        bs.version = new_scratch_version();
    }
}

void qsa_mixer(const BlockCtx& c, int il, const float* x, int T, int pos0, QsaCache& kv, float* out, float* gated_out,
               std::vector<std::vector<int32_t>>* sel_out) {
    const Spec& s = c.s;
    const int H = s.n_head, KH = s.n_head_kv, D = s.head_dim_k;
    const int r = s.qsa_block, IH = s.idx_heads, ID = s.idx_dim;
    const int width = s.idx_top_k + r - 1, nsel = (width + r - 1) / r;
    if (pos0 + T > kv.capacity) throw std::runtime_error("qsa_mixer: KV cache full");
    if (s.head_dim_v != D || ID > 256) throw std::runtime_error("qsa_mixer: unsupported shape");
    float* qfull = c.scratch.f32;                   // [T][H * 2D]  (q | gate per head)
    float* kraw = qfull + size_t(T) * H * 2 * D;    // [T][KH * D]
    float* vraw = kraw + size_t(T) * KH * D;        // [T][KH * D]
    float* q = vraw + size_t(T) * KH * D;           // [T][H][D]
    float* o = q + size_t(T) * H * D;               // [T][H][D]
    float* qi = o + size_t(T) * H * D;              // [T][IH * ID] indexer queries (normed, roped in place)
    float* ki = qi + size_t(T) * IH * ID;           // [T][ID] indexer raw keys
    if (size_t(ki + size_t(T) * ID - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("qsa_mixer: scratch too small");

    {
        const GpuTensor* ws[3] = {&c.w.layer(il, "attn_q.weight"), &c.w.layer(il, "attn_k.weight"), &c.w.layer(il, "attn_v.weight")};
        float* ys[3] = {qfull, kraw, vraw};
        linear_shared(c, ws, ys, 3, x, T);
    }
    const float theta_scale = powf(float(s.rope_base), -2.0f / float(s.rope_dims));
    const float eps = float(s.rms_eps);
    k_norm_rope<float><<<T * H, 128, 0, c.stream>>>(qfull, H * 2 * D, 2 * D, static_cast<const float*>(c.w.layer(il, "attn_q_norm.weight").dev),
                                                   q, H, D, s.rope_dims, pos0, theta_scale, eps, 0, c.dparams);
    // K goes straight into the cache rows pos0.. (cell = position); V is copied as is
    __half* Kc = static_cast<__half*>(kv.K);
    __half* Vc = static_cast<__half*>(kv.V);
    if (kv.q8) {   // K normed and roped in place (float), then both quantized into the cache rows
        if (D % 32) throw std::runtime_error("qsa_mixer: q8 KV needs head_dim % 32 == 0");
        k_norm_rope<float><<<T * KH, 128, 0, c.stream>>>(kraw, KH * D, D, static_cast<const float*>(c.w.layer(il, "attn_k_norm.weight").dev),
                                                        kraw, KH, D, s.rope_dims, pos0, theta_scale, eps, 0, c.dparams);
        const int groups = T * KH * D / 32;
        k_quant_q8<<<(groups * 32 + 255) / 256, 256, 0, c.stream>>>(kraw, static_cast<int8_t*>(kv.K), reinterpret_cast<__half*>(kv.Ks),
                                                                   T * KH, D, pos0, size_t(KH), c.dparams, static_cast<int8_t*>(kv.hK),
                                                                   reinterpret_cast<__half*>(kv.hKs), kv.slot_of_block, r, KH,
                                                                   static_cast<int8_t*>(kv.mK), reinterpret_cast<__half*>(kv.mKs));
        k_quant_q8<<<(groups * 32 + 255) / 256, 256, 0, c.stream>>>(vraw, static_cast<int8_t*>(kv.V), reinterpret_cast<__half*>(kv.Vs),
                                                                   T * KH, D, pos0, size_t(KH), c.dparams, static_cast<int8_t*>(kv.hV),
                                                                   reinterpret_cast<__half*>(kv.hVs), kv.slot_of_block, r, KH,
                                                                   static_cast<int8_t*>(kv.mV), reinterpret_cast<__half*>(kv.mVs));
    } else {
        k_norm_rope<__half><<<T * KH, 128, 0, c.stream>>>(kraw, KH * D, D, static_cast<const float*>(c.w.layer(il, "attn_k_norm.weight").dev),
                                                         Kc, KH, D, s.rope_dims, pos0, theta_scale, eps, size_t(KH) * D, c.dparams);
        k_copy_h<<<(T * KH * D + 255) / 256, 256, 0, c.stream>>>(vraw, Vc, T * KH * D, pos0, size_t(KH) * D, c.dparams);
    }

    // indexer: queries, raw keys, pooling of the blocks completed here, then per-token selection
    const LinearOut qk[2] = {{&c.w.layer(il, "indexer.q_proj.weight"), qi}, {&c.w.layer(il, "indexer.k_proj.weight"), ki}};
    if (linear_multi_ok(qk, 2, T)) linear_multi(c, qk, 2, x, T);
    else {
        const GpuTensor* ws2[2] = {qk[0].W, qk[1].W};
        float* ys2[2] = {qi, ki};
        linear_shared(c, ws2, ys2, 2, x, T);
    }
    k_norm_rope<float><<<T * IH, 128, 0, c.stream>>>(qi, IH * ID, ID, static_cast<const float*>(c.w.layer(il, "indexer.q_norm.weight").dev),
                                                    qi, IH, ID, s.rope_dims, pos0, theta_scale, eps, 0, c.dparams);
    k_idx_pool<<<T, 128, 0, c.stream>>>(ki, kv.idx_ring, static_cast<const float*>(c.w.layer(il, "indexer.k_norm.weight").dev),
                                       reinterpret_cast<__half*>(kv.idx_pooled), pos0, r, ID, s.rope_dims, theta_scale, eps, c.dparams);
    k_idx_ring<<<std::min(T, qsa_ring_slots(s)), 128, 0, c.stream>>>(ki, kv.idx_ring, pos0, T, r, ID, c.dparams);

    const int ldc = nsel * r;
    const bool graph = c.dparams != nullptr;
    if (graph && ID != 128) throw std::runtime_error("qsa_mixer: graph mode needs indexer dim 128");
    const int G = H / KH;
    if (D != 256 || G > kAttnMaxGroup || H % KH) throw std::runtime_error("qsa_mixer: attention needs head_dim 256 and a group <= 16");
    const int n_splits = (std::max(width, ldc) + kAttnSplit - 1) / kAttnSplit;
    // selection and attention in sub-batches of tokens (prefill chunks), so the per-token score
    // rows and attention partials stay small at depth; each token only needs its own position
    const int S = T > 256 ? 128 : T;
    if (sel_out && S < T) throw std::runtime_error("qsa_mixer: sel_out needs T <= 256");
    // sub-batches of kAttnTcMinTokens or more attend with the tensor-core kernel (sw46), unless
    // they read through the hot set (decode windows) or run in a graph
    const bool tc_kv = kv.mK || !kv.hot_blocks;
    for (int t0 = 0; t0 < T; t0 += S) {
        const int Ts = std::min(S, T - t0), p0 = pos0 + t0;
        const bool tc = tc_kv && !graph && Ts >= kAttnTcMinTokens;
        const float* qi_s = qi + size_t(t0) * IH * ID;
        const float* q_s = q + size_t(t0) * H * D;
        const int32_t* cells = nullptr;
        const int32_t* counts = nullptr;
        if (graph || p0 + Ts > width) {   // some token needs a selection (graph mode: always; dense below the width)
            const int max_nb = graph ? kv.capacity / r : (p0 + Ts) / r;
            BlockScratch& bs = c.scratch;
            qsa_scratch_reserve(c.s, bs, Ts, max_nb, tc ? -1 : 0);
            if (ID == 128 && IH == 4 && !graph && Ts >= kAttnTcMinTokens) {   // tensor-core scores (sw49)
                const int ty = (Ts + kIdxTcTokens - 1) / kIdxTcTokens;
                const int tiles = (max_nb + kIdxTcKeys - 1) / kIdxTcKeys;
                const int tx = std::max(1, std::min((tiles + 3) / 4, 2048 / ty));   // about 4 key tiles per CTA
                k_idx_scores_tc<<<dim3(tx, ty), 128, 0, c.stream>>>(qi_s, reinterpret_cast<const __half*>(kv.idx_pooled), bs.idx_scores, max_nb, p0, r, Ts);
            } else if (ID == 128) {
                const int per_block = 8 * kIdxKeysPerWarp;
                k_idx_scores128<<<dim3((max_nb + per_block - 1) / per_block, Ts), 256, size_t(IH) * 128 * 4, c.stream>>>(
                    qi_s, reinterpret_cast<const __half*>(kv.idx_pooled), bs.idx_scores, max_nb, p0, r, IH, c.dparams);
            } else {
                k_idx_scores<<<dim3((max_nb + 127) / 128, Ts), 128, 0, c.stream>>>(qi_s, reinterpret_cast<const __half*>(kv.idx_pooled), bs.idx_scores, max_nb, p0, r, IH, ID);
            }
            select_launch(bs.idx_scores, max_nb, bs.idx_cells, bs.idx_counts, ldc, p0, Ts, r, nsel, width, c.dparams, 0, c.stream);
            cells = bs.idx_cells;
            counts = bs.idx_counts;
            if (sel_out) {
                std::vector<int32_t> hc(size_t(Ts) * ldc), hn(Ts);
                ck(cudaMemcpyAsync(hc.data(), cells, hc.size() * 4, cudaMemcpyDeviceToHost, c.stream), "cells to host");
                ck(cudaMemcpyAsync(hn.data(), counts, hn.size() * 4, cudaMemcpyDeviceToHost, c.stream), "counts to host");
                ck(cudaStreamSynchronize(c.stream), "sync");
                sel_out->assign(Ts, {});
                for (int t = 0; t < Ts; ++t)
                    if (hn[t] >= 0) (*sel_out)[t].assign(hc.begin() + size_t(t) * ldc, hc.begin() + size_t(t) * ldc + hn[t]);
            }
        } else if (sel_out) {
            sel_out->assign(Ts, {});
        }
        const float scale = 1.0f / sqrtf(float(D));
        if (tc) {
            const float* qf_s = qfull + size_t(t0) * H * 2 * D;
            float* o_s = o + size_t(t0) * H * D;
            const dim3 grid(KH, Ts);
            auto launch_tc = [&](auto q8, const void* K, const void* V, const void* Ks, const void* Vs) {
                constexpr bool Q8 = decltype(q8)::value;
                constexpr int W = kAttnTcWarps;
                const __half* ks = static_cast<const __half*>(Ks);
                const __half* vs = static_cast<const __half*>(Vs);
                switch (G) {
                    case 12: k_attn_tc<12, Q8, W><<<grid, 32 * W, 0, c.stream>>>(q_s, K, V, ks, vs, H, KH, p0, scale, cells, counts, ldc, qf_s, o_s); break;
                    case 8: k_attn_tc<8, Q8, W><<<grid, 32 * W, 0, c.stream>>>(q_s, K, V, ks, vs, H, KH, p0, scale, cells, counts, ldc, qf_s, o_s); break;
                    case 16: k_attn_tc<16, Q8, W><<<grid, 32 * W, 0, c.stream>>>(q_s, K, V, ks, vs, H, KH, p0, scale, cells, counts, ldc, qf_s, o_s); break;
                    default: throw std::runtime_error("qsa_mixer: unsupported GQA group size");
                }
            };
            if (kv.mK)
                launch_tc(std::true_type{}, kv.mK, kv.mV, kv.mKs, kv.mVs);
            else if (kv.q8)
                launch_tc(std::true_type{}, kv.K, kv.V, kv.Ks, kv.Vs);
            else
                launch_tc(std::false_type{}, kv.K, kv.V, nullptr, nullptr);
            continue;
        }
        // split-K flash decode: partials per (token, head, 64-cell split), then a combine
        qsa_scratch_reserve(c.s, c.scratch, Ts, 0, n_splits);
        if (kv.hot_blocks && !kv.mK) {   // bring the selected blocks into the hot set first (not with a prefill mirror)
            k_hot_select<<<1, 1024, 0, c.stream>>>(kv.slot_of_block, kv.block_of_slot, kv.refbit, kv.pinned, kv.clock_hand, kv.promo, cells,
                                                   counts, ldc, Ts, p0, c.dparams, r, kv.hot_blocks);
            k_hot_copy<<<kHotPromote, 256, 0, c.stream>>>(static_cast<int8_t*>(kv.K), static_cast<int8_t*>(kv.V),
                                                          reinterpret_cast<__half*>(kv.Ks), reinterpret_cast<__half*>(kv.Vs),
                                                          static_cast<const int8_t*>(kv.hK), static_cast<const int8_t*>(kv.hV),
                                                          reinterpret_cast<const __half*>(kv.hKs), reinterpret_cast<const __half*>(kv.hVs),
                                                          kv.promo, r, KH);
        }
        const dim3 grid(n_splits, KH, Ts);
        auto launch = [&](auto kvr) {
            using KV = decltype(kvr);
            switch (G) {
                case 12: k_attn_part<12, KV><<<grid, 256, 0, c.stream>>>(q_s, kvr, H, KH, p0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part, c.dparams); break;
                case 8: k_attn_part<8, KV><<<grid, 256, 0, c.stream>>>(q_s, kvr, H, KH, p0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part, c.dparams); break;
                case 16: k_attn_part<16, KV><<<grid, 256, 0, c.stream>>>(q_s, kvr, H, KH, p0, scale, cells, counts, ldc, n_splits, c.scratch.attn_part, c.dparams); break;
                default: throw std::runtime_error("qsa_mixer: unsupported GQA group size");
            }
        };
        const KvQ8 g8{static_cast<const int8_t*>(kv.K), static_cast<const int8_t*>(kv.V), reinterpret_cast<const __half*>(kv.Ks),
                      reinterpret_cast<const __half*>(kv.Vs), KH};
        if (kv.mK) {   // prefill: the whole cache is mirrored in VRAM
            launch(KvQ8{static_cast<const int8_t*>(kv.mK), static_cast<const int8_t*>(kv.mV), reinterpret_cast<const __half*>(kv.mKs),
                        reinterpret_cast<const __half*>(kv.mVs), KH});
        } else if (kv.hot_blocks) {
            const KvQ8 h8{static_cast<const int8_t*>(kv.hK), static_cast<const int8_t*>(kv.hV), reinterpret_cast<const __half*>(kv.hKs),
                          reinterpret_cast<const __half*>(kv.hVs), KH};
            launch(KvQ8Hot{g8, h8, kv.slot_of_block, r, KH});
        } else if (kv.q8) {
            launch(g8);
        } else {
            launch(KvF16{Kc, Vc, KH});
        }
        k_attn_combine<<<Ts * H, 256, 0, c.stream>>>(c.scratch.attn_part, n_splits, qfull + size_t(t0) * H * 2 * D, o + size_t(t0) * H * D,
                                                     H, D, H * 2 * D, 2 * D, D);
    }
    if (gated_out) ck(cudaMemcpyAsync(gated_out, o, size_t(T) * H * D * 4, cudaMemcpyDeviceToDevice, c.stream), "copy gated");
    linear(c, c.w.layer(il, "attn_output.weight"), o, out, T);
    ck(cudaGetLastError(), "qsa_mixer");
}

}  // namespace flashrt::qwen4exp
