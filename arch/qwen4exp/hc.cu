// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// Hyper-connections: the stream norms, the down/up mixes (fused decode kernels), the combines.
#include "arch/qwen4exp/blocks_common.cuh"

#include "kernels/cuda/ggml_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flashrt::qwen4exp {

namespace {

// one block per (token, stream): y = x / rms(x) * w[stream]
__global__ void k_grouped_rms_norm(const float* x, const float* w, float* y, int n, int hc, float eps) {
    const int row = blockIdx.x;            // t * hc + s
    const int s = row % hc;
    const float* xr = x + size_t(row) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[size_t(row) * n + i] = xr[i] * inv * w[size_t(s) * n + i];
}

__global__ void k_scale_silu(float* x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i] * scale;
        x[i] = v / (1.0f + __expf(-v));
    }
}

// mixed[t][i] = mean over s of xn[t][s][i] * sigmoid(gate[t][s][i])
// As k_grouped_rms_norm for n % 4 == 0 and blockDim.x == n / 4: one float4 per thread, kept in
// registers between the reduction and the write. yb, if given, gets y rounded to BF16 as well
// (the input of BF16 products, so they need no conversion pass); y may then be null, with inv
// [row] getting each row's 1 / rms instead (k_gated_mean_x recomputes y from x).
__global__ void k_grouped_rms_norm_v4(const float* x, const float* w, float* y, int n, int hc, float eps, __nv_bfloat16* yb,
                                      float* inv_out) {
    const int row = blockIdx.x, s = row % hc, i = threadIdx.x;
    const float4 v = reinterpret_cast<const float4*>(x + size_t(row) * n)[i];
    const float ss = block_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float inv = rsqrtf(ss / n + eps);
    const float4 wv = reinterpret_cast<const float4*>(w + size_t(s) * n)[i];
    const float4 r = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
    if (y) reinterpret_cast<float4*>(y + size_t(row) * n)[i] = r;
    if (inv_out && i == 0) inv_out[row] = inv;
    if (yb) {
        __nv_bfloat162 b[2] = {__floats2bfloat162_rn(r.x, r.y), __floats2bfloat162_rn(r.z, r.w)};
        reinterpret_cast<uint2*>(yb + size_t(row) * n)[i] = *reinterpret_cast<uint2*>(b);
    }
}

// sum over the block of 4 values per thread, in a fixed order (the result in every thread)
__device__ float4 block_sum4(float4 v) {
    __shared__ float4 red4[32];
    for (int o = 16; o > 0; o >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, o);
        v.y += __shfl_xor_sync(0xffffffff, v.y, o);
        v.z += __shfl_xor_sync(0xffffffff, v.z, o);
        v.w += __shfl_xor_sync(0xffffffff, v.w, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) red4[warp] = v;
    __syncthreads();
    float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = 0; k < nw; ++k) {
        const float4 u = red4[k];
        r.x += u.x;
        r.y += u.y;
        r.z += u.z;
        r.w += u.w;
    }
    __syncthreads();
    return r;
}

// Prefill, 4 streams: k_hc_combine (x += out * 2 sigmoid(inject / 4) per stream; skipped when out
// is null) and then k_grouped_rms_norm_v4 on the new x, in one pass. One block per token; thread
// i holds float4 i of each stream, so out is read once and x once. Same arithmetic as the two
// kernels (y, inv_out: as there). w_inj, if given ([4][4 * n] BF16), gives inj_out[t][4]: the
// inject product of the normed row, reduced in the block (no separate product reading yb).
__global__ void k_hc_combine_norm4(float* x, const float* out, const float* inject, const float* w, float* y, __nv_bfloat16* yb, int n,
                                   float eps, float* inv_out, const uint16_t* w_inj, float* inj_out) {
    constexpr int HC = 4;
    const int t = blockIdx.x, i = threadIdx.x;
    float4 v[HC];
    if (out) {
        const float4 o = reinterpret_cast<const float4*>(out + size_t(t) * n)[i];
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            const float wv = 2.0f / (1.0f + __expf(-inject[t * HC + s] / HC));
            float4* xp = reinterpret_cast<float4*>(x + (size_t(t) * HC + s) * n) + i;
            float4 a = *xp;
            a.x += o.x * wv;
            a.y += o.y * wv;
            a.z += o.z * wv;
            a.w += o.w * wv;
            *xp = a;
            v[s] = a;
        }
    } else {
#pragma unroll
        for (int s = 0; s < HC; ++s) v[s] = reinterpret_cast<const float4*>(x + (size_t(t) * HC + s) * n)[i];
    }
    float4 ip = make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // this thread's share of the 4 inject outputs
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        const float4 a = v[s];
        const float ss = block_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
        const float inv = rsqrtf(ss / n + eps);
        const float4 wv = reinterpret_cast<const float4*>(w + size_t(s) * n)[i];
        const float4 r = make_float4(a.x * inv * wv.x, a.y * inv * wv.y, a.z * inv * wv.z, a.w * inv * wv.w);
        const size_t row = size_t(t) * HC + s;
        if (y) reinterpret_cast<float4*>(y + row * n)[i] = r;
        if (inv_out && i == 0) inv_out[row] = inv;
        __nv_bfloat162 b[2] = {__floats2bfloat162_rn(r.x, r.y), __floats2bfloat162_rn(r.z, r.w)};
        reinterpret_cast<uint2*>(yb + row * n)[i] = *reinterpret_cast<uint2*>(b);
        if (w_inj) {
            float d[4];
#pragma unroll
            for (int oo = 0; oo < 4; ++oo) {
                const uint2 wb = reinterpret_cast<const uint2*>(w_inj + size_t(oo) * HC * n + size_t(s) * n)[i];
                d[oo] = __uint_as_float(wb.x << 16) * r.x + __uint_as_float(wb.x & 0xffff0000u) * r.y +
                        __uint_as_float(wb.y << 16) * r.z + __uint_as_float(wb.y & 0xffff0000u) * r.w;
            }
            ip.x += d[0];
            ip.y += d[1];
            ip.z += d[2];
            ip.w += d[3];
        }
    }
    if (w_inj) {
        const float4 tot = block_sum4(ip);
        if (i == 0) reinterpret_cast<float4*>(inj_out)[t] = tot;
    }
}

// Decode (TT tokens, a step or a verify window): the hyper-connection RMS norm, down-projection
// and inject projection in one kernel. Block (rb, g) normalises stream g of every token (xn = x *
// inv_g * w) into shared memory (block row 0 also writes it out), then its warps take one row
// each of [W_down; W_inject] (BF16, rank + n_inject rows of hc*n columns) over stream g's n
// columns, each weight read once for all tokens. part[t][g][row] holds the per-stream partial
// dot products; consumers sum them over g in order (deterministic, and per token the same
// arithmetic for every TT).
// Weight matrices of the decode hc kernels, read 8 consecutive elements at a time (element offset
// e, a multiple of 8): BF16, or Q8P (kTypeQ8P: int8 [rows][cols], then fp16 scales per 32).
// load(e) fetches the raw bytes (so a kernel can issue many loads before decoding any), dec()
// expands them.
struct WBf16 {
    const uint16_t* w;
    using Raw = uint4;
    __device__ __forceinline__ Raw load(size_t e) const { return __ldg(reinterpret_cast<const uint4*>(w + e)); }
    __device__ __forceinline__ static void dec(const Raw& u, float (&v)[8]) {
        const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[2 * k] = __uint_as_float(wv[k] << 16);
            v[2 * k + 1] = __uint_as_float(wv[k] & 0xffff0000u);
        }
    }
    __device__ __forceinline__ void get8(size_t e, float (&v)[8]) const {
        const uint4 u = *reinterpret_cast<const uint4*>(w + e);
        const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[2 * k] = __uint_as_float(wv[k] << 16);
            v[2 * k + 1] = __uint_as_float(wv[k] & 0xffff0000u);
        }
    }
};
struct WQ8P {
    const int8_t* q;
    const __half* d;
    struct Raw {
        uint2 q;
        __half d;
    };
    __device__ __forceinline__ Raw load(size_t e) const { return Raw{__ldg(reinterpret_cast<const uint2*>(q + e)), d[e / 32]}; }
    __device__ __forceinline__ static void dec(const Raw& u, float (&v)[8]) {
        const float sc = __half2float(u.d);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = float(int8_t(u.q.x >> (8 * k))) * sc;
            v[4 + k] = float(int8_t(u.q.y >> (8 * k))) * sc;
        }
    }
    __device__ __forceinline__ void get8(size_t e, float (&v)[8]) const {
        const uint2 u = *reinterpret_cast<const uint2*>(q + e);
        const float sc = __half2float(d[e / 32]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = float(int8_t(u.x >> (8 * k))) * sc;
            v[4 + k] = float(int8_t(u.y >> (8 * k))) * sc;
        }
    }
};

template <int TT, typename WD>
__global__ void k_hc_down(const float* x, const float* w_norm, WD Wd, const uint16_t* Wi, float* xn_out,
                          float* part, int n, int rank, int n_inject, float eps) {
    extern __shared__ __align__(16) float xs[];   // [TT][n]
    const int g = blockIdx.y, hc = gridDim.y, rows = rank + n_inject;
    for (int t = 0; t < TT; ++t) {
        const float* xg = x + (size_t(t) * hc + g) * n;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xg[i] * xg[i];
        ss = block_sum(ss);
        const float inv = rsqrtf(ss / n + eps);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = xg[i] * inv * w_norm[size_t(g) * n + i];
            xs[size_t(t) * n + i] = v;
            if (blockIdx.x == 0) xn_out[(size_t(t) * hc + g) * n + i] = v;
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const size_t hcn = size_t(hc) * n;
    const WBf16 wi{Wi};
    const size_t e0 = (r < rank ? size_t(r) : size_t(r - rank)) * hcn + size_t(g) * n;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
#pragma unroll 5
    for (int ch = lane; ch < n / 8; ch += 32) {
        float wf[8];
        if (r < rank) Wd.get8(e0 + 8 * size_t(ch), wf);
        else wi.get8(e0 + 8 * size_t(ch), wf);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float4 x0 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch];
            const float4 x1 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch + 1];
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[t] += wf[k] * xv[k];
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (lane == 0) part[(size_t(t) * hc + g) * rows + r] = a;
    }
}

// v2 of k_hc_down (sw75): a warp takes RPW rows (each x value read from shared memory serves RPW
// weights) and loads all its weights into registers before the norm, whose 1 / rms is applied
// after the dot product: part = inv * W (x * w_norm). One barrier. Needs n % 256 == 0 and
// n / 256 <= kHcDownMaxK. The inject rows (BF16, r >= rank, whole warps since rank % RPW == 0)
// take the plain loop.
constexpr int kHcDownMaxK = 12, kHcDownWarps = 8, kHcDownRpw = 2;
template <int TT, typename WD>
__global__ void __launch_bounds__(32 * kHcDownWarps) k_hc_down2(const float* x, const float* w_norm, WD Wd, const uint16_t* Wi,
                                                                float* xn_out, float* part, int n, int rank, int n_inject, float eps,
                                                                const float* cout, const float* cinj, float* wv_out) {
    constexpr int RPW = kHcDownRpw, NW = kHcDownWarps;
    extern __shared__ __align__(16) float xs[];   // [TT][n]: x * w_norm
    __shared__ float red[TT][NW];
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();   // k_hc_up_mix2 may start loading its weights
#endif
    const int g = blockIdx.y, hc = gridDim.y, rows = rank + n_inject, nk = n / 256;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = (blockIdx.x * NW + warp) * RPW;
    const size_t hcn = size_t(hc) * n;
    // the weights first
    typename WD::Raw raw[RPW][kHcDownMaxK];
    const bool down = r0 < rank;
    if (down)
#pragma unroll
        for (int j = 0; j < RPW; ++j)
#pragma unroll
            for (int k = 0; k < kHcDownMaxK; ++k)
                if (k < nk) raw[j][k] = Wd.load(size_t(r0 + j) * hcn + size_t(g) * n + 8 * size_t(lane + 32 * k));
#if __CUDA_ARCH__ >= 900
    // launched as a programmatic dependent (FLASHRT_HC_EARLY), the weights above load while the
    // kernel before runs (k_moe_combine_db's wait for the CPU misses); a no-op otherwise
    cudaGridDependencySynchronize();
#endif
    // x * w_norm into shared memory, and the sum of squares. With cout, x is first combined with
    // the previous mixer's output (as k_hc_combine: x + out * 2 sigmoid(inject / hc), as an fma);
    // k_hc_up_mix2 stores that x, and block (0, 0) leaves the weights in wv_out for it.
    if (cout && blockIdx.x == 0 && g == 0 && threadIdx.x < TT * hc)
        wv_out[threadIdx.x] = 2.0f / (1.0f + __expf(-cinj[threadIdx.x] / hc));
    for (int t = 0; t < TT; ++t) {
        const float4* xg = reinterpret_cast<const float4*>(x + (size_t(t) * hc + g) * n);
        const float4* wg = reinterpret_cast<const float4*>(w_norm + size_t(g) * n);
        const float4* og = cout ? reinterpret_cast<const float4*>(cout + size_t(t) * n) : nullptr;
        const float wv = cout ? 2.0f / (1.0f + __expf(-cinj[t * hc + g] / hc)) : 0.0f;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            float4 v = xg[i];
            const float4 w = wg[i];
            if (og) {
                const float4 o = og[i];
                v = make_float4(__fmaf_rn(o.x, wv, v.x), __fmaf_rn(o.y, wv, v.y), __fmaf_rn(o.z, wv, v.z), __fmaf_rn(o.w, wv, v.w));
            }
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            reinterpret_cast<float4*>(xs + size_t(t) * n)[i] = make_float4(v.x * w.x, v.y * w.y, v.z * w.z, v.w * w.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(~0u, ss, o);
        if (lane == 0) red[t][warp] = ss;
    }
    __syncthreads();
    float inv[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) ss += red[t][w];
        inv[t] = rsqrtf(ss / n + eps);
    }
    if (blockIdx.x == 0)
        for (int t = 0; t < TT; ++t)
            for (int i = threadIdx.x; i < n; i += blockDim.x) xn_out[(size_t(t) * hc + g) * n + i] = xs[size_t(t) * n + i] * inv[t];
    if (r0 >= rows) return;
    float acc[RPW][TT] = {};
    if (down) {
#pragma unroll
        for (int k = 0; k < kHcDownMaxK; ++k)
            if (k < nk) {
                const int ch = lane + 32 * k;
                float wf[RPW][8];
#pragma unroll
                for (int j = 0; j < RPW; ++j) WD::dec(raw[j][k], wf[j]);
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const float4 x0 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch];
                    const float4 x1 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch + 1];
                    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                    for (int j = 0; j < RPW; ++j)
#pragma unroll
                        for (int q = 0; q < 8; ++q) acc[j][t] += wf[j][q] * xv[q];
                }
            }
    } else {
        const WBf16 wi{Wi};
#pragma unroll
        for (int j = 0; j < RPW; ++j) {
            if (r0 + j >= rows) break;
            const size_t e0 = size_t(r0 + j - rank) * hcn + size_t(g) * n;
            for (int ch = lane; ch < n / 8; ch += 32) {
                float wf[8];
                wi.get8(e0 + 8 * size_t(ch), wf);
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const float4 x0 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch];
                    const float4 x1 = reinterpret_cast<const float4*>(xs + size_t(t) * n)[2 * ch + 1];
                    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                    for (int q = 0; q < 8; ++q) acc[j][t] += wf[q] * xv[q];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < RPW; ++j)
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            float a = acc[j][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
            if (lane == 0 && r0 + j < rows) part[(size_t(t) * hc + g) * rows + r0 + j] = a * inv[t];
        }
}

// Decode (TT tokens), hc == 4: the hyper-connection up-projection fused with its neighbours.
// gate = W_up (BF16, [hc * n rows][rank]) * silu(lo * scale); mixed[t][i] = mean_s xn[t][s][i] *
// sigmoid(gate[t][s][i]). One block covers columns i0 .. i0+7 of all 4 streams (32 rows); 8 lanes
// per row, each lane 16-byte loads of 8 weights, each weight used for all tokens. Needs rank % 64
// == 0 and TT * rank <= 4096. lo comes as k_hc_down's partials [TT][HC][rank + n_inject]; block 0
// also sums the inject rows.
template <int TT, typename WU>
__global__ void k_hc_up_mix(WU W, const float* part, int n_inject, float scale, const float* xn, float* mixed,
                            float* inject, int n, int rank) {
    constexpr int HC = 4;
    __shared__ __align__(16) float xs[4096];   // [TT][rank]
    __shared__ float contrib[TT][HC][8];
    const int prow = rank + n_inject;
    for (int t = 0; t < TT; ++t)
        for (int j = threadIdx.x; j < rank; j += blockDim.x) {
            float lo = 0.0f;
#pragma unroll
            for (int g = 0; g < HC; ++g) lo += part[(size_t(t) * HC + g) * prow + j];
            const float v = lo * scale;
            xs[t * rank + j] = v / (1.0f + __expf(-v));
        }
    if (blockIdx.x == 0 && threadIdx.x < TT * n_inject) {
        const int t = threadIdx.x / n_inject, k = threadIdx.x % n_inject;
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < HC; ++g) a += part[(size_t(t) * HC + g) * prow + rank + k];
        inject[t * n_inject + k] = a;
    }
    __syncthreads();
    const int grp = threadIdx.x >> 3, l8 = threadIdx.x & 7;
    const int st = grp >> 3, il = grp & 7, i = blockIdx.x * 8 + il;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    if (i < n) {
        const size_t e0 = (size_t(st) * n + i) * rank;
        for (int ch = l8; ch < rank / 8; ch += 8) {
            float wf[8];
            W.get8(e0 + 8 * size_t(ch), wf);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float4 x0 = reinterpret_cast<const float4*>(xs + t * rank)[2 * ch];
                const float4 x1 = reinterpret_cast<const float4*>(xs + t * rank)[2 * ch + 1];
                const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[t] += wf[k] * xv[k];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float a = acc[t];
        a += __shfl_xor_sync(0xffffffff, a, 4);
        a += __shfl_xor_sync(0xffffffff, a, 2);
        a += __shfl_xor_sync(0xffffffff, a, 1);
        if (l8 == 0 && i < n) {
            const float x = xn[(size_t(t) * HC + st) * n + i];
            contrib[t][st][il] = x / (1.0f + __expf(-a));
        }
    }
    __syncthreads();
    if (threadIdx.x < 8 * TT) {
        const int t = threadIdx.x / 8, c8 = threadIdx.x % 8, ii = blockIdx.x * 8 + c8;
        if (ii < n) {
            float m = 0.0f;
            for (int s2 = 0; s2 < HC; ++s2) m += contrib[t][s2][c8];
            mixed[size_t(t) * n + ii] = m * (1.0f / HC);
        }
    }
}

// v2 of k_hc_up_mix (sw75), rank == 64 * NCH: launched as a programmatic dependent of
// k_hc_down2, it loads its weights while the down kernel runs, then waits for the partials
// (cudaGridDependencySynchronize; a no-op without the launch attribute). The partials and xn are
// read with plain loads after the wait. At most 64 registers: 4 blocks per SM, so the 320
// blocks run in one wave (at 80 registers they took two; sw75).
template <int TT, typename WU, int NCH, bool LATE = false>
__global__ void __launch_bounds__(256, 4) k_hc_up_mix2(WU W, const float* part, int n_inject, float scale, const float* xn, float* mixed,
                                                    float* inject, int n, float* xres, const float* cout, const float* wvb) {
    constexpr int HC = 4, rank = 64 * NCH;
    __shared__ __align__(16) float xs[TT * rank];
    __shared__ float contrib[TT][HC][8];
    const int prow = rank + n_inject;
    const int grp = threadIdx.x >> 3, l8 = threadIdx.x & 7;
    const int st = grp >> 3, il = grp & 7, i = min(blockIdx.x * 8 + il, n - 1);
    typename WU::Raw raw[NCH];
    const size_t e0 = (size_t(st) * n + i) * rank;
    if (!LATE)
#pragma unroll
        for (int k = 0; k < NCH; ++k) raw[k] = W.load(e0 + 8 * size_t(l8 + 8 * k));
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
    for (int t = 0; t < TT; ++t)
        for (int j = threadIdx.x; j < rank; j += blockDim.x) {
            float lo = 0.0f;
#pragma unroll
            for (int g = 0; g < HC; ++g) lo += part[(size_t(t) * HC + g) * prow + j];
            const float v = lo * scale;
            xs[t * rank + j] = v / (1.0f + __expf(-v));
        }
    if (blockIdx.x == 0 && threadIdx.x < TT * n_inject) {
        const int t = threadIdx.x / n_inject, k = threadIdx.x % n_inject;
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < HC; ++g) a += part[(size_t(t) * HC + g) * prow + rank + k];
        inject[t * n_inject + k] = a;
    }
    __syncthreads();
    if (LATE)
#pragma unroll
        for (int k = 0; k < NCH; ++k) raw[k] = W.load(e0 + 8 * size_t(l8 + 8 * k));
    float acc[TT] = {};
#pragma unroll
    for (int k = 0; k < NCH; ++k) {
        const int ch = l8 + 8 * k;
        float wf[8];
        WU::dec(raw[k], wf);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float4 x0 = reinterpret_cast<const float4*>(xs + t * rank)[2 * ch];
            const float4 x1 = reinterpret_cast<const float4*>(xs + t * rank)[2 * ch + 1];
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int q = 0; q < 8; ++q) acc[t] += wf[q] * xv[q];
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float a = acc[t];
        a += __shfl_xor_sync(0xffffffff, a, 4);
        a += __shfl_xor_sync(0xffffffff, a, 2);
        a += __shfl_xor_sync(0xffffffff, a, 1);
        if (l8 == 0) contrib[t][st][il] = xn[(size_t(t) * HC + st) * n + i] / (1.0f + __expf(-a));
    }
    if (xres && l8 == 0 && blockIdx.x * 8 + il < n)   // the combined residual (k_hc_down2 has read x)
        for (int t = 0; t < TT; ++t) {
            const size_t k = (size_t(t) * HC + st) * n + i;
            xres[k] = __fmaf_rn(cout[size_t(t) * n + i], wvb[t * HC + st], xres[k]);
        }
    __syncthreads();
    if (threadIdx.x < 8 * TT) {
        const int t = threadIdx.x / 8, c8 = threadIdx.x % 8, ii = blockIdx.x * 8 + c8;
        if (ii < n) {
            float m = 0.0f;
            for (int s2 = 0; s2 < HC; ++s2) m += contrib[t][s2][c8];
            mixed[size_t(t) * n + ii] = m * (1.0f / HC);
        }
    }
}

__global__ void k_hc_combine(float* x, const float* out, const float* inject, int n, int hc, int T);

// comb: the previous mixer's output and inject to combine into x first (x += out * 2 sigmoid(inject /
// hc)): folded into the v2 kernels, else a k_hc_combine launch
struct HcComb {
    float* x = nullptr;
    const float* out = nullptr;
    const float* inj = nullptr;
};

template <int TT, typename WD, typename WU>
void hc_fused_launch(const float* x, const float* w_norm, WD Wd, const uint16_t* Wi, WU Wu, float* xn, float* part, float* mixed,
                     float* inject, int n, int hc, int rank, int n_inj, float eps, cudaStream_t st, HcComb comb = {}) {
    const int rows = rank + n_inj;
    const size_t smem = size_t(TT) * n * 4;
    static bool attr = false;
    if (!attr && smem > 48 * 1024) {
        ck(cudaFuncSetAttribute(k_hc_down<TT, WD>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024), "hc_down smem");
        attr = true;
    }
    // the v2 kernels (sw75) and the folded combine (sw78) where the shape allows; the first
    // kernels otherwise
    const bool down2 = n % 256 == 0 && n / 256 <= kHcDownMaxK && rank % kHcDownRpw == 0;
    const bool upv2 = rank == 320 && hc == 4;
    const bool fold = comb.out && down2 && upv2 && n % 4 == 0;
    if (comb.out && !fold) {
        k_hc_combine<<<dim3((n + 255) / 256, TT), 256, 0, st>>>(comb.x, comb.out, comb.inj, n, hc, TT);
        comb = HcComb{};
    }
    float* wv = part + size_t(TT) * hc * rows;   // [TT][hc] combine weights, after the partials
    if (down2) {
        static bool attr2 = false;
        if (!attr2 && smem > 48 * 1024) {
            ck(cudaFuncSetAttribute(k_hc_down2<TT, WD>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024), "hc_down2 smem");
            attr2 = true;
        }
        constexpr int per = kHcDownWarps * kHcDownRpw;
        // FLASHRT_HC_EARLY=0: a plain launch. Otherwise a programmatic dependent of the kernel
        // before it, which k_moe_combine_db triggers at its start: the mix's weights (down here,
        // up in k_hc_up_mix2, which chains off this kernel's trigger) load during the miss wait
        static const bool early = [] {
            const char* e = std::getenv("FLASHRT_HC_EARLY");
            return !(e && e[0] == '0');
        }();
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((rows + per - 1) / per, hc);
        cfg.blockDim = dim3(32 * kHcDownWarps);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = early ? 1 : 0;
        ck(cudaLaunchKernelEx(&cfg, k_hc_down2<TT, WD>, x, w_norm, Wd, Wi, xn, part, n, rank, n_inj, eps, static_cast<const float*>(comb.out),
                              comb.inj, wv),
           "hc_down2");
    } else
        k_hc_down<TT, WD><<<dim3((rows + 15) / 16, hc), 512, smem, st>>>(x, w_norm, Wd, Wi, xn, part, n, rank, n_inj, eps);
    if (upv2) {
        // a programmatic dependent of k_hc_down2. One token: weights loaded before the wait; windows:
        // after the preamble (earlier loads were slower there; test_hc_decode, sw75)
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((n + 7) / 8);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        ck(cudaLaunchKernelEx(&cfg, TT == 1 ? k_hc_up_mix2<TT, WU, 5> : k_hc_up_mix2<TT, WU, 5, true>, Wu, static_cast<const float*>(part), n_inj,
                              1.0f / hc, static_cast<const float*>(xn), mixed, inject, n, comb.x, comb.out, static_cast<const float*>(wv)),
           "hc_up_mix2");
    } else
        k_hc_up_mix<TT, WU><<<(n + 7) / 8, 256, 0, st>>>(Wu, part, n_inj, 1.0f / hc, xn, mixed, inject, n, rank);
}

// Q8P (kTypeQ8P) -> BF16, for the hc paths that multiply BF16 (one thread per element)
__global__ void k_q8p_to_bf16(const int8_t* q, const __half* d, __nv_bfloat16* out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(float(q[i]) * __half2float(d[i / 32]));
}

// k_gated_mean with xn recomputed from x: xn[t][s][i] = x[t][s][i] * inv[t * hc + s] * w[s][i]
// (the norm's own expression, so the same values), sparing the norm's write of xn; gate float or BF16
__device__ __forceinline__ float gate_f(float v) { return v; }
__device__ __forceinline__ float gate_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename GT>
__global__ void k_gated_mean_x(const float* x, const float* inv, const float* w, const GT* gate, float* mixed, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t k = (size_t(t) * hc + s) * n + i;
        const float xn = x[k] * inv[size_t(t) * hc + s] * w[size_t(s) * n + i];
        acc += xn / (1.0f + __expf(-gate_f(gate[k])));
    }
    mixed[size_t(t) * n + i] = acc * (1.0f / hc);
}

__global__ void k_gated_mean(const float* xn, const float* gate, float* mixed, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t k = (size_t(t) * hc + s) * n + i;
        acc += xn[k] / (1.0f + __expf(-gate[k]));
    }
    mixed[size_t(t) * n + i] = acc * (1.0f / hc);
}

__global__ void k_hc_combine(float* x, const float* out, const float* inject, int n, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= T) return;
    const float o = out[size_t(t) * n + i];
    for (int s = 0; s < hc; ++s) {
        const float wv = 2.0f / (1.0f + __expf(-inject[t * hc + s] / hc));
        x[(size_t(t) * hc + s) * n + i] += o * wv;
    }
}

}  // namespace

namespace {
// hc_mix, after hc_combine(x, comb_out, comb_inject) if comb_out is given (fused into the norm
// in prefill)
void hc_mix_impl(const BlockCtx& c, int il, int which, float* x, const float* comb_out, const float* comb_inject, int T, float* mixed,
                 float* inject, float* xn_out);
}  // namespace

void hc_decode_raw(int T, float* x, const float* w_norm, const void* down_q8p, const void* inject_bf16, const void* up_bf16, float* xn,
                   float* part, float* mixed, float* inject, int n, int rank, float eps, cudaStream_t st, const float* comb_out,
                   const float* comb_inject) {
    const HcComb cb{comb_out ? x : nullptr, comb_out, comb_inject};
    const int8_t* q = static_cast<const int8_t*>(down_q8p);
    const WQ8P wd{q, reinterpret_cast<const __half*>(q + size_t(rank) * 4 * n)};
    const WBf16 wu{static_cast<const uint16_t*>(up_bf16)};
    const uint16_t* wi = static_cast<const uint16_t*>(inject_bf16);
    switch (T) {
        case 1: hc_fused_launch<1>(x, w_norm, wd, wi, wu, xn, part, mixed, inject, n, 4, rank, 4, eps, st, cb); break;
        case 2: hc_fused_launch<2>(x, w_norm, wd, wi, wu, xn, part, mixed, inject, n, 4, rank, 4, eps, st, cb); break;
        case 3: hc_fused_launch<3>(x, w_norm, wd, wi, wu, xn, part, mixed, inject, n, 4, rank, 4, eps, st, cb); break;
        default: hc_fused_launch<4>(x, w_norm, wd, wi, wu, xn, part, mixed, inject, n, 4, rank, 4, eps, st, cb); break;
    }
    ck(cudaGetLastError(), "hc_decode_raw");
}

void hc_mix(const BlockCtx& c, int il, int which, const float* x, int T, float* mixed, float* inject, float* xn_out) {
    hc_mix_impl(c, il, which, const_cast<float*>(x), nullptr, nullptr, T, mixed, inject, xn_out);
}

void hc_combine_mix(const BlockCtx& c, int il, int which, float* x, const float* out, const float* comb_inject, int T, float* mixed,
                    float* inject) {
    hc_mix_impl(c, il, which, x, out, comb_inject, T, mixed, inject, nullptr);
}

namespace {
void hc_mix_impl(const BlockCtx& c, int il, int which, float* x, const float* comb_out, const float* comb_inject, int T, float* mixed,
                 float* inject, float* xn_out) {
    const Spec& s = c.s;
    const int n = s.d_model, hc = s.hc_count, hcd = hc * n;
    std::string pre;
    if (which == 2) pre = "output_hc_";
    else if (which == 3) pre = "blk." + std::to_string(il) + ".nextn.hc_head_";
    else pre = "blk." + std::to_string(il) + (which == 0 ? ".hc_attn_" : ".hc_ffn_");
    const GpuTensor& w_norm = c.w.get(pre + "norm.weight");
    const GpuTensor& w_down = c.w.get(pre + "down.weight");
    const GpuTensor& w_up = c.w.get(pre + "up.weight");

    float* xn = xn_out ? xn_out : c.scratch.f32;                          // [T][hcd]
    float* lo = c.scratch.f32 + size_t(T) * hcd;                          // [T][rank]
    float* gate = lo + size_t(T) * s.hc_rank;                             // [T][hcd]
    const GpuTensor* w_inj = which < 2 ? &c.w.get(pre + "inject.weight") : nullptr;
    using ggml_type::kBF16;
    // Q8P down/up (GpuWeights' hc conversion): the fused decode kernels read it directly; the other
    // paths multiply a BF16 copy dequantized into scratch below
    const bool q8d = w_down.type == kTypeQ8P, q8u = w_up.type == kTypeQ8P;
    const bool bf_or_q8 = (w_down.type == kBF16 || q8d) && (w_up.type == kBF16 || q8u);
    const bool v4 = n % 4 == 0 && n / 4 <= 1024 && (n / 4) % 32 == 0;
    const bool prefill_bf16 = v4 && T >= kGemmMinTokens && bf_or_q8 && (!w_inj || w_inj->type == kBF16);
    const bool decode_fused = T <= 4 && hc == 4 && s.hc_rank % 64 == 0 && T * s.hc_rank <= 4096 && n % 8 == 0 &&
                              size_t(T) * n * 4 <= 96 * 1024 && bf_or_q8 && (!w_inj || w_inj->type == kBF16);
    if (comb_out && !(prefill_bf16 && hc == 4) && !decode_fused) {   // not fused: the combine first
        hc_combine(c, x, comb_out, comb_inject, T);
        comb_out = nullptr;
    }
    if (decode_fused) {   // (the combine, if any, goes into hc_fused_launch)
        // decode steps and verify windows: norm + down + inject in one kernel, then up + silu +
        // gated mean in another, each weight read once for the T tokens
        const int n_inj = w_inj ? hc : 0;
        float* part = lo;   // [T][hc][rows], fits: lo is followed by gate [T][hcd]
        const uint16_t* wi = w_inj ? static_cast<const uint16_t*>(w_inj->dev) : nullptr;
        auto q8p = [](const GpuTensor& t) {
            const int8_t* q = static_cast<const int8_t*>(t.dev);
            return WQ8P{q, reinterpret_cast<const __half*>(q + size_t(t.rows()) * t.cols())};
        };
        auto bf = [](const GpuTensor& t) { return WBf16{static_cast<const uint16_t*>(t.dev)}; };
        auto run = [&](auto tt) {
            constexpr int TT = decltype(tt)::value;
            const float* wn = static_cast<const float*>(w_norm.dev);
            const float eps = float(s.rms_eps);
            const HcComb cb{comb_out ? x : nullptr, comb_out, comb_inject};
            if (q8d && q8u) hc_fused_launch<TT>(x, wn, q8p(w_down), wi, q8p(w_up), xn, part, mixed, inject, n, hc, s.hc_rank, n_inj, eps, c.stream, cb);
            else if (q8d) hc_fused_launch<TT>(x, wn, q8p(w_down), wi, bf(w_up), xn, part, mixed, inject, n, hc, s.hc_rank, n_inj, eps, c.stream, cb);
            else if (q8u) hc_fused_launch<TT>(x, wn, bf(w_down), wi, q8p(w_up), xn, part, mixed, inject, n, hc, s.hc_rank, n_inj, eps, c.stream, cb);
            else hc_fused_launch<TT>(x, wn, bf(w_down), wi, bf(w_up), xn, part, mixed, inject, n, hc, s.hc_rank, n_inj, eps, c.stream, cb);
        };
        switch (T) {
            case 1: run(std::integral_constant<int, 1>{}); break;
            case 2: run(std::integral_constant<int, 2>{}); break;
            case 3: run(std::integral_constant<int, 3>{}); break;
            default: run(std::integral_constant<int, 4>{}); break;
        }
        ck(cudaGetLastError(), "hc_mix");
        return;
    }
    GpuTensor down_bf, up_bf;   // Q8P down/up dequantized for the paths below
    if (q8d || q8u) {
        auto deq = [&](const GpuTensor& t, void* dst, GpuTensor& out) {
            const size_t ne = size_t(t.rows()) * t.cols();
            const int8_t* q = static_cast<const int8_t*>(t.dev);
            k_q8p_to_bf16<<<unsigned((ne + 255) / 256), 256, 0, c.stream>>>(q, reinterpret_cast<const __half*>(q + ne),
                                                                           static_cast<__nv_bfloat16*>(dst), ne);
            out = t;
            out.dev = dst;
            out.type = kBF16;
            out.bytes = ne * 2;
        };
        const size_t nd = size_t(w_down.rows()) * w_down.cols(), nu = size_t(w_up.rows()) * w_up.cols();
        if (c.scratch.hc_bf16_bytes < (nd + nu) * 2) throw std::runtime_error("hc_mix: hc BF16 scratch too small");
        if (q8d) deq(w_down, c.scratch.hc_bf16, down_bf);
        if (q8u) deq(w_up, static_cast<char*>(c.scratch.hc_bf16) + nd * 2, up_bf);
    }
    const GpuTensor& wd = q8d ? down_bf : w_down;
    const GpuTensor& wu = q8u ? up_bf : w_up;
    if (prefill_bf16) {
        // prefill: the norm writes xn in BF16 too, into the GEMM workspace, and the down and inject
        // products read it there (before the up product, which reuses that space)
        BlockScratch& bs = c.scratch;
        const size_t ws = gemm::workspace_bytes(hcd, T);
        if (bs.gemm_ws_bytes < ws) {
            if (bs.gemm_ws) cudaFree(bs.gemm_ws);
            ck(cudaMalloc(&bs.gemm_ws, ws), "cudaMalloc gemm workspace");
            bs.gemm_ws_bytes = ws;
        }
        auto* xb = static_cast<__nv_bfloat16*>(gemm::bf16_staging(bs.gemm_ws, bs.gemm_ws_bytes, hcd, T));
        // without xn_out, xn is not written: the gated mean recomputes it from x and 1 / rms
        float* xnw = xn_out ? xn : nullptr;
        float* inv = xn_out ? nullptr : xn;   // [T][hc], in xn's space
        const float* wn = static_cast<const float*>(w_norm.dev);
        // with 4 streams, one block per token does the combine (if any), the norm and the inject
        // product (4 outputs: a GEMM of its own ran at a fraction of bandwidth)
        const bool fuse_inj = hc == 4 && w_inj && w_inj->rows() == 4;
        if (hc == 4)
            k_hc_combine_norm4<<<T, n / 4, 0, c.stream>>>(x, comb_out, comb_inject, wn, xnw, xb, n, float(s.rms_eps), inv,
                                                         fuse_inj ? static_cast<const uint16_t*>(w_inj->dev) : nullptr, inject);
        else
            k_grouped_rms_norm_v4<<<T * hc, n / 4, 0, c.stream>>>(x, wn, xnw, n, hc, float(s.rms_eps), xb, inv);
        gemm::gemm_bf16(wd.dev, xb, lo, hcd, s.hc_rank, T, c.stream);
        if (w_inj && !fuse_inj) gemm::gemm_bf16(w_inj->dev, xb, inject, hcd, w_inj->rows(), T, c.stream);
        k_scale_silu<<<(T * s.hc_rank + 255) / 256, 256, 0, c.stream>>>(lo, T * s.hc_rank, 1.0f / hc);
        // the up product writes the gate in BF16 (half the traffic of gate and gated mean, sw59)
        if (!xn_out && wu.type == kBF16) {
            gemm::gemm_bf16_out(wu.dev, lo, gate, s.hc_rank, hcd, T, bs.gemm_ws, bs.gemm_ws_bytes, c.stream);
            k_gated_mean_x<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(x, inv, wn, reinterpret_cast<const __nv_bfloat16*>(gate), mixed, n,
                                                                         hc, T);
        } else {
            linear(c, wu, lo, gate, T);
            if (xn_out) k_gated_mean<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(xn, gate, mixed, n, hc, T);
            else k_gated_mean_x<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(x, inv, wn, static_cast<const float*>(gate), mixed, n, hc, T);
        }
        ck(cudaGetLastError(), "hc_mix");
        return;
    }
    if (v4)
        k_grouped_rms_norm_v4<<<T * hc, n / 4, 0, c.stream>>>(x, static_cast<const float*>(w_norm.dev), xn, n, hc, float(s.rms_eps), nullptr,
                                                             nullptr);
    else
        k_grouped_rms_norm<<<T * hc, 256, 0, c.stream>>>(x, static_cast<const float*>(w_norm.dev), xn, n, hc, float(s.rms_eps));
    linear(c, wd, xn, lo, T);
    k_scale_silu<<<(T * s.hc_rank + 255) / 256, 256, 0, c.stream>>>(lo, T * s.hc_rank, 1.0f / hc);
    linear(c, wu, lo, gate, T);
    k_gated_mean<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(xn, gate, mixed, n, hc, T);
    if (w_inj) linear(c, *w_inj, xn, inject, T);
    ck(cudaGetLastError(), "hc_mix");
}
}  // namespace

void rms_norm_rows(const BlockCtx& c, const float* x, const float* w, float* y, int n, int groups, int rows) {
    k_grouped_rms_norm<<<rows, 256, 0, c.stream>>>(x, w, y, n, groups, float(c.s.rms_eps));
    ck(cudaGetLastError(), "rms_norm_rows");
}

namespace {
__global__ void k_hc_init(const float* emb, float* x, int n, int hc) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = emb[size_t(t) * n + i];
        for (int s = 0; s < hc; ++s) x[(size_t(t) * hc + s) * n + i] = v;
    }
}
}  // namespace

void hc_init(const BlockCtx& c, const float* emb, float* x, int T) {
    const int n = c.s.d_model;
    k_hc_init<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(emb, x, n, c.s.hc_count);
    ck(cudaGetLastError(), "hc_init");
}

void hc_combine(const BlockCtx& c, float* x, const float* out, const float* inject, int T) {
    const int n = c.s.d_model;
    k_hc_combine<<<dim3((n + 255) / 256, T), 256, 0, c.stream>>>(x, out, inject, n, c.s.hc_count, T);
    ck(cudaGetLastError(), "hc_combine");
}

}  // namespace flashrt::qwen4exp
