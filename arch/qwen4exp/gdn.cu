// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// Gated DeltaNet: the conv, the delta rule (decode, windows, and chunked on tensor cores for
// prefill), states and rewinds.
#include "arch/qwen4exp/blocks_common.cuh"

#include "kernels/cuda/ggml_gemv.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flashrt::qwen4exp {

namespace {

// one thread per channel: causal conv over the (conv-1)-token history and T new tokens, silu;
// the history is advanced to the last (conv-1) inputs
// norm_heads > 0: the first norm_heads blocks are each one q or k head (blockDim = head dim) and
// L2-normalise it per token (as k_l2_norm)
__global__ void k_gdn_conv(const float* qkv, float* hist, const float* w, float* y, int channels, int T, int K, int norm_heads, float eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool norm = int(blockIdx.x) < norm_heads;   // block-uniform; such blocks are full
    if (c >= channels) return;
    float win[8];
    for (int k = 0; k < K - 1; ++k) win[k] = hist[size_t(k) * channels + c];
    for (int t = 0; t < T; ++t) {
        win[K - 1] = qkv[size_t(t) * channels + c];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += win[k] * w[size_t(c) * K + k];
        float v = acc / (1.0f + __expf(-acc));
        if (norm) v *= rsqrtf(block_sum(v * v) + eps);
        y[size_t(t) * channels + c] = v;
        for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
    }
    for (int k = 0; k < K - 1; ++k) hist[size_t(k) * channels + c] = win[k];
}

// The same conv for many tokens, parallel over (channel, token): input t' < 0 comes from the
// history. k_gdn_conv_hist then advances the history (after every token has read it).
// norm_heads > 0 (blockDim = head dim): the first norm_heads blocks of a token are its q and k
// heads, L2-normalised here (as k_l2_norm; sw81)
__global__ void k_gdn_conv_par(const float* qkv, const float* hist, const float* w, float* y, int channels, int T, int K, int norm_heads,
                               float eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const bool norm = int(blockIdx.x) < norm_heads;   // block-uniform; such blocks are full
    if (c >= channels) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int tt = t - (K - 1) + k;
        const float v = tt >= 0 ? qkv[size_t(tt) * channels + c] : hist[size_t(K - 1 + tt) * channels + c];
        acc += v * w[size_t(c) * K + k];
    }
    float out = acc / (1.0f + __expf(-acc));
    if (norm) out *= rsqrtf(block_sum(out * out) + eps);
    y[size_t(t) * channels + c] = out;
}
__global__ void k_gdn_conv_hist(const float* qkv, float* hist, int channels, int T, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[8];
    for (int k = 0; k < K - 1; ++k) {   // new history row k is input T - (K-1) + k
        const int tt = T - (K - 1) + k;
        win[k] = tt >= 0 ? qkv[size_t(tt) * channels + c] : hist[size_t(K - 1 + tt) * channels + c];
    }
    for (int k = 0; k < K - 1; ++k) hist[size_t(k) * channels + c] = win[k];
}

// one block per (token, head) of `dim` values: x /= sqrt(sum x^2 + eps), in place
__global__ void k_l2_norm(float* x, int dim, int stride_tok, int heads, float eps) {
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    float* v = x + size_t(t) * stride_tok + size_t(h) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += v[i] * v[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] *= inv;
}

// per (token, value head): g = softplus(alpha + dt_bias) * a, beta = sigmoid(beta)
__global__ void k_gdn_gates(float* alpha, float* beta, const float* dt_bias, const float* a, int heads, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * T) return;
    const int h = i % heads;
    const float z = alpha[i] + dt_bias[h];
    const float sp = z > 20.0f ? z : log1pf(__expf(z));
    alpha[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + __expf(-beta[i]));
}

// one block per value head, one thread per value row i; S is [head][j][i]. For each token:
// S *= exp(g); d = beta * (v - S k); S += d k^T; o = S q / sqrt(dk). Key head = h % k_heads.
__global__ void k_gdn_delta(float* S, const float* conv_out, const float* g, const float* beta, float* o, int T,
                            int k_heads, int v_heads, int dk, int channels) {
    extern __shared__ float sh[];   // q[dk], k[dk]
    float* qs = sh;
    float* ks = sh + dk;
    const int h = blockIdx.x, i = threadIdx.x, hk = h % k_heads;
    float* Sh = S + size_t(h) * dk * dk;
    const float scale = rsqrtf(float(dk));
    for (int t = 0; t < T; ++t) {
        const float* row = conv_out + size_t(t) * channels;
        for (int j = threadIdx.x; j < dk; j += blockDim.x) {
            qs[j] = row[size_t(hk) * dk + j] * scale;
            ks[j] = row[size_t(k_heads) * dk + size_t(hk) * dk + j];
        }
        __syncthreads();
        const float decay = __expf(g[t * v_heads + h]);
        const float b = beta[t * v_heads + h];
        const float vi = row[size_t(2 * k_heads) * dk + size_t(h) * dk + i];
        float sk = 0.0f;
        for (int j = 0; j < dk; ++j) {
            const float sv = Sh[size_t(j) * dk + i] * decay;
            Sh[size_t(j) * dk + i] = sv;
            sk += sv * ks[j];
        }
        const float d = b * (vi - sk);
        float oi = 0.0f;
        for (int j = 0; j < dk; ++j) {
            const float sv = Sh[size_t(j) * dk + i] + d * ks[j];
            Sh[size_t(j) * dk + i] = sv;
            oi += sv * qs[j];
        }
        o[(size_t(t) * v_heads + h) * dk + i] = oi;
        __syncthreads();
    }
}

// The same delta rule with the state in registers: one block per (head, 32 columns), 32 x 8
// threads; thread (x, y) holds S[j][i] for column i = 32 * blockIdx.y + x and rows
// j = y * DK/8 .. +DK/8, so S is read and written once per call however many tokens it has.
// S_in -> S_out (the same buffer in a plain call); with S_bak, the state before the call is
// saved there too (a speculative window's backup). A rewind replays from the backup.
template <int DK>
__global__ void k_gdn_delta_reg(const float* S_in, float* S_out, float* S_bak, const float* conv_out, const float* g, const float* beta,
                                float* o, int T, int k_heads, int v_heads, int channels) {
    constexpr int JG = 8, JPT = DK / JG;
    __shared__ float qs[DK], ks[DK];
    __shared__ float red[JG][33];
    const int h = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y, i = blockIdx.y * 32 + tx, hk = h % k_heads;
    const int tid = ty * 32 + tx;
    const float* Si = S_in + size_t(h) * DK * DK;
    float st[JPT];
#pragma unroll
    for (int jj = 0; jj < JPT; ++jj) st[jj] = Si[size_t(ty * JPT + jj) * DK + i];
    if (S_bak)
#pragma unroll
        for (int jj = 0; jj < JPT; ++jj) S_bak[size_t(h) * DK * DK + size_t(ty * JPT + jj) * DK + i] = st[jj];
    const float scale = rsqrtf(float(DK));
    for (int t = 0; t < T; ++t) {
        const float* row = conv_out + size_t(t) * channels;
        for (int j = tid; j < DK; j += 32 * JG) {
            qs[j] = row[size_t(hk) * DK + j] * scale;
            ks[j] = row[size_t(k_heads) * DK + size_t(hk) * DK + j];
        }
        __syncthreads();
        const float decay = __expf(g[t * v_heads + h]);
        float sk = 0.0f;
#pragma unroll
        for (int jj = 0; jj < JPT; ++jj) {
            st[jj] *= decay;
            sk += st[jj] * ks[ty * JPT + jj];
        }
        red[ty][tx] = sk;
        __syncthreads();
        float tot = 0.0f;
#pragma unroll
        for (int y = 0; y < JG; ++y) tot += red[y][tx];
        const float vi = row[size_t(2 * k_heads) * DK + size_t(h) * DK + i];
        const float d = beta[t * v_heads + h] * (vi - tot);
        float oi = 0.0f;
#pragma unroll
        for (int jj = 0; jj < JPT; ++jj) {
            st[jj] += d * ks[ty * JPT + jj];
            oi += st[jj] * qs[ty * JPT + jj];
        }
        __syncthreads();
        red[ty][tx] = oi;
        __syncthreads();
        if (ty == 0) {
            float r = 0.0f;
#pragma unroll
            for (int y = 0; y < JG; ++y) r += red[y][tx];
            o[(size_t(t) * v_heads + h) * DK + i] = r;
        }
        __syncthreads();
    }
    float* So = S_out + size_t(h) * DK * DK;
#pragma unroll
    for (int jj = 0; jj < JPT; ++jj) So[size_t(ty * JPT + jj) * DK + i] = st[jj];
}

// The same delta rule for long calls (prefill). Four lanes share two columns of S, 32 rows each,
// so a token's column sums are shuffles, with no block barrier. The tokens' q, k, v, gate and beta
// come through shared memory in tiles of 8, three tiles in flight (cp.async), as the recurrence
// can not wait for DRAM once per token. Shared-memory reads bound it: a warp's q and k reads
// have 4 distinct addresses, so each lane reads them once per token for two columns, from a
// layout where those addresses are adjacent (step s of quarter qd at s * 16 + qd * 4). Two warps
// (32 columns) per block: the 192 blocks spread evenly over the SMs.
constexpr int kGdnTile = 8, kGdnStages = 3, kGdnColWarps = 2;
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}
template <int DK>
__global__ void __launch_bounds__(32 * kGdnColWarps) k_gdn_delta_col(const float* S_in, float* S_out, float* S_bak, const float* conv_out,
                                                                    const float* g, const float* beta, float* o, int T, int k_heads,
                                                                    int v_heads, int channels) {
    constexpr int R = DK / 4, NC = 16 * kGdnColWarps;   // rows per lane, columns per block
    __shared__ __align__(16) float qs[kGdnStages][kGdnTile][DK], ks[kGdnStages][kGdnTile][DK], vs[kGdnStages][kGdnTile][NC];
    __shared__ float gs[kGdnStages][kGdnTile], bs[kGdnStages][kGdnTile];
    const int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, hk = h % k_heads;
    const int qd = lane & 3, ic = warp * 16 + (lane >> 2) * 2, i = blockIdx.y * NC + ic, j0 = qd * R;
    const size_t q_off = size_t(hk) * DK, k_off = size_t(k_heads) * DK + size_t(hk) * DK,
                 v_off = size_t(2 * k_heads) * DK + size_t(h) * DK + size_t(blockIdx.y) * NC;
    const int n_tiles = (T + kGdnTile - 1) / kGdnTile;
    auto load = [&](int tile) {   // always commits a group (empty past the end), so the waits count right
        if (tile < n_tiles) {
            const int t0 = tile * kGdnTile, n = min(kGdnTile, T - t0), buf = tile % kGdnStages;
            for (int e = threadIdx.x; e < n * (DK / 4); e += blockDim.x) {   // 16-byte pieces: dims d .. d + 3
                const int tt = e / (DK / 4), d = (e % (DK / 4)) * 4, at = ((d % R) / 4) * 16 + (d / R) * 4;
                const float* row = conv_out + size_t(t0 + tt) * channels;
                cp_async16(&qs[buf][tt][at], row + q_off + d);
                cp_async16(&ks[buf][tt][at], row + k_off + d);
            }
            for (int e = threadIdx.x; e < n * (NC / 4); e += blockDim.x) {
                const int tt = e / (NC / 4), p = e % (NC / 4);
                cp_async16(&vs[buf][tt][p * 4], conv_out + size_t(t0 + tt) * channels + v_off + p * 4);
            }
            if (threadIdx.x < n) {
                cp_async4(&gs[buf][threadIdx.x], g + size_t(t0 + threadIdx.x) * v_heads + h);
                cp_async4(&bs[buf][threadIdx.x], beta + size_t(t0 + threadIdx.x) * v_heads + h);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    const float* Si = S_in + size_t(h) * DK * DK;
    float s0[R], s1[R];   // columns i, i + 1
#pragma unroll
    for (int jj = 0; jj < R; ++jj) {
        const float2 v = *reinterpret_cast<const float2*>(Si + size_t(j0 + jj) * DK + i);
        s0[jj] = v.x;
        s1[jj] = v.y;
    }
    if (S_bak)
#pragma unroll
        for (int jj = 0; jj < R; ++jj) *reinterpret_cast<float2*>(S_bak + size_t(h) * DK * DK + size_t(j0 + jj) * DK + i) = make_float2(s0[jj], s1[jj]);
    const float scale = rsqrtf(float(DK));
    for (int k = 0; k < kGdnStages - 1; ++k) load(k);
    for (int tile = 0; tile < n_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kGdnStages - 2));
        __syncthreads();   // this tile is in; the tile computed last is no longer read
        load(tile + kGdnStages - 1);
        const int buf = tile % kGdnStages, t0 = tile * kGdnTile, n = min(kGdnTile, T - t0);
        for (int tt = 0; tt < n; ++tt) {
            float kv[R], qv[R];
#pragma unroll
            for (int s = 0; s < R / 4; ++s) {
                const float4 k4 = *reinterpret_cast<const float4*>(&ks[buf][tt][s * 16 + qd * 4]);
                const float4 q4 = *reinterpret_cast<const float4*>(&qs[buf][tt][s * 16 + qd * 4]);
                kv[4 * s] = k4.x; kv[4 * s + 1] = k4.y; kv[4 * s + 2] = k4.z; kv[4 * s + 3] = k4.w;
                qv[4 * s] = q4.x; qv[4 * s + 1] = q4.y; qv[4 * s + 2] = q4.z; qv[4 * s + 3] = q4.w;
            }
            const float decay = __expf(gs[buf][tt]), b = bs[buf][tt];
            const float2 vi = *reinterpret_cast<const float2*>(&vs[buf][tt][ic]);
            float a0 = 0.0f, a1 = 0.0f, c0 = 0.0f, c1 = 0.0f;
#pragma unroll
            for (int jj = 0; jj < R; jj += 2) {
                s0[jj] *= decay;
                s0[jj + 1] *= decay;
                s1[jj] *= decay;
                s1[jj + 1] *= decay;
                a0 += s0[jj] * kv[jj];
                a1 += s0[jj + 1] * kv[jj + 1];
                c0 += s1[jj] * kv[jj];
                c1 += s1[jj + 1] * kv[jj + 1];
            }
            float sk0 = a0 + a1, sk1 = c0 + c1;
            sk0 += __shfl_xor_sync(~0u, sk0, 1);
            sk1 += __shfl_xor_sync(~0u, sk1, 1);
            sk0 += __shfl_xor_sync(~0u, sk0, 2);
            sk1 += __shfl_xor_sync(~0u, sk1, 2);
            const float d0 = b * (vi.x - sk0), d1 = b * (vi.y - sk1);
            a0 = a1 = c0 = c1 = 0.0f;
#pragma unroll
            for (int jj = 0; jj < R; jj += 2) {
                s0[jj] += d0 * kv[jj];
                s0[jj + 1] += d0 * kv[jj + 1];
                s1[jj] += d1 * kv[jj];
                s1[jj + 1] += d1 * kv[jj + 1];
                a0 += s0[jj] * qv[jj];
                a1 += s0[jj + 1] * qv[jj + 1];
                c0 += s1[jj] * qv[jj];
                c1 += s1[jj + 1] * qv[jj + 1];
            }
            float o0 = a0 + a1, o1 = c0 + c1;
            o0 += __shfl_xor_sync(~0u, o0, 1);
            o1 += __shfl_xor_sync(~0u, o1, 1);
            o0 += __shfl_xor_sync(~0u, o0, 2);
            o1 += __shfl_xor_sync(~0u, o1, 2);
            if (qd == 0) *reinterpret_cast<float2*>(o + (size_t(t0 + tt) * v_heads + h) * DK + i) = make_float2(o0 * scale, o1 * scale);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    float* So = S_out + size_t(h) * DK * DK;
#pragma unroll
    for (int jj = 0; jj < R; ++jj) *reinterpret_cast<float2*>(So + size_t(j0 + jj) * DK + i) = make_float2(s0[jj], s1[jj]);
}

// ---- the chunked (WY) form of the delta rule, for prefill, on fp16 tensor cores
// Per chunk of C tokens with cumulative log decay G_t (gamma_t = exp(G_t)) and start state S0:
//   the new values U solve (I + A) U = beta V - diag(beta gamma) K S0, A[t][s] = beta_t
//   exp(G_t - G_s) k_t.k_s for s < t; so U = U~ - W S0 with U~ = T diag(beta) V and
//   W = T diag(beta gamma) K, T = (I + A)^-1 (both free of S0);
//   O = diag(gamma) Q S0 + P U, P[t][s] = exp(G_t - G_s) q_t.k_s for s <= t;
//   S0' = gamma_C S0 + K^T diag(exp(G_C - G)) U.
// k_gdn_chunk_prep does the S0-free part for every (chunk, head) in parallel; k_gdn_chunk_state
// carries S0 through the chunks, one block per (head, NC value columns): the columns of S evolve
// independently. The products run on fp16 mma (fp32 accumulation); T is solved in fp32, and the
// state stays fp32 in the state kernel's registers (an fp16 copy feeds W S0 and Q S0), as in
// FLA's chunked kernels. Prep writes its products as ready-made mma A fragments (afrag) so that
// the state kernel loads each with one 16-byte load per lane.
constexpr int kGdnChunk = 64, kGdnSlab = 8;   // tokens per chunk; chunks per prep/state pass

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint4& r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm4t(uint4& r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// element (r, c) of a [rows][re] fp16 matrix in shared memory with its 16-byte pieces XOR-swizzled
// by the row: ldmatrix over 8 consecutive rows is conflict-free without padding (re >= 64)
__device__ __forceinline__ int gsw(int r, int c, int re) { return r * re + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7); }
// element (row, col) of a [rows][kd] fp16 matrix stored as m16n8k16 A fragments: 16 x 16 tiles in
// row-major tile order, each 32 lanes x 8 halves (a lane's 4 registers, one 16-byte load)
__device__ __forceinline__ int afrag(int row, int col, int kd) {
    const int tile = (row >> 4) * (kd >> 4) + (col >> 4);
    const int lane = (row & 7) * 4 + ((col & 7) >> 1), reg = ((row >> 3) & 1) + ((col >> 3) & 1) * 2;
    return tile * 256 + lane * 8 + reg * 2 + (col & 1);
}

// Per (chunk, head), 8 warps. Shared memory (48 KiB, two blocks per SM): K and Q as fp16
// [C][DK]; A^T fp32 [C][C]. The Q region then holds V (fp16 [C][DK]); the A region then holds
// T diag(beta) and T diag(beta gamma) (fp16 [C][C] each). All global loads are issued early:
// the block is latency-bound otherwise (sw71).
template <int DK>
__global__ void __launch_bounds__(256, 2) k_gdn_chunk_prep(const float* conv, const float* g, const float* beta, int T, int t_base,
                                                          int k_heads, int v_heads, int channels, __half* Qh, __half* Wf, __half* Kt,
                                                          __half* Pf, float* Ut, float* gcb) {
    static_assert(DK == 128, "state 128");
    constexpr int C = kGdnChunk, QK_IT = (C / 2) * (DK / 2) / 256, V_IT = C * (DK / 2) / 256;
    extern __shared__ __align__(16) unsigned char gsm_raw[];
    __half* Ks = reinterpret_cast<__half*>(gsm_raw);
    __half* Qs = Ks + C * DK;
    __half* Vs = Qs;
    float* At = reinterpret_cast<float*>(Qs + C * DK);   // At[s][t] = A[t][s]
    __half* T1 = reinterpret_cast<__half*>(At);
    __half* T2 = T1 + C * C;
    __shared__ float Gs[C], Bs[C];
    const int ci = blockIdx.x, h = blockIdx.y, hk = h % k_heads, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g8 = lane >> 2, tq = lane & 3;
    const int t0 = t_base + ci * C, n = min(C, T - t0);
    const size_t item = size_t(ci) * v_heads + h;
    const size_t q_off = size_t(hk) * DK, k_off = size_t(k_heads) * DK + size_t(hk) * DK, v_off = size_t(2 * k_heads) * DK + size_t(h) * DK;
    const float scale = rsqrtf(float(DK));
    // 1. loads: the decays and betas (warp 0), q and k in 2 x 2 pieces (tokens t, t + 1; dims d, d + 1)
    float ga = 0.0f, gb = 0.0f, ba = 0.0f, bb = 0.0f;
    if (warp == 0) {
        if (lane < n) {
            ga = g[size_t(t0 + lane) * v_heads + h];
            ba = beta[size_t(t0 + lane) * v_heads + h];
        }
        if (lane + 32 < n) {
            gb = g[size_t(t0 + lane + 32) * v_heads + h];
            bb = beta[size_t(t0 + lane + 32) * v_heads + h];
        }
    }
    float2 qv[QK_IT][2], kv[QK_IT][2];
#pragma unroll
    for (int it = 0; it < QK_IT; ++it) {
        const int e = tid + it * 256, t = (e / (DK / 2)) * 2, d = (e % (DK / 2)) * 2;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            qv[it][u] = kv[it][u] = make_float2(0.0f, 0.0f);
            if (t + u < n) {
                const float* row = conv + size_t(t0 + t + u) * channels;
                qv[it][u] = *reinterpret_cast<const float2*>(row + q_off + d);
                kv[it][u] = *reinterpret_cast<const float2*>(row + k_off + d);
            }
        }
    }
    // the cumulative log decay; padding tokens decay by 1 and have beta 0
    if (warp == 0) {
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float ua = __shfl_up_sync(~0u, ga, off), ub = __shfl_up_sync(~0u, gb, off);
            if (lane >= off) {
                ga += ua;
                gb += ub;
            }
        }
        gb += __shfl_sync(~0u, ga, 31);
        Gs[lane] = ga;
        Gs[lane + 32] = gb;
        Bs[lane] = ba;
        Bs[lane + 32] = bb;
    }
    __syncthreads();
    const float GC = Gs[C - 1];
    // 2. fp16 q and k here; Q^ = scale gamma Q (rows t) and K^ = exp(G_C - G) K transposed (rows d)
    //    as A fragments for the state kernel
    {
        __half* qh = Qh + item * C * DK;
        __half* kt = Kt + item * C * DK;
#pragma unroll
        for (int it = 0; it < QK_IT; ++it) {
            const int e = tid + it * 256, t = (e / (DK / 2)) * 2, d = (e % (DK / 2)) * 2;
            const float2 q0 = qv[it][0], q1 = qv[it][1], k0 = kv[it][0], k1 = kv[it][1];
            *reinterpret_cast<uint32_t*>(&Qs[gsw(t, d, DK)]) = pack_h2(q0.x, q0.y);
            *reinterpret_cast<uint32_t*>(&Qs[gsw(t + 1, d, DK)]) = pack_h2(q1.x, q1.y);
            *reinterpret_cast<uint32_t*>(&Ks[gsw(t, d, DK)]) = pack_h2(k0.x, k0.y);
            *reinterpret_cast<uint32_t*>(&Ks[gsw(t + 1, d, DK)]) = pack_h2(k1.x, k1.y);
            const float sa = scale * __expf(Gs[t]), sb = scale * __expf(Gs[t + 1]);
            *reinterpret_cast<uint32_t*>(&qh[afrag(t, d, DK)]) = pack_h2(sa * q0.x, sa * q0.y);
            *reinterpret_cast<uint32_t*>(&qh[afrag(t + 1, d, DK)]) = pack_h2(sb * q1.x, sb * q1.y);
            const float ka = __expf(GC - Gs[t]), kb = __expf(GC - Gs[t + 1]);
            *reinterpret_cast<uint32_t*>(&kt[afrag(d, t, C)]) = pack_h2(ka * k0.x, kb * k1.x);
            *reinterpret_cast<uint32_t*>(&kt[afrag(d + 1, t, C)]) = pack_h2(ka * k0.y, kb * k1.y);
        }
    }
    // v, used in step 5: loaded now, stored over Q once Q is done
    float2 vv[V_IT];
#pragma unroll
    for (int it = 0; it < V_IT; ++it) {
        const int e = tid + it * 256, t = e / (DK / 2), d = (e % (DK / 2)) * 2;
        vv[it] = t < n ? *reinterpret_cast<const float2*>(conv + size_t(t0 + t) * channels + v_off + d) : make_float2(0.0f, 0.0f);
    }
    __syncthreads();
    // 3. K K^T -> A^T (fp32, shared) and Q K^T -> P (scaled, A fragments, global): the lower
    //    triangle, warp = (product, row tile)
    {
        const int prod = warp >> 2, mt = warp & 3;
        const __half* Am = prod ? Qs : Ks;
        float acc[8][4] = {};
#pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            uint4 a;
            ldsm4(a, &Am[gsw(mt * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8, DK)]);
#pragma unroll
            for (int jp = 0; jp < 4; ++jp)
                if (jp <= mt) {
                    uint4 b;
                    ldsm4(b, &Ks[gsw(jp * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8, DK)]);
                    mma_f16(acc[2 * jp], a, b.x, b.y);
                    mma_f16(acc[2 * jp + 1], a, b.z, b.w);
                }
        }
        __half* pf = Pf + item * C * C;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int t = mt * 16 + g8 + hh * 8, s = j * 8 + tq * 2;
                const float e0 = s <= t ? __expf(Gs[t] - Gs[s]) : 0.0f, e1 = s + 1 <= t ? __expf(Gs[t] - Gs[s + 1]) : 0.0f;
                if (prod == 0) {
                    At[s * C + t] = s < t ? Bs[t] * e0 * acc[j][2 * hh] : 0.0f;
                    At[(s + 1) * C + t] = s + 1 < t ? Bs[t] * e1 * acc[j][2 * hh + 1] : 0.0f;
                } else if (j < 2 * mt + 2)
                    *reinterpret_cast<uint32_t*>(&pf[afrag(t, s, C)]) = pack_h2(scale * e0 * acc[j][2 * hh], scale * e1 * acc[j][2 * hh + 1]);
            }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < V_IT; ++it) {
        const int e = tid + it * 256, t = e / (DK / 2), d = (e % (DK / 2)) * 2;
        *reinterpret_cast<uint32_t*>(&Vs[gsw(t, d, DK)]) = pack_h2(vv[it].x, vv[it].y);
    }
    // 4. T = (I + A)^-1 in fp32 by column substitution in registers: warp w owns columns
    //    8w .. 8w + 7, lane rows lane and lane + 32; step s subtracts A[r][s] x[s] from the rows
    //    below s (x[s] final by then: A is strictly lower)
    float x0[8], x1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x0[j] = lane == warp * 8 + j ? 1.0f : 0.0f;
        x1[j] = lane + 32 == warp * 8 + j ? 1.0f : 0.0f;
    }
#pragma unroll 8
    for (int s = 0; s < 32; ++s) {
        const float a0 = At[s * C + lane], a1 = At[s * C + lane + 32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float xs = __shfl_sync(~0u, x0[j], s);
            x0[j] -= a0 * xs;
            x1[j] -= a1 * xs;
        }
    }
#pragma unroll 8
    for (int s = 32; s < C; ++s) {
        const float a1 = At[s * C + lane + 32];
#pragma unroll
        for (int j = 0; j < 8; ++j) x1[j] -= a1 * __shfl_sync(~0u, x1[j], s - 32);
    }
    __syncthreads();   // A is done: T diag(beta) and T diag(beta gamma) go over it
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = warp * 8 + j;
        const float b = Bs[c], bg = b * __expf(Gs[c]);
        T1[gsw(lane, c, C)] = __float2half_rn(x0[j] * b);
        T2[gsw(lane, c, C)] = __float2half_rn(x0[j] * bg);
        T1[gsw(lane + 32, c, C)] = __float2half_rn(x1[j] * b);
        T2[gsw(lane + 32, c, C)] = __float2half_rn(x1[j] * bg);
    }
    __syncthreads();
    // 5. W = T diag(beta gamma) K (A fragments, fp16) and U~ = T diag(beta) V (fp32 rows):
    //    warp = (product, row tile), the columns in two halves
    {
        const int prod = warp >> 2, mt = warp & 3;
        const __half* Tm = prod ? T1 : T2;
        const __half* Bm = prod ? Vs : Ks;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float acc[8][4] = {};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
                if (kk <= mt) {
                    uint4 a;
                    ldsm4(a, &Tm[gsw(mt * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8, C)]);
#pragma unroll
                    for (int jp = 0; jp < 4; ++jp) {
                        uint4 b;
                        ldsm4t(b, &Bm[gsw(kk * 16 + (lane & 15), (half * 8 + 2 * jp + (lane >> 4)) * 8, DK)]);
                        mma_f16(acc[2 * jp], a, b.x, b.y);
                        mma_f16(acc[2 * jp + 1], a, b.z, b.w);
                    }
                }
            if (prod == 0) {   // n-tiles 2q and 2q + 1 are the A fragment of k-step half * 4 + q
                __half* wf = Wf + item * C * DK;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 u = {pack_h2(acc[2 * q][0], acc[2 * q][1]), pack_h2(acc[2 * q][2], acc[2 * q][3]),
                                     pack_h2(acc[2 * q + 1][0], acc[2 * q + 1][1]), pack_h2(acc[2 * q + 1][2], acc[2 * q + 1][3])};
                    *reinterpret_cast<uint4*>(&wf[(mt * (DK / 16) + half * 4 + q) * 256 + lane * 8]) = u;
                }
            } else {
                float* ut = Ut + item * C * DK;
#pragma unroll
                for (int j = 0; j < 8; ++j)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const int t = mt * 16 + g8 + hh * 8, d = (half * 8 + j) * 8 + tq * 2;
                        *reinterpret_cast<float2*>(&ut[t * DK + d]) = make_float2(acc[j][2 * hh], acc[j][2 * hh + 1]);
                    }
            }
        }
    }
    if (tid == 0) gcb[item] = __expf(GC);
}

// Per (head, NC value columns), NW warps; warp w owns state rows (w % 8) * 16 .. + 16 and NTW
// n-tiles of 8 columns. Per chunk:
//   1. X = [W; Q^] S0 (rows 0..63 W, 64..127 Q^; fp16 S0 from shared memory);
//   2. W warps: U = U~ - X -> fp16 U in shared memory;
//   3. all: S0 = gamma_C S0 + K^ U (fp32, in registers); Q^ warps: O = X + P U;
//   4. the fp16 copy of S0 for the next chunk.
template <int DK, int NC, int NW, int MINB>
__global__ void __launch_bounds__(32 * NW, MINB) k_gdn_chunk_state(float* S, const __half* Qh, const __half* Wf, const __half* Kt,
                                                            const __half* Pf, const float* Ut, const float* gcb, float* o, int T, int t_base,
                                                            int n_chunks, int v_heads) {
    constexpr int C = kGdnChunk, NTW = NC / NW, SP = NC + 8;   // n-tiles per warp; padded row of the fp16 copies
    static_assert(NW % 8 == 0 && NTW % 2 == 0, "warps: 8 row tiles x column groups");
    extern __shared__ __align__(16) unsigned char gsm_raw[];
    __half* Sb = reinterpret_cast<__half*>(gsm_raw);   // [DK][SP]
    __half* Us = Sb + DK * SP;                          // [C][SP]
    const int h = blockIdx.x, i0 = blockIdx.y * NC, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int mt = warp & 7, nb = (warp >> 3) * NTW, g8 = lane >> 2, tq = lane & 3;
    float* Sh = S + size_t(h) * DK * DK;
    float st[NTW][4];
#pragma unroll
    for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int d = mt * 16 + g8 + hh * 8, c = (nb + j) * 8 + tq * 2;
            const float2 v = *reinterpret_cast<const float2*>(Sh + size_t(d) * DK + i0 + c);
            st[j][2 * hh] = v.x;
            st[j][2 * hh + 1] = v.y;
            *reinterpret_cast<uint32_t*>(&Sb[d * SP + c]) = pack_h2(v.x, v.y);
        }
    __syncthreads();
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int t0 = t_base + ci * C, n = min(C, T - t0);
        const size_t item = size_t(ci) * v_heads + h;
        // 1.
        const __half* Am = (mt < 4 ? Wf : Qh) + item * C * DK + size_t((mt & 3) * (DK / 16)) * 256 + lane * 8;
        float2 ut[NTW][2];
        if (mt < 4) {
            const float* u = Ut + item * C * DK + i0;
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                    ut[j][hh] = *reinterpret_cast<const float2*>(u + size_t(mt * 16 + g8 + hh * 8) * DK + (nb + j) * 8 + tq * 2);
        }
        float x[NTW][4] = {};
#pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            const uint4 a = *reinterpret_cast<const uint4*>(Am + kk * 256);
#pragma unroll
            for (int jp = 0; jp < NTW / 2; ++jp) {
                uint4 b;
                ldsm4t(b, &Sb[(kk * 16 + (lane & 15)) * SP + (nb + 2 * jp + (lane >> 4)) * 8]);
                mma_f16(x[2 * jp], a, b.x, b.y);
                mma_f16(x[2 * jp + 1], a, b.z, b.w);
            }
        }
        // 2.
        if (mt < 4)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                    *reinterpret_cast<uint32_t*>(&Us[(mt * 16 + g8 + hh * 8) * SP + (nb + j) * 8 + tq * 2]) =
                        pack_h2(ut[j][hh].x - x[j][2 * hh], ut[j][hh].y - x[j][2 * hh + 1]);
        __syncthreads();   // U is in; everyone is done reading S0's fp16 copy
        // 3.
        const float gc = gcb[item];
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) st[j][e] *= gc;
        const __half* Km = Kt + item * C * DK + size_t(mt * (C / 16)) * 256 + lane * 8;
        const __half* Pm = Pf + item * C * C + size_t((mt & 3) * (C / 16)) * 256 + lane * 8;
#pragma unroll
        for (int kk = 0; kk < C / 16; ++kk) {
            const uint4 a = *reinterpret_cast<const uint4*>(Km + kk * 256);
            const bool pu = mt >= 4 && kk <= mt - 4;
            uint4 p;
            if (pu) p = *reinterpret_cast<const uint4*>(Pm + kk * 256);
#pragma unroll
            for (int jp = 0; jp < NTW / 2; ++jp) {
                uint4 b;
                ldsm4t(b, &Us[(kk * 16 + (lane & 15)) * SP + (nb + 2 * jp + (lane >> 4)) * 8]);
                mma_f16(st[2 * jp], a, b.x, b.y);
                mma_f16(st[2 * jp + 1], a, b.z, b.w);
                if (pu) {
                    mma_f16(x[2 * jp], p, b.x, b.y);
                    mma_f16(x[2 * jp + 1], p, b.z, b.w);
                }
            }
        }
        if (mt >= 4)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int t = (mt - 4) * 16 + g8 + hh * 8;
                    if (t < n)
                        *reinterpret_cast<float2*>(o + (size_t(t0 + t) * v_heads + h) * DK + i0 + (nb + j) * 8 + tq * 2) =
                            make_float2(x[j][2 * hh], x[j][2 * hh + 1]);
                }
        // 4.
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
                *reinterpret_cast<uint32_t*>(&Sb[(mt * 16 + g8 + hh * 8) * SP + (nb + j) * 8 + tq * 2]) = pack_h2(st[j][2 * hh], st[j][2 * hh + 1]);
        __syncthreads();   // the fp16 S0 is in; everyone is done reading U
    }
#pragma unroll
    for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
            *reinterpret_cast<float2*>(Sh + size_t(mt * 16 + g8 + hh * 8) * DK + i0 + (nb + j) * 8 + tq * 2) = make_float2(st[j][2 * hh], st[j][2 * hh + 1]);
}

// Rewinds a history of H rows (oldest first, C values each) after a call of T inputs to its
// first n: row j = row j + n of [old history ; the call's inputs].
__global__ void k_hist_rewind(float* hist, const float* old, const float* rows, int H, int C, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (c >= C) return;
    const int q = j + n;
    hist[size_t(j) * C + c] = q < H ? old[size_t(q) * C + c] : rows[size_t(q - H) * C + c];
}

// per (token, head): y = rms_norm(o) * w * sigmoid(z)
__global__ void k_gated_rms_norm(const float* o, const float* w, const float* z, float* y, int dim, float eps) {
    const size_t base = size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += o[base + i] * o[base + i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y[base + i] = o[base + i] * inv * w[i] / (1.0f + __expf(-z[base + i]));
}

}  // namespace

namespace {
int gdn_channels(const Spec& s) { return 2 * s.ssm_groups * s.ssm_state + s.ssm_heads * s.ssm_state; }

}

GdnState alloc_gdn_state(const Spec& s) {
    GdnState st;
    ck(cudaMalloc(&st.S, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4), "cudaMalloc gdn S");
    ck(cudaMalloc(&st.conv, size_t(s.ssm_conv - 1) * gdn_channels(s) * 4), "cudaMalloc gdn conv");
    reset_gdn_state(s, st, nullptr);
    return st;
}

void reset_gdn_state(const Spec& s, GdnState& st, cudaStream_t stream) {
    ck(cudaMemsetAsync(st.S, 0, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4, stream), "memset gdn S");
    ck(cudaMemsetAsync(st.conv, 0, size_t(s.ssm_conv - 1) * gdn_channels(s) * 4, stream), "memset gdn conv");
}

void free_gdn_state(GdnState& st) {
    if (st.S) cudaFree(st.S);
    if (st.conv) cudaFree(st.conv);
    st = GdnState{};
}

GdnWindow alloc_gdn_window(const Spec& s, int max_tokens) {
    GdnWindow w;
    w.max_tokens = max_tokens;
    const size_t ch = gdn_channels(s), H = s.ssm_heads;
    ck(cudaMalloc(&w.S_bak, H * s.ssm_state * s.ssm_state * 4), "cudaMalloc gdn backup");
    ck(cudaMalloc(&w.conv_old, size_t(s.ssm_conv - 1) * ch * 4), "cudaMalloc gdn conv backup");
    ck(cudaMalloc(&w.qkv, size_t(max_tokens) * ch * 4), "cudaMalloc gdn window");
    ck(cudaMalloc(&w.conv, size_t(max_tokens) * ch * 4), "cudaMalloc gdn window");
    ck(cudaMalloc(&w.g, size_t(max_tokens) * H * 4), "cudaMalloc gdn window");
    ck(cudaMalloc(&w.beta, size_t(max_tokens) * H * 4), "cudaMalloc gdn window");
    return w;
}

void free_gdn_window(GdnWindow& w) {
    for (float* p : {w.S_bak, w.conv_old, w.qkv, w.conv, w.g, w.beta})
        if (p) cudaFree(p);
    w = GdnWindow{};
}

void gdn_mixer(const BlockCtx& c, int il, const float* x, int T, GdnState& st, float* out, float* o_inner, GdnWindow* win) {
    const Spec& s = c.s;
    const int ch = gdn_channels(s), H = s.ssm_heads, dk = s.ssm_state, inner = H * dk;
    if (s.ssm_conv > 8 || dk > 1024) throw std::runtime_error("gdn_mixer: unsupported shape");
    if (win && (T > win->max_tokens || dk != 128)) throw std::runtime_error("gdn_mixer: window too long or unsupported shape");
    float* qkv = c.scratch.f32;                    // [T][ch]
    float* conv = qkv + size_t(T) * ch;           // [T][ch]
    float* z = conv + size_t(T) * ch;             // [T][inner]
    float* alpha = z + size_t(T) * inner;         // [T][H]
    float* beta = alpha + size_t(T) * H;          // [T][H]
    float* o = beta + size_t(T) * H;              // [T][inner]
    float* fin = o + size_t(T) * inner;           // [T][inner]
    if (size_t(fin + size_t(T) * inner - c.scratch.f32) > c.scratch.f32_elems) throw std::runtime_error("gdn_mixer: scratch too small");
    if (win) {   // the rewind inputs are kept in the window buffers
        qkv = win->qkv;
        conv = win->conv;
        alpha = win->g;
        beta = win->beta;
        ck(cudaMemcpyAsync(win->conv_old, st.conv, size_t(s.ssm_conv - 1) * ch * 4, cudaMemcpyDeviceToDevice, c.stream), "gdn conv backup");
    }

    {
        const GpuTensor* ws[2] = {&c.w.layer(il, "attn_qkv.weight"), &c.w.layer(il, "attn_gate.weight")};
        float* ys[2] = {qkv, z};
        linear_shared(c, ws, ys, 2, x, T);
    }
    const float* dt_bias = static_cast<const float*>(c.w.layer(il, "ssm_dt.bias").dev);
    const float* ssm_a = static_cast<const float*>(c.w.layer(il, "ssm_a").dev);
    const LinearOut ab[2] = {{&c.w.layer(il, "ssm_alpha.weight"), alpha, 1, dt_bias, ssm_a}, {&c.w.layer(il, "ssm_beta.weight"), beta, 2}};
    const bool ab_fused = linear_multi_ok(ab, 2, T);   // decode: alpha, beta and the gates in one launch
    if (ab_fused) linear_multi(c, ab, 2, x, T);
    else {
        const GpuTensor* ws2[2] = {ab[1].W, ab[0].W};
        float* ys2[2] = {beta, alpha};
        linear_shared(c, ws2, ys2, 2, x, T);
    }
    const bool conv_l2 = dk == 128;   // the conv kernel normalises q and k (a block per head)
    if (T > 8) {
        const int bt = conv_l2 ? 128 : 256;
        k_gdn_conv_par<<<dim3((ch + bt - 1) / bt, T), bt, 0, c.stream>>>(qkv, st.conv,
                                                                        static_cast<const float*>(c.w.layer(il, "ssm_conv1d.weight").dev), conv,
                                                                        ch, T, s.ssm_conv, conv_l2 ? 2 * s.ssm_groups : 0, float(s.rms_eps));
        k_gdn_conv_hist<<<(ch + 255) / 256, 256, 0, c.stream>>>(qkv, st.conv, ch, T, s.ssm_conv);
    } else {
        k_gdn_conv<<<(ch + 127) / 128, 128, 0, c.stream>>>(qkv, st.conv, static_cast<const float*>(c.w.layer(il, "ssm_conv1d.weight").dev),
                                                          conv, ch, T, s.ssm_conv, conv_l2 ? 2 * s.ssm_groups : 0, float(s.rms_eps));
    }
    // L2-normalise the q and k heads (the first 2 * groups heads of each token's channels)
    if (!conv_l2) k_l2_norm<<<T * 2 * s.ssm_groups, 128, 0, c.stream>>>(conv, dk, ch, 2 * s.ssm_groups, float(s.rms_eps));
    if (!ab_fused) k_gdn_gates<<<(T * H + 127) / 128, 128, 0, c.stream>>>(alpha, beta, dt_bias, ssm_a, H, T);
    if (dk == 128 && T >= kGdnChunk && !win) {   // prefill: the chunked form on tensor cores (sw71)
        BlockScratch& bs = c.scratch;
        const size_t need = gdn_chunk_ws_bytes(s);
        if (bs.gdn_ws_bytes < need) {
            if (bs.gdn_ws) cudaFree(bs.gdn_ws);
            ck(cudaMalloc(&bs.gdn_ws, need), "cudaMalloc gdn chunk workspace");
            bs.gdn_ws_bytes = need;
        }
        gdn_delta_prefill(s, st.S, conv, alpha, beta, o, T, true, bs.gdn_ws, c.stream);
    } else if (dk == 128 && T >= 16)   // prefill (short batches, windows); decode keeps the block kernel
        k_gdn_delta_col<128><<<dim3(H, 128 / (16 * kGdnColWarps)), 32 * kGdnColWarps, 0, c.stream>>>(st.S, st.S, win ? win->S_bak : nullptr, conv, alpha, beta, o, T,
                                                                     s.ssm_groups, H, ch);
    else if (dk == 128)
        gdn_delta_decode(s, st.S, st.S, win ? win->S_bak : nullptr, conv, alpha, beta, o, T, c.stream);
    else k_gdn_delta<<<H, dk, size_t(2) * dk * 4, c.stream>>>(st.S, conv, alpha, beta, o, T, s.ssm_groups, H, dk, ch);
    if (o_inner) ck(cudaMemcpyAsync(o_inner, o, size_t(T) * inner * 4, cudaMemcpyDeviceToDevice, c.stream), "copy o");
    const GpuTensor& w_out = c.w.layer(il, "ssm_out.weight");
    const float* w_norm = static_cast<const float*>(c.w.layer(il, "ssm_norm.weight").dev);
    if (q8_act(w_out, T) && gemv::gated_rms_norm_q8_1_ok(dk, H)) {   // decode: the norm writes ssm_out's activations
        gemv::gated_rms_norm_q8_1(o, w_norm, z, dk, H, float(s.rms_eps), T, c.scratch.q8, c.stream);
        gemv::matvec_q(w_out.type, w_out.dev, c.scratch.q8, out, inner, w_out.rows(), T, c.stream);
    } else {
        k_gated_rms_norm<<<T * H, 128, 0, c.stream>>>(o, w_norm, z, fin, dk, float(s.rms_eps));
        linear(c, w_out, fin, out, T);
    }
    ck(cudaGetLastError(), "gdn_mixer");
}

void gdn_delta_decode(const Spec& s, const float* S_in, float* S_out, float* S_bak, const float* conv, const float* g, const float* beta,
                      float* o, int T, cudaStream_t stream) {
    if (s.ssm_state != 128) throw std::runtime_error("gdn_delta_decode: state size must be 128");
    k_gdn_delta_reg<128><<<dim3(s.ssm_heads, 128 / 32), dim3(32, 8), 0, stream>>>(S_in, S_out, S_bak, conv, g, beta, o, T, s.ssm_groups,
                                                                                s.ssm_heads, gdn_channels(s));
    ck(cudaGetLastError(), "gdn_delta_decode");
}

void gdn_rewind(const BlockCtx& c, GdnState& st, const GdnWindow& win, int T, int n) {
    if (n >= T) return;
    const Spec& s = c.s;
    const int ch = gdn_channels(s);
    if (n < 0 || T > win.max_tokens) throw std::runtime_error("gdn_rewind: bad window");
    float* o = c.scratch.f32;   // the replay's outputs are not needed
    gdn_delta_decode(s, win.S_bak, st.S, nullptr, win.conv, win.g, win.beta, o, n, c.stream);
    k_hist_rewind<<<dim3((ch + 255) / 256, s.ssm_conv - 1), 256, 0, c.stream>>>(st.conv, win.conv_old, win.qkv, s.ssm_conv - 1, ch, n);
    ck(cudaGetLastError(), "gdn_rewind");
}

size_t gdn_chunk_ws_bytes(const Spec& s) {
    const size_t items = size_t(kGdnSlab) * s.ssm_heads, C = kGdnChunk, dk = s.ssm_state;
    return items * (C * (3 * dk + C) * 2 + C * dk * 4 + 4);   // Q^, W, K^ and P (fp16), U~ (fp32), gamma_C
}

namespace {
template <int NC, int NW, int MINB>
void gdn_state_launch(float* S, const __half* Qh, const __half* Wf, const __half* Kt, const __half* Pf, const float* Ut, const float* gcb,
                      float* o, int T, int t_base, int n_chunks, int H, cudaStream_t stream) {
    constexpr int DK = 128;
    const size_t smem = size_t(DK + kGdnChunk) * (NC + 8) * 2;
    static bool attr = false;
    if (!attr) {
        ck(cudaFuncSetAttribute(k_gdn_chunk_state<DK, NC, NW, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)), "gdn state smem");
        attr = true;
    }
    k_gdn_chunk_state<DK, NC, NW, MINB><<<dim3(H, DK / NC), 32 * NW, smem, stream>>>(S, Qh, Wf, Kt, Pf, Ut, gcb, o, T, t_base, n_chunks, H);
}
}  // namespace

void gdn_delta_prefill(const Spec& s, float* S, const float* conv, const float* g, const float* beta, float* o, int T, bool chunked, void* ws,
                       cudaStream_t stream) {
    const int H = s.ssm_heads, dk = s.ssm_state, groups = s.ssm_groups, ch = gdn_channels(s);
    if (dk != 128) throw std::runtime_error("gdn_delta_prefill: state 128 only");
    if (!chunked) {
        k_gdn_delta_col<128><<<dim3(H, 128 / (16 * kGdnColWarps)), 32 * kGdnColWarps, 0, stream>>>(S, S, nullptr, conv, g, beta, o, T, groups,
                                                                                                    H, ch);
        ck(cudaGetLastError(), "gdn column");
        return;
    }
    constexpr int C = kGdnChunk, DK = 128;
    const size_t items = size_t(kGdnSlab) * H;
    __half* Qh = static_cast<__half*>(ws);
    __half* Wf = Qh + items * C * DK;
    __half* Kt = Wf + items * C * DK;
    __half* Pf = Kt + items * C * DK;
    float* Ut = reinterpret_cast<float*>(Pf + items * C * C);
    float* gcb = Ut + items * C * DK;
    const size_t smem_prep = size_t(3 * C * DK) * 2;   // K, Q (fp16) and A^T (fp32 [C][C] = C * DK halves)
    static bool attr = false;
    if (!attr) {
        ck(cudaFuncSetAttribute(k_gdn_chunk_prep<DK>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_prep)), "gdn prep smem");
        attr = true;
    }
    for (int t_base = 0; t_base < T; t_base += kGdnSlab * C) {
        const int n_chunks = std::min(kGdnSlab, (T - t_base + C - 1) / C);
        k_gdn_chunk_prep<DK><<<dim3(n_chunks, H), 256, smem_prep, stream>>>(conv, g, beta, T, t_base, groups, H, ch, Qh, Wf, Kt, Pf, Ut, gcb);
        // 32 value columns x 8 warps, 3 blocks per SM: the best of the variants swept in sw72
        // (64 or 128 columns, 2 blocks per SM, slabs of 4 or 16 chunks)
        gdn_state_launch<32, 8, 3>(S, Qh, Wf, Kt, Pf, Ut, gcb, o, T, t_base, n_chunks, H, stream);
    }
    ck(cudaGetLastError(), "gdn chunked");
}

void hist_rewind(cudaStream_t stream, float* hist, const float* old, const float* rows, int H, int C, int n) {
    k_hist_rewind<<<dim3((C + 255) / 256, H), 256, 0, stream>>>(hist, old, rows, H, C, n);
    ck(cudaGetLastError(), "hist_rewind");
}

}  // namespace flashrt::qwen4exp
