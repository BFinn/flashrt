// SPDX-License-Identifier: Apache-2.0
#include "src/ggml-cuda/common.cuh"
#include "src/ggml-cuda/mmq.cuh"
#include "src/ggml-cuda/quantize.cuh"

#include "kernels/cuda/ggml_gemm.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace flashrt::gemm {

namespace detail {
#define FLASHRT_MMQ_DECL(name) void name(const mmq_args& args, float* fixup, cudaStream_t stream);
FLASHRT_MMQ_DECL(mmq_q2_0)
FLASHRT_MMQ_DECL(mmq_q4_0)
FLASHRT_MMQ_DECL(mmq_q5_0)
FLASHRT_MMQ_DECL(mmq_q8_0)
FLASHRT_MMQ_DECL(mmq_q3_k)
FLASHRT_MMQ_DECL(mmq_q4_k)
FLASHRT_MMQ_DECL(mmq_q5_k)
FLASHRT_MMQ_DECL(mmq_q6_k)
FLASHRT_MMQ_DECL(mmq_iq4_xs)
FLASHRT_MMQ_DECL(mmq_iq4_nl)
#undef FLASHRT_MMQ_DECL
}  // namespace detail

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// block size (values) and bytes of the MMQ types; 0 = not an MMQ type
struct TypeInfo {
    int64_t blck, bytes;
    void (*run)(const mmq_args&, float*, cudaStream_t);
};
TypeInfo info(uint32_t t) {
    using namespace detail;
    switch (ggml_type(t)) {
        case GGML_TYPE_Q2_0: return {QK2_0, sizeof(block_q2_0), mmq_q2_0};
        case GGML_TYPE_Q4_0: return {QK4_0, sizeof(block_q4_0), mmq_q4_0};
        case GGML_TYPE_Q5_0: return {QK5_0, sizeof(block_q5_0), mmq_q5_0};
        case GGML_TYPE_Q8_0: return {QK8_0, sizeof(block_q8_0), mmq_q8_0};
        case GGML_TYPE_Q3_K: return {QK_K, sizeof(block_q3_K), mmq_q3_k};
        case GGML_TYPE_Q4_K: return {QK_K, sizeof(block_q4_K), mmq_q4_k};
        case GGML_TYPE_Q5_K: return {QK_K, sizeof(block_q5_K), mmq_q5_k};
        case GGML_TYPE_Q6_K: return {QK_K, sizeof(block_q6_K), mmq_q6_k};
        case GGML_TYPE_IQ4_XS: return {QK_K, sizeof(block_iq4_xs), mmq_iq4_xs};
        case GGML_TYPE_IQ4_NL: return {QK4_NL, sizeof(block_iq4_nl), mmq_iq4_nl};
        default: return {0, 0, nullptr};
    }
}

size_t up256(size_t x) { return (x + 255) & ~size_t(255); }

// the stream-k fixup: one I x J (128 x 128) tile per SM
size_t fixup_bytes() { return size_t(ggml_cuda_info().devices[ggml_cuda_get_device()].nsm) * 128 * 128 * 4; }

// Expert grouping (moe_prepare): tokens per block of the histogram and placement passes
constexpr int kGroupTokens = 512, kGroupMaxExperts = 4096;
int64_t group_blocks(int64_t rows) { return (rows + kGroupTokens - 1) / kGroupTokens; }   // rows >= tokens

// workspace: [fixup | activations (Q8_1 MMQ or BF16) | ids_src1 | ids_dst | expert_bounds | group counts]
struct Ws {
    float* fixup;
    char* act;
    int32_t *ids_src1, *ids_dst, *bounds, *group;
};
Ws carve(void* ws, size_t ws_bytes, int64_t ncols, int64_t rows, bool bf16) {
    Ws w;
    char* p = static_cast<char*>(ws);
    w.fixup = reinterpret_cast<float*>(p);
    p += up256(fixup_bytes());
    w.act = p;
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const size_t act = std::max<size_t>(size_t(rows) * padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + 128 * sizeof(block_q8_1_mmq),
                                        bf16 ? size_t(rows) * ncols * 2 : 0);
    p += up256(act);
    w.ids_src1 = reinterpret_cast<int32_t*>(p);
    p += up256(size_t(rows) * 4);
    w.ids_dst = reinterpret_cast<int32_t*>(p);
    p += up256(size_t(rows) * 4);
    w.bounds = reinterpret_cast<int32_t*>(p);
    p += up256(4096 * 4);
    w.group = reinterpret_cast<int32_t*>(p);
    p += up256(size_t(group_blocks(rows)) * kGroupMaxExperts * 4);
    if (size_t(p - static_cast<char*>(ws)) > ws_bytes) throw std::runtime_error("gemm: workspace too small");
    return w;
}

cublasHandle_t cublas() {
    static std::once_flag once;
    static cublasHandle_t h = nullptr;
    std::call_once(once, [] {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
    });
    return h;
}

// Groups the (token, slot) pairs by expert, as ggml's mm_ids_helper does (same outputs: within
// an expert in token order), but in O(T K) work: ggml's has each expert's warp scan every slot.
// 1. per block of kGroupTokens tokens: a histogram of experts (integer atomics: exact)
__global__ void k_group_hist(const int32_t* ids, int T, int K, int E, int32_t* cnt) {
    __shared__ int32_t h[kGroupMaxExperts];
    for (int e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
    __syncthreads();
    const int s0 = blockIdx.x * kGroupTokens * K, s1 = min(T, (blockIdx.x + 1) * kGroupTokens) * K;
    for (int sl = s0 + threadIdx.x; sl < s1; sl += blockDim.x) atomicAdd(&h[ids[sl]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[size_t(blockIdx.x) * E + e] = h[e];
}
// 2. one block: bounds[e] = slots of lower experts; cnt[b][e] becomes block b's first row of e
__global__ void k_group_scan(int32_t* cnt, int nb, int E, int32_t* bounds) {
    __shared__ int32_t tot[kGroupMaxExperts];
    __shared__ int32_t warp_sum[32];
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        int run = 0;
        for (int b = 0; b < nb; ++b) {
            const int c = cnt[size_t(b) * E + e];
            cnt[size_t(b) * E + e] = run;
            run += c;
        }
        tot[e] = run;
    }
    __syncthreads();
    // exclusive scan of tot over e: each thread a contiguous run, then the runs' offsets
    const int per = (E + blockDim.x - 1) / blockDim.x, e0 = threadIdx.x * per, e1 = min(E, e0 + per);
    int mine = 0;
    for (int e = e0; e < e1; ++e) mine += tot[e];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = mine;
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(~0u, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) warp_sum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < int(blockDim.x >> 5) ? warp_sum[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(~0u, w, o);
            if (lane >= o) w += n;
        }
        warp_sum[lane] = w;
    }
    __syncthreads();
    int run = (warp > 0 ? warp_sum[warp - 1] : 0) + incl - mine;
    for (int e = e0; e < e1; ++e) {
        const int t = tot[e];
        tot[e] = run;
        bounds[e] = run;
        run += t;
    }
    if (threadIdx.x == blockDim.x - 1) bounds[E] = run;
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        for (int b = 0; b < nb; ++b) cnt[size_t(b) * E + e] += tot[e];
}
// 3. per block, one warp, slots in order: a slot's row is its expert's next row in the block
// (lanes with the same expert ranked by lane: token order)
__global__ void k_group_place(const int32_t* ids, int T, int K, int E, const int32_t* cnt, int32_t* ids_src1, int32_t* ids_dst,
                              int nchannels_y, int sis1, bool inverse) {
    __shared__ int32_t next[kGroupMaxExperts];
    for (int e = threadIdx.x; e < E; e += blockDim.x) next[e] = cnt[size_t(blockIdx.x) * E + e];
    __syncwarp();
    const int lane = threadIdx.x;
    const int s0 = blockIdx.x * kGroupTokens * K, s1 = min(T, (blockIdx.x + 1) * kGroupTokens) * K;
    for (int b = s0; b < s1; b += 32) {
        const int sl = b + lane;
        const int e = sl < s1 ? ids[sl] : -1 - lane;   // distinct negatives: never peers
        const unsigned peers = __match_any_sync(~0u, e);
        if (e >= 0) {
            const int row = next[e] + __popc(peers & ((1u << lane) - 1));
            const int it = sl / K, iex = sl % K;
            ids_dst[row] = sl;
            if (inverse) ids_src1[sl] = row;
            else ids_src1[row] = it * sis1 + iex % nchannels_y;
        }
        __syncwarp();
        if (e >= 0 && lane == __ffs(peers) - 1) next[e] += __popc(peers);
        __syncwarp();
    }
}

__global__ void k_to_bf16(const float* x, __nv_bfloat16* y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

}  // namespace

bool supported(uint32_t t) { return info(t).blck > 0 || t == GGML_TYPE_BF16 || t == GGML_TYPE_F32; }

size_t workspace_bytes(int64_t ncols, int64_t rows, bool bf16) {
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const size_t act = std::max<size_t>(size_t(rows) * padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + 128 * sizeof(block_q8_1_mmq),
                                        bf16 ? size_t(rows) * ncols * 2 : 0);
    return up256(fixup_bytes()) + up256(act) + 2 * up256(size_t(rows) * 4) + up256(4096 * 4) +
           up256(size_t(group_blocks(rows)) * kGroupMaxExperts * 4);
}

void gemm(uint32_t t, const void* W, const float* x, float* y, int64_t ncols, int64_t nrows, int64_t T, void* ws, size_t ws_bytes,
          cudaStream_t stream) {
    Ws w = carve(ws, ws_bytes, ncols, T, t == GGML_TYPE_BF16);
    if (t == GGML_TYPE_BF16 || t == GGML_TYPE_F32) {
        cublasHandle_t h = cublas();
        cublasSetStream(h, stream);
        const float one = 1.0f, zero = 0.0f;
        cublasStatus_t st;
        if (t == GGML_TYPE_BF16) {
            const size_t n = size_t(T) * ncols;
            k_to_bf16<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, reinterpret_cast<__nv_bfloat16*>(w.act), n);
            st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, int(nrows), int(T), int(ncols), &one, W, CUDA_R_16BF, int(ncols), w.act,
                              CUDA_R_16BF, int(ncols), &zero, y, CUDA_R_32F, int(nrows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        } else {
            st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, int(nrows), int(T), int(ncols), &one, static_cast<const float*>(W), int(ncols), x,
                             int(ncols), &zero, y, int(nrows));
        }
        if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gemm: cuBLAS failed (" + std::to_string(int(st)) + ")");
        return;
    }
    const TypeInfo ti = info(t);
    if (!ti.blck || ncols % ti.blck) throw std::runtime_error("gemm: unsupported type or shape (type " + std::to_string(t) + ")");
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    quantize_mmq_q8_1_cuda(x, nullptr, w.act, ggml_type(t), ncols, ncols, ncols * T, ncols * T, padded, T, 1, 1, stream);
    const int64_t s01 = ncols / ti.blck;
    const int64_t s12 = T * padded * int64_t(sizeof(block_q8_1)) / (QK8_1 * int64_t(sizeof(int)));
    const mmq_args args = {static_cast<const char*>(W), ggml_type(t), reinterpret_cast<const int*>(w.act), nullptr, nullptr, y, nullptr,
                           ncols, nrows, T, s01, T, nrows,
                           1, 1, s01 * nrows, s12, nrows * T,
                           1, 1, s01 * nrows, s12, nrows * T,
                           T, T};
    ti.run(args, w.fixup, stream);
    ck(cudaGetLastError(), "gemm");
}

void gemm_bf16(const void* W, const void* x_bf16, float* y, int64_t ncols, int64_t nrows, int64_t T, cudaStream_t stream) {
    cublasHandle_t h = cublas();
    cublasSetStream(h, stream);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, int(nrows), int(T), int(ncols), &one, W, CUDA_R_16BF, int(ncols),
                                           x_bf16, CUDA_R_16BF, int(ncols), &zero, y, CUDA_R_32F, int(nrows), CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gemm_bf16: cuBLAS failed (" + std::to_string(int(st)) + ")");
}

void gemm_bf16_out(const void* W, const float* x, void* y_bf16, int64_t ncols, int64_t nrows, int64_t T, void* ws, size_t ws_bytes,
                   cudaStream_t stream) {
    Ws w = carve(ws, ws_bytes, ncols, T, true);
    const size_t n = size_t(T) * ncols;
    k_to_bf16<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, reinterpret_cast<__nv_bfloat16*>(w.act), n);
    cublasHandle_t h = cublas();
    cublasSetStream(h, stream);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, int(nrows), int(T), int(ncols), &one, W, CUDA_R_16BF, int(ncols),
                                           w.act, CUDA_R_16BF, int(ncols), &zero, y_bf16, CUDA_R_16BF, int(nrows), CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gemm_bf16_out: cuBLAS failed (" + std::to_string(int(st)) + ")");
}

void* bf16_staging(void* ws, size_t ws_bytes, int64_t ncols, int64_t T) { return carve(ws, ws_bytes, ncols, T, true).act; }

MoePlan moe_prepare(uint32_t t, int E, const float* x, bool x_per_slot, const int32_t* ids, int64_t T, int K, int64_t ncols, void* ws,
                    size_t ws_bytes, cudaStream_t stream) {
    const TypeInfo ti = info(t);
    if (!ti.blck || ncols % ti.blck || E > 4095) throw std::runtime_error("gemm::moe_prepare: unsupported type or shape");
    MoePlan p;
    p.type = t;
    p.n_experts = E;
    p.K = K;
    p.T = T;
    p.ncols = ncols;
    p.rows = T * K;
    p.ne11 = x_per_slot ? K : 1;
    Ws w = carve(ws, ws_bytes, ncols, p.rows, false);
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const bool dedup = p.ne11 == 1 && K > 1;
    {
        const int nb = int((T + kGroupTokens - 1) / kGroupTokens);
        k_group_hist<<<nb, 256, 0, stream>>>(ids, int(T), K, E, w.group);
        k_group_scan<<<1, 512, 0, stream>>>(w.group, nb, E, w.bounds);
        k_group_place<<<nb, 32, 0, stream>>>(ids, int(T), K, E, w.group, w.ids_src1, w.ids_dst, int(p.ne11), int(p.ne11), dedup);
    }
    if (dedup)
        quantize_scatter_mmq_q8_1_cuda(x, w.ids_src1, w.act, ggml_type(t), ncols, ncols, padded, T, p.rows, K, stream);
    else
        quantize_mmq_q8_1_cuda(x, w.ids_src1, w.act, ggml_type(t), ncols, ncols, ncols * K, ncols * p.rows, padded, p.rows, 1, 1, stream);
    // the launch grid covers the largest expert's tokens, not all T (most tiles would be empty)
    static thread_local int32_t* hb = nullptr;
    if (!hb) ck(cudaHostAlloc(&hb, 4096 * 4, cudaHostAllocDefault), "cudaHostAlloc bounds");
    ck(cudaMemcpyAsync(hb, w.bounds, size_t(E + 1) * 4, cudaMemcpyDeviceToHost, stream), "bounds to host");
    ck(cudaStreamSynchronize(stream), "moe_prepare");
    int64_t mx = 1;
    for (int e = 0; e < E; ++e) mx = std::max<int64_t>(mx, hb[e + 1] - hb[e]);
    p.ncols_max = mx;
    static const int force_j = [] {
        const char* e = std::getenv("FLASHRT_MOE_J");
        return e ? std::atoi(e) : 0;
    }();
    p.ncols_opt = force_j > 0 ? std::min<int64_t>(mx, force_j) : mx;
    p.act = reinterpret_cast<const int*>(w.act);
    p.ids_dst = w.ids_dst;
    p.bounds = w.bounds;
    p.fixup = w.fixup;
    return p;
}

void moe_run(const MoePlan& p, const void* W, int64_t expert_stride_bytes, float* y, int64_t nrows, cudaStream_t stream) {
    const TypeInfo ti = info(p.type);
    if (expert_stride_bytes % ti.bytes) throw std::runtime_error("gemm::moe_run: bad expert stride");
    const int64_t padded = GGML_PAD(p.ncols, MATRIX_ROW_PADDING);
    const int64_t s01 = p.ncols / ti.blck, s02 = expert_stride_bytes / ti.bytes;
    const int64_t s12 = p.ne11 * padded * int64_t(sizeof(block_q8_1)) / (QK8_1 * int64_t(sizeof(int)));
    const int E = p.n_experts;
    const mmq_args args = {static_cast<const char*>(W), ggml_type(p.type), p.act, p.ids_dst, p.bounds, y, nullptr,
                           p.ncols, nrows, p.rows, s01, p.rows, nrows,
                           E, E, s02, s12, nrows * p.K,
                           1, 1, s02 * E, s12 * p.T, nrows * p.rows,
                           p.ncols_max, p.ncols_opt};
    ti.run(args, p.fixup, stream);
    ck(cudaGetLastError(), "gemm::moe_run");
}

void moe(uint32_t t, const void* W, int64_t expert_stride_bytes, int E, const float* x, bool x_per_slot, const int32_t* ids, int64_t T,
         int K, float* y, int64_t ncols, int64_t nrows, void* ws, size_t ws_bytes, cudaStream_t stream) {
    const MoePlan p = moe_prepare(t, E, x, x_per_slot, ids, T, K, ncols, ws, ws_bytes, stream);
    moe_run(p, W, expert_stride_bytes, y, nrows, stream);
}

}  // namespace flashrt::gemm
