// SPDX-License-Identifier: Apache-2.0
#include "kernels/cuda/sample.h"

#include <stdexcept>
#include <string>

namespace flashrt::sample {

namespace {

// (v, i) ranks above (w, j): larger value, or equal and lower index
__device__ __forceinline__ bool above(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

// One block of 1024 threads per row: the top-k candidates, then choose().
//  1. each thread's largest element (its strided share of the row);
//  2. the k-th largest of those 1024 maxima bounds the k-th largest element from below (k
//     threads hold an element at least that large), so only elements >= it can be candidates;
//  3. they are compacted into shared memory (at most kCand; a denser row falls back to scanning
//     the whole row per round) and the top k taken in k rounds of a block-wide max.
constexpr int kThreads = 1024, kCand = 4096;

__device__ void block_best(float& v, int& i) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffff, v, o);
        const int i2 = __shfl_xor_sync(0xffffffff, i, o);
        if (above(v2, i2, v, i)) { v = v2; i = i2; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x];
        i = bi[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffff, v, o);
            const int i2 = __shfl_xor_sync(0xffffffff, i, o);
            if (above(v2, i2, v, i)) { v = v2; i = i2; }
        }
        if (threadIdx.x == 0) { bv[0] = v; bi[0] = i; }
    }
    __syncthreads();
    v = bv[0];
    i = bi[0];
    __syncthreads();
}

// The top K (value, index) of row x [V] into top_v / top_i (shared, valid for every thread after
// the call), sorted descending, ties to the lower index.
__device__ void row_topk(const float* x, int V, int K, float* top_v, int* top_i) {
    const int tid = threadIdx.x;
    __shared__ float tmax[kThreads];
    __shared__ float cv[kCand];
    __shared__ int ci[kCand];
    __shared__ int n_cand;
    __shared__ float bound;
    // 1. per-thread maxima
    float m = -INFINITY;
    for (int i = tid; i < V; i += kThreads) m = fmaxf(m, x[i]);
    tmax[tid] = m;
    if (tid == 0) n_cand = 0;
    __syncthreads();
    // 2. the K-th largest thread maximum
    int rank = 0;
    for (int j = 0; j < kThreads; ++j) rank += above(tmax[j], j, m, tid);
    if (rank == K - 1) bound = m;
    __syncthreads();
    const float b = bound;
    // 3. compact the candidates
    for (int i = tid; i < V; i += kThreads) {
        const float v = x[i];
        if (v >= b) {
            const int slot = atomicAdd(&n_cand, 1);
            if (slot < kCand) { cv[slot] = v; ci[slot] = i; }
        }
    }
    __syncthreads();
    const int nc = n_cand;
    const bool dense = nc > kCand;
    float pv = INFINITY;
    int pi = -1;
    for (int r = 0; r < K; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        if (!dense) {
            for (int j = tid; j < nc; j += kThreads)
                if (above(pv, pi, cv[j], ci[j]) && above(cv[j], ci[j], bv, bi)) { bv = cv[j]; bi = ci[j]; }
        } else {
            for (int i = tid; i < V; i += kThreads) {
                const float v = x[i];
                if (above(pv, pi, v, i) && above(v, i, bv, bi)) { bv = v; bi = i; }
            }
        }
        block_best(bv, bi);
        if (tid == 0) { top_v[r] = bv; top_i[r] = bi; }
        pv = bv;
        pi = bi;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kThreads) k_sample(const float* logits, int V, Params p, int K, uint64_t seed, int64_t pos0, int32_t* out) {
    const int row = blockIdx.x;
    __shared__ float top_v[kMaxTopK];
    __shared__ int top_i[kMaxTopK];
    row_topk(logits + size_t(row) * V, V, K, top_v, top_i);
    if (threadIdx.x == 0) out[row] = choose(top_v, top_i, K, p, draw(seed, pos0 + row));
}

__global__ void __launch_bounds__(kThreads) k_draft_row(const float* x, int V, const DraftCfg* cfg, const int32_t* dp, const int32_t* ids,
                                                        int32_t* v, int32_t* q_ids, float* q_p, int32_t* q_n) {
    const Params p = cfg->p;
    const int K = p.temperature > 0.0f ? min(max(p.top_k, 1), kMaxTopK) : 1;
    __shared__ float top_v[kMaxTopK];
    __shared__ int top_i[kMaxTopK];
    row_topk(x, V, K, top_v, top_i);
    if (threadIdx.x == 0) {
        float prob[kMaxTopK];
        const int n = chain_probs(top_v, K, p, prob);
        const int64_t pos = int64_t(dp[1]) + 1;
        const int step = dp[3];
        const float u = draw(cfg->seed ^ kDraftSalt, pos);
        float cum = 0.0f;
        int c = n - 1;
        for (int i = 0; i < n; ++i) {
            cum += prob[i];
            if (u < cum) {
                c = i;
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            q_ids[step * kMaxTopK + i] = ids ? ids[top_i[i]] : top_i[i];
            q_p[step * kMaxTopK + i] = prob[i];
        }
        q_n[step] = n;
        v[1] = ids ? ids[top_i[c]] : top_i[c];
    }
}

__global__ void __launch_bounds__(kThreads) k_spec_verify(const float* logits, int V, Params p, int K, uint64_t seed, int64_t pos0, int rows,
                                                          const int32_t* drafts, const int32_t* q_ids, const float* q_p, const int32_t* q_n,
                                                          int32_t* out) {
    const int row = blockIdx.x;
    __shared__ float top_v[kMaxTopK];
    __shared__ int top_i[kMaxTopK];
    row_topk(logits + size_t(row) * V, V, K, top_v, top_i);
    if (threadIdx.x != 0) return;
    const int64_t pos = pos0 + row;
    if (row == rows - 1) {   // the last row: a plain sample
        out[row] = choose(top_v, top_i, K, p, draw(seed, pos));
        return;
    }
    float pp[kMaxTopK];
    const int n = chain_probs(top_v, K, p, pp);
    const int32_t d = drafts[row];
    const int qn = q_n[row];
    auto q_of = [&](int32_t t) {
        for (int i = 0; i < qn; ++i)
            if (q_ids[row * kMaxTopK + i] == t) return q_p[row * kMaxTopK + i];
        return 0.0f;
    };
    float pd = 0.0f;
    for (int i = 0; i < n; ++i)
        if (top_i[i] == d) pd = pp[i];
    const float qd = q_of(d);
    if (draw(seed ^ kAccSalt, pos) * qd < pd) {   // accept with min(1, p / q)
        out[row] = d;
        out[rows + row] = 1;
        return;
    }
    float r[kMaxTopK], rs = 0.0f;   // the residual max(0, p - q) over p's support
    for (int i = 0; i < n; ++i) rs += r[i] = fmaxf(0.0f, pp[i] - q_of(top_i[i]));
    int c = 0;
    if (rs > 0.0f) {
        const float target = draw(seed, pos) * rs;
        float cum = 0.0f;
        c = n - 1;
        for (int i = 0; i < n; ++i) {
            cum += r[i];
            if (target < cum) {
                c = i;
                break;
            }
        }
    }
    out[row] = top_i[c];
    out[rows + row] = 0;
}

}  // namespace

void sample_rows(const float* logits, int rows, int n_vocab, const Params& p, uint64_t seed, int64_t pos0, int32_t* out_dev,
                 cudaStream_t stream) {
    if (p.temperature > 0.0f && (p.top_k < 1 || p.top_k > kMaxTopK))
        throw std::runtime_error("sample_rows: top_k must be 1.." + std::to_string(kMaxTopK));
    if (rows < 1 || n_vocab < kThreads) throw std::runtime_error("sample_rows: bad shape");
    const int K = p.temperature > 0.0f ? p.top_k : 1;
    k_sample<<<rows, kThreads, 0, stream>>>(logits, n_vocab, p, K, seed, pos0, out_dev);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("sample_rows: ") + cudaGetErrorString(e));
}

void draft_row(const float* logits, int V, const DraftCfg* cfg_dev, const int32_t* dp, const int32_t* ids, int32_t* v, int32_t* q_ids,
               float* q_p, int32_t* q_n, cudaStream_t stream) {
    if (V < kThreads) throw std::runtime_error("draft_row: bad shape");
    k_draft_row<<<1, kThreads, 0, stream>>>(logits, V, cfg_dev, dp, ids, v, q_ids, q_p, q_n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("draft_row: ") + cudaGetErrorString(e));
}

void spec_verify(const float* logits, int rows, int n_vocab, const Params& p, uint64_t seed, int64_t pos0, const int32_t* drafts,
                 const int32_t* q_ids, const float* q_p, const int32_t* q_n, int32_t* out_dev, cudaStream_t stream) {
    if (p.temperature <= 0.0f || p.top_k < 1 || p.top_k > kMaxTopK) throw std::runtime_error("spec_verify: needs temperature > 0, top_k 1..64");
    if (rows < 1 || n_vocab < kThreads) throw std::runtime_error("spec_verify: bad shape");
    k_spec_verify<<<rows, kThreads, 0, stream>>>(logits, n_vocab, p, p.top_k, seed, pos0, rows, drafts, q_ids, q_p, q_n, out_dev);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("spec_verify: ") + cudaGetErrorString(e));
}

}  // namespace flashrt::sample
