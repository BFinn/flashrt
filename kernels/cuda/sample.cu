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

__global__ void __launch_bounds__(kThreads) k_sample(const float* logits, int V, Params p, int K, uint64_t seed, int64_t pos0, int32_t* out) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* x = logits + size_t(row) * V;
    __shared__ float tmax[kThreads];
    __shared__ float cv[kCand];
    __shared__ int ci[kCand];
    __shared__ int n_cand;
    __shared__ float top_v[kMaxTopK];
    __shared__ int top_i[kMaxTopK];
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
    if (tid == 0) out[row] = choose(top_v, top_i, K, p, draw(seed, pos0 + row));
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

}  // namespace flashrt::sample
