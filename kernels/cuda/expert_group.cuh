// SPDX-License-Identifier: Apache-2.0
// The first pass of grouping (token, slot) pairs by expert, shared by ggml_gemm.cu (ggml's MMQ
// layout) and moe_q2.cu (flashrt's tiles): per block of kTokens tokens, a histogram of the
// experts its slots route to. Integer atomics, so the counts are exact and order-free. The scan
// and placement passes differ between the two users and stay with them.
#pragma once

#include <cstdint>

namespace flashrt::expert_group {

constexpr int kTokens = 512;        // tokens per block of the histogram and placement passes
constexpr int kMaxExperts = 4096;   // experts a layer may have (shared-memory histogram)

namespace {   // each translation unit gets its own kernel
__global__ void k_hist(const int32_t* ids, int T, int K, int E, int32_t* cnt) {
    __shared__ int32_t h[kMaxExperts];
    for (int e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
    __syncthreads();
    const int s0 = blockIdx.x * kTokens * K, s1 = min(T, (blockIdx.x + 1) * kTokens) * K;
    for (int sl = s0 + threadIdx.x; sl < s1; sl += blockDim.x) atomicAdd(&h[ids[sl]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[size_t(blockIdx.x) * E + e] = h[e];
}
}  // namespace

}  // namespace flashrt::expert_group
