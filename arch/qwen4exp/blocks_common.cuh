// SPDX-License-Identifier: Apache-2.0
// Semantics follow llama.cpp's qwen4exp graph (src/models/qwen4exp.cpp, MIT); the kernels
// are written for flashrt.
// What the qwen4exp block files (blocks.cu, hc.cu, gdn.cu, qsa.cu, moe_ref.cu, ple.cu) share.
// Internal: the public interface is blocks.hpp.
#pragma once

#include "arch/qwen4exp/blocks.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// block-wide sum; every thread gets it
inline __device__ float block_sum(float v) {
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? red[threadIdx.x] : 0.0f;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

}  // namespace

// Rewinds a history of H rows (oldest first, C values each) after a call of inputs to its first
// n: row j = row j + n of [old history ; the call's inputs] (gdn.cu; the PLE conv history too).
void hist_rewind(cudaStream_t stream, float* hist, const float* old, const float* rows, int H, int C, int n);

}  // namespace flashrt::qwen4exp
