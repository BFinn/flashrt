// SPDX-License-Identifier: Apache-2.0
// Routed experts for prefill chunks on int8 tensor cores, straight from the expert arena's planar
// Q2_0 layout (quant/q2_0/q2_0.hpp): no conversion to ggml's layout, gate and up in one kernel
// with SwiGLU and the down input's quantization in its epilogue, and the tile list built on the
// GPU (no host sync).
//
// The planar layout was chosen so that one 32-bit load per lane gives the mma.m16n8k32 A
// fragments of a 64-weight block for both k-steps (shifts of 0, 2, 4, 6 and a mask). The codes
// enter the MMA unsigned (0..3); the -1 of Q2_0's (q - 1) * d is folded into the activations'
// per-block sums, which also turn the int32 results into floats without I2F.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashrt::moe_q2 {

// Device workspace for T tokens, K experts each, of E experts with d_model n and d_ff ff.
size_t workspace_bytes(int T, int K, int n, int ff, int E);

// Expert e's blob is at experts + e * stride: gate [ff][n], up [ff][n], down [n][ff], each in
// the planar layout (q2_0::expert_view). ids [T][K] (device). yd [T * K][n] gets, for slot
// t * K + k, down(silu(gate x_t) * up x_t) of expert ids[t * K + k]. n and ff: multiples of 128.
// block64: the activations are quantized in blocks of 64 (the weights' block) instead of 32,
// which halves the kernels' scale arithmetic.
void run(const uint8_t* experts, size_t stride, int E, int n, int ff, const float* x, const int32_t* ids, int T, int K, float* yd,
         void* ws, size_t ws_bytes, cudaStream_t stream, bool block64 = false);

}  // namespace flashrt::moe_q2
