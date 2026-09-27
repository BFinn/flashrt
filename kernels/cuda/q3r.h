// SPDX-License-Identifier: Apache-2.0
// Q3R: a lossless decode layout for ggml Q3_K matrices, and its single-token mat-vec.
//
// ggml's Q3_K (256-element super-blocks: 2-bit plane, high-bit mask, 6-bit packed scales,
// fp16 d) reads at about 320 GB/s through MMVQ on sm_120. Q3R holds the same values, split
// per matrix into planes that a warp reads with coalesced loads and decodes with shifts:
//   low  [rows][K/4]   2 low bits of element e at bits 2*(e%16) of the group's uint32
//   high [rows][K/8]   the high bit of element e at bit e%16 of the group's uint16
//   sc   [rows][K/16]  int8 scale of each 16-element group (the 6-bit scale minus 32)
//   d    [rows][K/256] fp16 super-block scale
// Weight e = d * sc * ((low | high << 2) - 4), exactly ggml's dequantized value.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashrt::q3r {

// Bytes of the Q3R copy of a rows x K matrix (K % 256 == 0), planes 256-byte aligned.
size_t bytes(int64_t rows, int64_t K);

// Converts ggml Q3_K rows (rows * K/256 blocks of 110 bytes) to Q3R, on the GPU.
void repack(const void* q3k, void* q3r, int64_t rows, int64_t K, cudaStream_t stream);

// y[rows] = W x for one token, float activations (no Q8_1 rounding). K <= 11264.
void matvec(const void* q3r, const float* x, float* y, int64_t rows, int64_t K, cudaStream_t stream);

// Kernel variants for bench_gemv --q3r (1 = matvec's current kernel).
void matvec_variant(int v, const void* q3r, const float* x, float* y, int64_t rows, int64_t K, cudaStream_t stream);

}  // namespace flashrt::q3r
