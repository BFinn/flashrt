// SPDX-License-Identifier: Apache-2.0
// Q3R: a lossless GPU layout for ggml Q3_K matrices, and its mat-vec (int8 activations, dp4a).
//
// ggml's Q3_K MMVQ reads at about 320-390 GB/s on sm_120 (bench/results/2026-09-27-sw10-q3r).
// Q3R holds the same weights (weight = d * sc * (q - 4), q in 0..7) in planes that one shift,
// one mask and one OR per 4 elements turn into dp4a operands:
//   lo [rows][K/64][16]   2 low bits of q: element j of a 64-block in byte j%16, bits 2*(j/16)
//   hi [rows][K/128][16]  high bit of q for two 64-blocks: element j of the even block in byte
//                         j%16 bit j/16, of the odd block bit 4 + j/16
//   sc [rows][K/16]       int8: the 6-bit group scale minus 32
//   d  [rows][K/256]      fp16 super-block scale
// 3.56 bits per weight (Q3_K: 3.44). Activations are quantized to int8 per 64 values.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashrt::q3r {

// Bytes of a rows x K matrix in Q3R (K % 256 == 0).
size_t bytes(int64_t rows, int64_t K);

// Converts ggml Q3_K rows (rows * K/256 blocks of 110 bytes) to Q3R, on the GPU. src and dst
// must not overlap.
void repack(const void* q3k, void* q3r, int64_t rows, int64_t K, cudaStream_t stream);

// The inverse of repack: Q3R back to ggml Q3_K rows (rows * K/256 blocks of 110 bytes), on the
// GPU (prefill runs ggml's Q3_K matrix-matrix kernel on the result).
void unpack(const void* q3r, void* q3k, int64_t rows, int64_t K, cudaStream_t stream);

// ggml Q3_K rows -> ggml Q8_0 rows, exactly: a Q3_K weight is d * sc * (q - 4) with sc in -32..31
// and q - 4 in -4..3, so -(q - 4) * sc lies in -128..124 and Q8_0 holds it with scale -d (the
// product itself reaches +128). Prefill multiplies Q8_0 about 1.37x faster than Q3_K (ggml MMQ,
// tools/bench_mmq). q8 must hold rows * K / 32 * 34 bytes.
void q3k_to_q8_0(const void* q3k, void* q8, int64_t rows, int64_t K, cudaStream_t stream);

// y[T][rows] = W x[T][K]. K <= 8192.
void matvec(const void* q3r, const float* x, float* y, int64_t rows, int64_t K, int T, cudaStream_t stream);

}  // namespace flashrt::q3r
