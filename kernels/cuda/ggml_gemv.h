// SPDX-License-Identifier: Apache-2.0
// Dense mat-vec on the GPU for ggml-typed weights: y = W x for 1..8 tokens.
//
// Quantized weights (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K, ...) go
// through ggml's MMVQ kernels against Q8_1-quantized activations; F32, F16 and BF16 weights go
// through ggml's MMVF kernels against F32 activations. Both are vendored unmodified from
// llama.cpp (MIT, third_party/ggml). This header exposes no ggml types.
//
// Layout: W is row-major [nrows][ncols] in ggml's block format (contiguous rows); x is
// [n_tok][ncols] F32; y is [n_tok][nrows] F32. Weight buffers need kWeightTailPad zeroed
// bytes after the last row, because the kernels may read past it.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashrt::gemv {

constexpr int kMaxTokens = 8;
constexpr size_t kWeightTailPad = 64 * 1024;

bool supported(uint32_t ggml_type);
bool is_float(uint32_t ggml_type);        // F32 / F16 / BF16: MMVF path
int64_t row_bytes(uint32_t ggml_type, int64_t ncols);

// Scratch for the Q8_1 activations of one call.
size_t q8_1_bytes(int64_t ncols, int n_tok);

// y = W x. For quantized W, x is first quantized into `scratch` (q8_1_bytes); for float W,
// scratch is unused. Everything is enqueued on `stream`.
void matvec(uint32_t ggml_type, const void* W, const float* x, float* y, int64_t ncols, int64_t nrows, int n_tok,
            void* scratch, cudaStream_t stream);

// The two halves separately, to quantize one activation once for several matrices.
void quantize_q8_1(const float* x, int64_t ncols, int n_tok, void* xq, cudaStream_t stream);
void matvec_q(uint32_t ggml_type, const void* W, const void* xq, float* y, int64_t ncols, int64_t nrows, int n_tok,
              cudaStream_t stream);

}  // namespace flashrt::gemv
