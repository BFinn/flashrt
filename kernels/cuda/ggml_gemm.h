// SPDX-License-Identifier: Apache-2.0
// Matrix-matrix products for prefill (many tokens at once), where the mat-vec kernels would
// re-read every weight once per 8 tokens.
//
// Quantized weights go through ggml's MMQ kernels (int8 tensor cores against Q8_1 activations,
// vendored unmodified from llama.cpp, MIT); flashrt launches them itself with its own stream-k
// fixup buffer (ggml's launcher wants a ggml backend context). BF16 and F32 weights go through
// cuBLAS (BF16: the activations are rounded to BF16, as llama.cpp's prefill does).
//
// Layouts as in ggml_gemv.h: W row-major [nrows][ncols] in ggml's block format, x [T][ncols]
// F32, y [T][nrows] F32. Weight buffers need gemv::kWeightTailPad zeroed bytes after the end.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashrt::gemm {

// Types MMQ takes: Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL; and BF16, F32.
bool supported(uint32_t ggml_type);

// Device workspace one call needs (activations in MMQ's Q8_1 layout or BF16, routing lists,
// the stream-k fixup). For moe(), T is tokens * experts per token.
size_t workspace_bytes(int64_t ncols, int64_t T);

// y = W x for T tokens.
void gemm(uint32_t ggml_type, const void* W, const float* x, float* y, int64_t ncols, int64_t nrows, int64_t T, void* ws,
          size_t ws_bytes, cudaStream_t stream);

// Grouped expert products in two steps, so several weight tensors share one routing: prepare()
// groups the tokens by expert and quantizes the activations (it waits for the stream once, to
// size the launch grid by the largest expert's token count), run() multiplies one tensor. The
// plan lives in ws, which must stay untouched between them.
struct MoePlan {
    uint32_t type = 0;
    int n_experts = 0, K = 0;
    int64_t T = 0, ncols = 0, rows = 0, ne11 = 0, ncols_max = 0;
    const int* act = nullptr;
    const int32_t *ids_dst = nullptr, *bounds = nullptr;
    float* fixup = nullptr;
};
MoePlan moe_prepare(uint32_t ggml_type, int n_experts, const float* x, bool x_per_slot, const int32_t* ids, int64_t T, int K,
                    int64_t ncols, void* ws, size_t ws_bytes, cudaStream_t stream);
void moe_run(const MoePlan& plan, const void* W, int64_t expert_stride_bytes, float* y, int64_t nrows, cudaStream_t stream);

// Grouped expert product (MoE prefill): experts back to back, W_e at W + e * expert_stride_bytes,
// each [nrows][ncols]. ids [T][K] (device) are each token's experts. y [T][K][nrows] gets
// W_{ids[t][k]} x_{t,k}, where x_{t,k} is row t of x ([T][ncols]) or, with x_per_slot, row t * K + k
// of x ([T][K][ncols]).
void moe(uint32_t ggml_type, const void* W, int64_t expert_stride_bytes, int n_experts, const float* x, bool x_per_slot,
         const int32_t* ids, int64_t T, int K, float* y, int64_t ncols, int64_t nrows, void* ws, size_t ws_bytes,
         cudaStream_t stream);

}  // namespace flashrt::gemm
