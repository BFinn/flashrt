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
// quantize_q8_1 with the producer fused in front (decode's small epilogues), same layout:
// silu(g) * u, [n_tok][ncols] each;
void swiglu_q8_1(const float* g, const float* u, int64_t ncols, int n_tok, void* xq, cudaStream_t stream);
// per (token, head of dim): o * rsqrt(mean(o^2) + eps) * w * sigmoid(z) (dim 128; heads * dim a
// multiple of the row padding, 512).
bool gated_rms_norm_q8_1_ok(int dim, int heads);
void gated_rms_norm_q8_1(const float* o, const float* w, const float* z, int dim, int heads, float eps, int n_tok, void* xq,
                         cudaStream_t stream);
void matvec_q(uint32_t ggml_type, const void* W, const void* xq, float* y, int64_t ncols, int64_t nrows, int n_tok,
              cudaStream_t stream);

// Grouped Q2_0 mat-vec over experts held in equally strided slots (the VRAM expert cache):
// for channel i < n_ids, dst[i][rows] = W_{ids[i]} x_i, where W_e starts at w + e * slot_stride
// bytes (a whole number of Q2_0 blocks). With gate != nullptr the result is
// silu(G_{ids[i]} x_i) * (W_{ids[i]} x_i), G_e at gate + e * slot_stride (fused SwiGLU).
// xq holds Q8_1 activations (quantize_q8_1): one row shared by every channel, or, with
// per_channel_act, row i for channel i. dst is [n_ids][nrows].
void moe_q2_0(const void* w, const void* gate, const void* xq, const int32_t* ids, float* dst, int n_ids, int64_t ncols,
              int64_t nrows, int64_t slot_stride_bytes, bool per_channel_act, cudaStream_t stream);

// Grouped mat-vec over experts stored back to back (a GGUF expert tensor [E][nrows][ncols], or
// equally strided slots), for n_tok <= 8 tokens with k experts each (ids [n_tok][k], device):
// dst[t][j][rows] = W_{ids[t][j]} x, W_e at w + e * expert_stride_bytes. x is token t's row of
// xq (Q8_1, quantize_q8_1 with n_tok rows) or, with act_per_expert, row t * k + j. With gate,
// silu(G x) * (W x), G_e at gate + e * expert_stride_bytes (fused SwiGLU). Types: Q2_0, Q4_0,
// Q5_0, Q8_0, Q4_K, Q5_K, Q6_K.
void moe_q(uint32_t ggml_type, const void* w, const void* gate, const void* xq, const int32_t* ids, float* dst, int n_tok, int k,
           int64_t ncols, int64_t nrows, int64_t expert_stride_bytes, bool act_per_expert, cudaStream_t stream);

// Dequantize n contiguous elements (whole blocks) of a ggml-typed buffer to F32.
void dequantize(uint32_t ggml_type, const void* src, float* dst, int64_t n, cudaStream_t stream);

}  // namespace flashrt::gemv
