// SPDX-License-Identifier: Apache-2.0
// flashrt wrapper over ggml's vendored CUDA mat-vec kernels (MIT, third_party/ggml).
//
// The vendored .cu files are included here, unmodified, so their static dispatchers
// (mul_mat_vec_q_switch_type, mul_mat_vec_f_cuda) are callable from this translation unit.
// Arguments are set the way ggml_cuda_mul_mat_vec_q / ggml_cuda_mul_mat_vec_f set them for a
// contiguous 2-D weight and n_tok activation columns.
#include "src/ggml-cuda/mmvq.cu"
#include "src/ggml-cuda/mmvf.cu"
#include "src/ggml-cuda/quantize.cu"

#include "kernels/cuda/ggml_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace flashrt::gemv {

namespace {

int64_t block_size(uint32_t t) {
    switch (ggml_type(t)) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q2_0: return QK2_0;
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q4_1: return QK4_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_IQ4_NL: return QK4_NL;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS: return QK_K;
        default: return 0;
    }
}

int64_t type_size(uint32_t t) {
    switch (ggml_type(t)) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q2_0: return sizeof(block_q2_0);
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default: return 0;
    }
}

void check_args(uint32_t t, int64_t ncols, int n_tok) {
    if (!supported(t)) throw std::runtime_error("gemv: unsupported ggml type " + std::to_string(t));
    if (ncols % block_size(t)) throw std::runtime_error("gemv: ncols is not a multiple of the block size");
    if (n_tok < 1 || n_tok > kMaxTokens) throw std::runtime_error("gemv: n_tok must be 1..8");
}

}  // namespace

bool supported(uint32_t t) { return block_size(t) > 0; }
bool is_float(uint32_t t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; }
int64_t row_bytes(uint32_t t, int64_t ncols) { return block_size(t) ? ncols / block_size(t) * type_size(t) : 0; }

size_t q8_1_bytes(int64_t ncols, int n_tok) {
    return size_t(n_tok) * size_t(GGML_PAD(ncols, MATRIX_ROW_PADDING)) / QK8_1 * sizeof(block_q8_1);
}

void quantize_q8_1(const float* x, int64_t ncols, int n_tok, void* xq, cudaStream_t stream) {
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    // ne00 = ncols, row stride ncols; ne0 = padded (the tail is written as zeros); ne1 = tokens
    quantize_row_q8_1_cuda(x, nullptr, xq, GGML_TYPE_Q8_0, ncols, ncols, ncols * n_tok, ncols * n_tok, padded, n_tok, 1, 1,
                           stream);
}

void matvec_q(uint32_t t, const void* W, const void* xq, float* y, int64_t ncols, int64_t nrows, int n_tok,
              cudaStream_t stream) {
    check_args(t, ncols, n_tok);
    const int64_t s01 = ncols / block_size(t);                            // row stride in blocks
    const int64_t s11 = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;      // activation stride in Q8_1 blocks
    const ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q_switch_type(W, ggml_type(t), xq, nullptr, fusion, y, int(ncols), int(nrows), n_tok, int(s01), int(s11),
                              int(nrows), 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, stream);
}

void matvec(uint32_t t, const void* W, const float* x, float* y, int64_t ncols, int64_t nrows, int n_tok, void* scratch,
            cudaStream_t stream) {
    check_args(t, ncols, n_tok);
    const ggml_cuda_mm_fusion_args_device fusion{};
    switch (ggml_type(t)) {
        case GGML_TYPE_F32:
            mul_mat_vec_f_cuda(static_cast<const float*>(W), x, nullptr, fusion, y, ncols, nrows, n_tok, ncols, ncols,
                               int(nrows), 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, GGML_PREC_DEFAULT, stream);
            return;
        case GGML_TYPE_F16:
            mul_mat_vec_f_cuda(static_cast<const half*>(W), x, nullptr, fusion, y, ncols, nrows, n_tok, ncols, ncols,
                               int(nrows), 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, GGML_PREC_F32, stream);
            return;
        case GGML_TYPE_BF16:
            mul_mat_vec_f_cuda(static_cast<const nv_bfloat16*>(W), x, nullptr, fusion, y, ncols, nrows, n_tok, ncols,
                               ncols, int(nrows), 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, GGML_PREC_DEFAULT, stream);
            return;
        default:
            quantize_q8_1(x, ncols, n_tok, scratch, stream);
            matvec_q(t, W, scratch, y, ncols, nrows, n_tok, stream);
    }
}

}  // namespace flashrt::gemv
