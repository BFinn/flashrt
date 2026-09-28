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
#include "src/ggml-cuda/convert.cu"

#include "kernels/cuda/ggml_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>
#include <type_traits>

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

namespace {
// one q8_1 block per warp (as ggml's quantize_q8_1): the 32 lanes' values -> d = amax / 127, sum
__device__ __forceinline__ void store_q8_1(block_q8_1* y, int64_t i_cont, float xi) {
    float amax = fabsf(xi), sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int64_t ib = i_cont / QK8_1, iqs = i_cont % QK8_1;
    y[ib].qs[iqs] = amax == 0.0f ? 0 : int8_t(roundf(xi / d));
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}
__global__ void k_swiglu_q8_1(const float* g, const float* u, block_q8_1* y, int64_t ncols, int64_t padded) {
    const int64_t i0 = int64_t(blockDim.x) * blockIdx.x + threadIdx.x, t = blockIdx.y;
    if (i0 >= padded) return;   // padded is a multiple of the block
    float xi = 0.0f;
    if (i0 < ncols) {
        const float gv = g[t * ncols + i0];
        xi = gv / (1.0f + __expf(-gv)) * u[t * ncols + i0];
    }
    store_q8_1(y, t * padded + i0, xi);
}
// block = (token, head), dim threads
__global__ void k_gated_rms_norm_q8_1(const float* o, const float* w, const float* z, block_q8_1* y, int dim, int heads, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x / heads, h = blockIdx.x % heads, i = threadIdx.x;
    const size_t at = (size_t(t) * heads + h) * dim + i;
    const float v = o[at];
    float ss = warp_reduce_sum(v * v);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    for (int k = 0; k < (dim >> 5); ++k) ss += red[k];
    const float inv = rsqrtf(ss / dim + eps);
    store_q8_1(y, int64_t(at), v * inv * w[i] / (1.0f + __expf(-z[at])));
}
}  // namespace

void swiglu_q8_1(const float* g, const float* u, int64_t ncols, int n_tok, void* xq, cudaStream_t stream) {
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    k_swiglu_q8_1<<<dim3(unsigned(padded / 256), unsigned(n_tok)), 256, 0, stream>>>(g, u, static_cast<block_q8_1*>(xq), ncols, padded);
}

bool gated_rms_norm_q8_1_ok(int dim, int heads) { return dim == 128 && (int64_t(dim) * heads) % MATRIX_ROW_PADDING == 0; }

void gated_rms_norm_q8_1(const float* o, const float* w, const float* z, int dim, int heads, float eps, int n_tok, void* xq,
                         cudaStream_t stream) {
    if (!gated_rms_norm_q8_1_ok(dim, heads)) throw std::runtime_error("gemv::gated_rms_norm_q8_1: unsupported shape");
    k_gated_rms_norm_q8_1<<<unsigned(n_tok * heads), unsigned(dim), 0, stream>>>(o, w, z, static_cast<block_q8_1*>(xq), dim, heads, eps);
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

void moe_q2_0(const void* w, const void* gate, const void* xq, const int32_t* ids, float* dst, int n_ids, int64_t ncols,
              int64_t nrows, int64_t slot_stride_bytes, bool per_channel_act, cudaStream_t stream) {
    if (ncols % QK2_0 || slot_stride_bytes % int64_t(sizeof(block_q2_0)) || n_ids < 1)
        throw std::runtime_error("gemv::moe_q2_0: bad shape");
    const int64_t act_blocks = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;   // Q8_1 blocks per activation row
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    const int warp = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    mul_mat_vec_q_moe_launch<GGML_TYPE_Q2_0>(
        w, xq, ids, fusion, dst, uint32_t(ncols), init_fastdiv_values(per_channel_act ? uint32_t(n_ids) : 1u), uint32_t(nrows),
        uint32_t(ncols / QK2_0), uint32_t(act_blocks), uint32_t(nrows),
        uint32_t(slot_stride_bytes / int64_t(sizeof(block_q2_0))), per_channel_act ? uint32_t(act_blocks) : 0u, uint32_t(nrows),
        1u, 0u, warp, n_ids, stream);
}

void moe_q(uint32_t t, const void* w, const void* gate, const void* xq, const int32_t* ids, float* dst, int n_tok, int k,
           int64_t ncols, int64_t nrows, int64_t expert_stride_bytes, bool act_per_expert, cudaStream_t stream) {
    check_args(t, ncols, n_tok);
    if (is_float(t) || expert_stride_bytes % type_size(t) || k < 1) throw std::runtime_error("gemv::moe_q: bad type or shape");
    const int64_t act_blocks = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;   // Q8_1 blocks per activation row
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    const int warp = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // channel = expert slot j of the token (grid y), column = token (block y)
    const uint3 nch_y = init_fastdiv_values(act_per_expert ? uint32_t(k) : 1u);
    const uint32_t s_col_y = uint32_t(act_per_expert ? act_blocks * k : act_blocks);
    const uint32_t s_ch_y = act_per_expert ? uint32_t(act_blocks) : 0u;
    const uint32_t s_ch_x = uint32_t(expert_stride_bytes / type_size(t));
    const uint32_t s_row_x = uint32_t(ncols / block_size(t));
    auto run = [&](auto tag) {
        constexpr ggml_type T = decltype(tag)::value;
        mul_mat_vec_q_moe_launch<T>(w, xq, ids, fusion, dst, uint32_t(ncols), nch_y, uint32_t(nrows), s_row_x, s_col_y,
                                    uint32_t(nrows * k), s_ch_x, s_ch_y, uint32_t(nrows), uint32_t(n_tok), uint32_t(k), warp, k,
                                    stream);
    };
    switch (ggml_type(t)) {
        case GGML_TYPE_Q2_0: run(std::integral_constant<ggml_type, GGML_TYPE_Q2_0>{}); break;
        case GGML_TYPE_Q4_0: run(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{}); break;
        case GGML_TYPE_Q5_0: run(std::integral_constant<ggml_type, GGML_TYPE_Q5_0>{}); break;
        case GGML_TYPE_Q8_0: run(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{}); break;
        case GGML_TYPE_Q4_K: run(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: run(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K: run(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{}); break;
        default: throw std::runtime_error("gemv::moe_q: unsupported type " + std::to_string(t));
    }
}

void dequantize(uint32_t t, const void* src, float* dst, int64_t n, cudaStream_t stream) {
    if (!supported(t) || n % block_size(t)) throw std::runtime_error("gemv::dequantize: unsupported type or length");
    if (t == GGML_TYPE_F32) {
        if (cudaMemcpyAsync(dst, src, size_t(n) * 4, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            throw std::runtime_error("gemv::dequantize: copy failed");
        return;
    }
    const to_fp32_cuda_t fn = ggml_get_to_fp32_cuda(ggml_type(t));
    if (!fn) throw std::runtime_error("gemv::dequantize: no kernel for type " + std::to_string(t));
    fn(src, dst, n, stream);
}

}  // namespace flashrt::gemv
