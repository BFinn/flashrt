// SPDX-License-Identifier: Apache-2.0
// flashrt's launcher for ggml's MMQ kernels (third_party/ggml/src/ggml-cuda/mmq.cuh, MIT): the
// same grid and stream-k logic as ggml's launch_mul_mat_q, with the fixup buffer passed in
// instead of taken from a ggml backend context. Included by one translation unit per weight
// type (mmq_<type>.cu), which instantiates mmq_run<type>.
#pragma once

#include "src/ggml-cuda/mmq.cuh"

namespace flashrt::gemm::detail {

constexpr int kMmqJ[4] = {16, 32, 64, 128};

// Floats the stream-k fixup buffer needs (for every J and type: I = 128 on NVIDIA).
inline size_t mmq_fixup_floats() {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return size_t(nsm) * 128 * 128;
}

template <ggml_type type, int J, bool fallback>
void mmq_launch(const mmq_args& args, float* fixup, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
    const dim3 block_dims(warp_size, nwarps, 1);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, true>), nbytes_shared);

    const int nty = (args.nrows_x + config.I - 1) / config.I;
    const int ntx = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio = args.nsamples_y / args.nsamples_x;
    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd = init_fastdiv_values(sample_ratio);

    if (!config.stream_k) {
        const dim3 block_nums(nty, ntx, ntzw);
        mul_mat_q<type, J, fallback><<<block_nums, block_dims, nbytes_shared, stream>>>(
            args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale, blocks_per_ne00_fd, args.nrows_x,
            args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst, channel_ratio_fd, nchannels_y_fd, args.stride_channel_x,
            args.stride_channel_y, args.stride_channel_dst, sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y,
            args.stride_sample_dst, ntx_fd);
        return;
    }
    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm * tiles_nwaves);
    const dim3 block_nums_stream_k(tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);
    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;
    if (fixup_needed && size_t(block_nums_stream_k.x) * config.J * config.I > mmq_fixup_floats())
        GGML_ABORT("flashrt mmq: fixup buffer too small");
    mul_mat_q<type, J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>(
        args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, fixup, args.y_scale, blocks_per_ne00_fd, args.nrows_x,
        args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst, channel_ratio_fd, nchannels_y_fd, args.stride_channel_x,
        args.stride_channel_y, args.stride_channel_dst, sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y,
        args.stride_sample_dst, ntx_fd);
    if (!fixup_needed) return;
    const dim3 block_nums_fixup(block_nums_stream_k.x, config.I / warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y / 2, block_dims.z);
    mul_mat_q_stream_k_fixup<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>(
        args.ids_dst, args.expert_bounds, args.dst, fixup, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.nrows_dst,
        nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst, ntx_fd);
}

// Picks the tile width J for args.ncols_opt (the smallest of 16..128 covering it, else 128).
template <ggml_type type>
void mmq_run(const mmq_args& args, float* fixup, cudaStream_t stream) {
    const bool fallback = args.nrows_x % 128 != 0;
    int J = 128;
    for (int j : kMmqJ)
        if (args.ncols_opt <= j) {
            J = j;
            break;
        }
#define FLASHRT_MMQ_J(j)                                                        \
    if (J == j) {                                                               \
        if (fallback) mmq_launch<type, j, true>(args, fixup, stream);          \
        else mmq_launch<type, j, false>(args, fixup, stream);                  \
        return;                                                                 \
    }
    FLASHRT_MMQ_J(16)
    FLASHRT_MMQ_J(32)
    FLASHRT_MMQ_J(64)
    FLASHRT_MMQ_J(128)
#undef FLASHRT_MMQ_J
}

}  // namespace flashrt::gemm::detail

// One translation unit per weight type defines its entry point with this (the MMQ kernels are
// file-local templates, so each type's instances live in its own file and compile in parallel).
#define FLASHRT_MMQ_ENTRY(T, name)                                                                   \
    namespace flashrt::gemm::detail {                                                                \
    void name(const mmq_args& args, float* fixup, cudaStream_t stream) { mmq_run<T>(args, fixup, stream); } \
    }
