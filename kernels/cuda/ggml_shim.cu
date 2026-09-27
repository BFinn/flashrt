// SPDX-License-Identifier: Apache-2.0
// Definitions for ggml runtime symbols that the vendored CUDA kernels reference but flashrt
// does not link (they live in ggml.c and ggml-cuda.cu). Only what the linker asks for is here.
#include "src/ggml-cuda/common.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace {
ggml_cuda_device_info make_info() {
    ggml_cuda_device_info info{};
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    info.device_count = info.physical_device_count = std::min(n, GGML_CUDA_MAX_DEVICES);
    for (int id = 0; id < info.device_count; ++id) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, id) != cudaSuccess) continue;
        auto& d = info.devices[id];
        d.cc = 100 * p.major + 10 * p.minor;   // ggml's encoding for NVIDIA devices
        d.nsm = p.multiProcessorCount;
        d.smpb = p.sharedMemPerBlock;
        d.smpbo = p.sharedMemPerBlockOptin;
        d.integrated = p.integrated != 0;
        d.total_vram = p.totalGlobalMem;
        d.warp_size = p.warpSize;
        d.supports_cooperative_launch = p.cooperativeLaunch != 0;
        d.physical_device = id;
        d.physical_share_count = 1;
        d.virtual_index = 0;
    }
    return info;
}
}  // namespace

const ggml_cuda_device_info& ggml_cuda_info() {
    static const ggml_cuda_device_info info = make_info();
    return info;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur == device) return;
    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    std::fprintf(stderr, "CUDA error: %s\n  in %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    std::abort();
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    std::fprintf(stderr, "flashrt: ggml's CUDA memory pool is not used; new_pool_for_device was called\n");
    std::abort();
}
