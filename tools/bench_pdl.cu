// SPDX-License-Identifier: Apache-2.0
// bench_pdl: what programmatic dependent launch saves per kernel boundary on this GPU, inside a
// CUDA graph, for a chain of small dependent kernels shaped like decode's (84-336 blocks of 256
// threads). Each kernel streams its own weight slice (independent of the chain) and reads the
// previous kernel's output (dependent). Arms:
//   plain:        ordinary launches;
//   pdl:          launched as programmatic dependents; each kernel triggers at its start and waits
//                 (cudaGridDependencySynchronize) before touching anything;
//   pdl+prefetch: as pdl, but the weights are loaded before the wait.
//
//   bench_pdl [chain length] [weight KiB per kernel]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// y[i] = x[i] * 0.5 + sum of this thread's weight words (as floats); PF: weights before the wait
template <bool PDL, bool PF>
__global__ void __launch_bounds__(256) k_link(const float* x, float* y, const uint4* w, int words_per_thread) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
#if __CUDA_ARCH__ >= 900
    if (PDL) cudaTriggerProgrammaticLaunchCompletion();
#endif
    float acc = 0.0f;
    uint4 r[8];
    const int nw = words_per_thread < 8 ? words_per_thread : 8;
    if (PF)
#pragma unroll
        for (int k = 0; k < 8; ++k)
            if (k < nw) r[k] = __ldg(w + size_t(k) * nt + i);
#if __CUDA_ARCH__ >= 900
    if (PDL) cudaGridDependencySynchronize();
#endif
    if (!PF)
#pragma unroll
        for (int k = 0; k < 8; ++k)
            if (k < nw) r[k] = __ldg(w + size_t(k) * nt + i);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (k < nw) acc += __uint_as_float(r[k].x & 0x3f800000u) + __uint_as_float(r[k].w & 0x3f800000u);
    y[i] = x[i] * 0.5f + acc * 1e-6f;
}
}  // namespace

int main(int argc, char** argv) {
    const int chain = argc > 1 ? std::atoi(argv[1]) : 600;
    const int wkib = argc > 2 ? std::atoi(argv[2]) : 64;
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    std::printf("chain of %d kernels, %d KiB of weights each (distinct per kernel)\n", chain, wkib);
    for (const int blocks : {84, 168, 336}) {
        const int nt = blocks * 256;
        const int wpt = std::max(1, int((size_t(wkib) * 1024 / 16) / nt));   // uint4 words per thread
        float* buf;
        uint4* w;
        ck(cudaMalloc(&buf, size_t(2) * nt * 4), "buf");
        ck(cudaMemset(buf, 0, size_t(2) * nt * 4), "memset");
        ck(cudaMalloc(&w, size_t(chain) * wpt * nt * 16), "weights");
        ck(cudaMemset(w, 0, size_t(chain) * wpt * nt * 16), "memset w");
        for (int arm = 0; arm < 3; ++arm) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "capture");
            for (int k = 0; k < chain; ++k) {
                const float* x = buf + size_t(k & 1) * nt;
                float* y = buf + size_t((k + 1) & 1) * nt;
                const uint4* wk = w + size_t(k) * wpt * nt;
                if (arm == 0) k_link<false, false><<<blocks, 256, 0, st>>>(x, y, wk, wpt);
                else {
                    cudaLaunchConfig_t cfg{};
                    cfg.gridDim = dim3(blocks);
                    cfg.blockDim = dim3(256);
                    cfg.stream = st;
                    cudaLaunchAttribute at[1];
                    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at[0].val.programmaticStreamSerializationAllowed = 1;
                    cfg.attrs = at;
                    cfg.numAttrs = 1;
                    if (arm == 1) ck(cudaLaunchKernelEx(&cfg, k_link<true, false>, x, y, wk, wpt), "launch");
                    else ck(cudaLaunchKernelEx(&cfg, k_link<true, true>, x, y, wk, wpt), "launch");
                }
            }
            ck(cudaStreamEndCapture(st, &g), "end capture");
            ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
            for (int r = 0; r < 3; ++r) ck(cudaGraphLaunch(ge, st), "warm");
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 20;
            cudaEventRecord(e0, st);
            for (int r = 0; r < reps; ++r) cudaGraphLaunch(ge, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            static const char* names[] = {"plain", "pdl", "pdl+prefetch"};
            std::printf("  %3d blocks, %-13s %6.2f us per kernel\n", blocks, names[arm], 1e3 * ms / reps / chain);
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
        }
        cudaFree(buf);
        cudaFree(w);
    }
    return 0;
}
