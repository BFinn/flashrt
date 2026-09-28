// SPDX-License-Identifier: Apache-2.0
// bench_mmq: gemm::gemm (ggml's MMQ) throughput per weight type on the dense shapes prefill runs
// (rows x cols of the Q3_K tensors), T tokens of random activations. The weights are random
// bytes with sane fp16 scales; MMQ's speed does not depend on the values.
//
//   bench_mmq [T]
#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace flashrt;

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 16384;
    struct Type {
        const char* name;
        uint32_t id;
        int blk, bytes;   // values per block, bytes per block
    };
    const Type types[] = {{"Q3_K", 11, 256, 110}, {"Q8_0", 8, 32, 34}, {"Q4_K", 12, 256, 144}, {"IQ4_XS", 23, 256, 136}, {"Q5_K", 13, 256, 176},
                          {"Q2_0", 42, 64, 18}};
    const int64_t shapes[][2] = {{10240, 2560}, {6144, 2560}, {12288, 2560}, {2560, 6144}};   // rows, cols
    std::mt19937 rng(1);
    float* x;
    float* y;
    cudaMalloc(&x, size_t(T) * 6144 * 4);
    cudaMalloc(&y, size_t(T) * 12288 * 4);
    {
        std::vector<float> h(size_t(T) * 6144);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : h) v = nd(rng);
        cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    }
    const size_t ws_bytes = gemm::workspace_bytes(6144, T);
    void* ws;
    cudaMalloc(&ws, ws_bytes);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const auto& sh : shapes) {
        const int64_t R = sh[0], K = sh[1];
        std::printf("rows %lld x cols %lld, T %lld:\n", (long long)R, (long long)K, (long long)T);
        for (const Type& ty : types) {
            if (K % ty.blk) continue;
            const size_t bytes = size_t(R) * (K / ty.blk) * ty.bytes;
            std::vector<uint8_t> h(bytes);
            for (auto& b : h) b = uint8_t(rng());
            // fp16 scales: 0x2000 (about 0.0078) wherever a block keeps one at its start (Q8_0, Q2_0) or
            // at its end (K-quants keep d there); IQ4_XS keeps d first too
            for (size_t b = 0; b < bytes; b += ty.bytes) {
                const size_t at = (ty.id == 11 || ty.id == 13) ? b + ty.bytes - 2 : (ty.id == 12 ? b : b);
                h[at] = 0x00;
                h[at + 1] = 0x20;
                if (ty.id == 12) {   // Q4_K: d and dmin first
                    h[b + 2] = 0x00;
                    h[b + 3] = 0x20;
                }
            }
            void* W;
            cudaMalloc(&W, bytes + gemv::kWeightTailPad);
            cudaMemset(W, 0, bytes + gemv::kWeightTailPad);
            cudaMemcpy(W, h.data(), bytes, cudaMemcpyHostToDevice);
            gemm::gemm(ty.id, W, x, y, K, R, T, ws, ws_bytes, nullptr);
            cudaDeviceSynchronize();
            const int reps = 5;
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) gemm::gemm(ty.id, W, x, y, K, R, T, ws, ws_bytes, nullptr);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            ms /= reps;
            std::printf("  %-7s %7.3f ms  %6.1f TOPS\n", ty.name, ms, 2.0 * R * K * T / (ms * 1e-3) / 1e12);
            cudaFree(W);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
