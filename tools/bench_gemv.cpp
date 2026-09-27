// SPDX-License-Identifier: Apache-2.0
// bench_gemv: GPU mat-vec time for qwen4exp's dense weight shapes, and the VRAM bandwidth
// it reaches (the dense path is bandwidth-bound: ~3.5 GB read per token or verify window).
//
//   bench_gemv [--iters N] [--q3r]
//
// --q3r compares ggml's Q3_K MMVQ with the Q3R kernel variants (kernels/cuda/q3r.h) at
// qwen4exp's Q3_K shapes, one token.
//
// Weights are random bytes (timing only). Each case is timed with CUDA events over N calls
// for 1 and 4 tokens; quantized cases include the Q8_1 activation quantization. Calls rotate
// over enough copies of the matrix (>= 256 MB) that the 64 MB L2 cannot hold them, so the
// numbers are VRAM reads, as in a real forward pass.
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/q3r.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
struct Case {
    const char* name;
    uint32_t type;   // ggml type id
    int ncols, nrows;
};
// ggml type ids: F32 0, F16 1, Q4_0 2, Q5_0 6, Q8_0 8, Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14,
// IQ4_NL 20, IQ4_XS 23, BF16 30, Q2_0 42
const Case kCases[] = {
    {"attn_qkv (GDN)  IQ4_XS", 23, 2560, 10240},
    {"attn_gate       Q4_K", 12, 2560, 6144},
    {"ssm_out         Q5_K", 13, 6144, 2560},
    {"attn_q (QSA)    Q2_0", 42, 2560, 12288},
    {"attn_output     Q6_K", 14, 6144, 2560},
    {"hc_*_down       BF16", 30, 10240, 320},
    {"hc_*_up         BF16", 30, 320, 10240},
    {"ffn_gate_inp    BF16", 30, 2560, 512},
    {"ffn_up_shexp    Q3_K", 11, 2560, 640},
    {"ffn_down_shexp  Q4_0", 2, 640, 2560},
    {"output (head)   Q5_K", 13, 2560, 248320},
};
}  // namespace

int bench_q3r(int iters);

int main(int argc, char** argv) {
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--q3r")) return bench_q3r(iters);
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("bench_gemv on %s, %d iterations per case\n", prop.name, iters);
    std::printf("%-24s %8s %8s %11s %10s %10s %10s\n", "case", "cols", "rows", "MB", "us 1 tok", "GB/s", "us 4 tok");

    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937_64 rng(1);
    for (const Case& c : kCases) {
        const size_t wbytes = size_t(flashrt::gemv::row_bytes(c.type, c.ncols)) * c.nrows;
        std::vector<uint64_t> host((wbytes + 7) / 8);
        for (auto& v : host) v = rng() & 0x3f3f3f3f3f3f3f3full;   // keep float/fp16 exponents finite-ish
        const size_t slot = (wbytes + flashrt::gemv::kWeightTailPad + 255) & ~size_t(255);
        const int copies = int(std::max<size_t>(1, (size_t(256) << 20) / slot + 1));
        void* scratch;
        char* w;
        float *x, *y;
        cudaMalloc(reinterpret_cast<void**>(&w), slot * copies);
        cudaMemset(w, 0, slot * copies);
        for (int k = 0; k < copies; ++k) cudaMemcpy(w + slot * k, host.data(), wbytes, cudaMemcpyHostToDevice);
        cudaMalloc(&x, size_t(4) * c.ncols * 4);
        cudaMemset(x, 0, size_t(4) * c.ncols * 4);
        cudaMalloc(&y, size_t(4) * c.nrows * 4);
        cudaMalloc(&scratch, flashrt::gemv::q8_1_bytes(c.ncols, 4));
        float us[2];
        for (int k = 0; k < 2; ++k) {
            const int nt = k == 0 ? 1 : 4;
            for (int i = 0; i < 10; ++i) flashrt::gemv::matvec(c.type, w, x, y, c.ncols, c.nrows, nt, scratch, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i)
                flashrt::gemv::matvec(c.type, w + slot * (i % copies), x, y, c.ncols, c.nrows, nt, scratch, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            us[k] = ms * 1000.0f / iters;
        }
        const cudaError_t err = cudaGetLastError();
        std::printf("%-24s %8d %8d %11.2f %10.2f %10.0f %10.2f%s\n", c.name, c.ncols, c.nrows, wbytes / 1e6, us[0],
                    wbytes / (us[0] * 1e3), us[1], err == cudaSuccess ? "" : "  CUDA ERROR");
        cudaFree(w);
        cudaFree(x);
        cudaFree(y);
        cudaFree(scratch);
    }
    return 0;
}

// ggml Q3_K MMVQ against Q3R, 1 and 4 tokens, VRAM-resident (rotating copies).
int bench_q3r(int iters) {
    const int shapes[][2] = {{2560, 6144}, {2560, 10240}, {2560, 12288}, {6144, 2560}, {2560, 640}};
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937_64 rng(1);
    std::printf("%-14s %8s %12s %12s %12s %12s\n", "shape", "MB q3k", "ggml 1 tok", "Q3R 1 tok", "ggml 4 tok", "Q3R 4 tok");
    for (const auto& sh : shapes) {
        const int K = sh[0], R = sh[1];
        const size_t wbytes = size_t(flashrt::gemv::row_bytes(11, K)) * R;
        std::vector<uint64_t> host((wbytes + 7) / 8);
        for (auto& v : host) v = rng() & 0x3f3f3f3f3f3f3f3full;
        const size_t slot = (wbytes + flashrt::gemv::kWeightTailPad + 255) & ~size_t(255);
        const size_t qslot = (flashrt::q3r::bytes(R, K) + 255) & ~size_t(255);
        const int copies = int(std::max<size_t>(1, (size_t(256) << 20) / slot + 1));
        char *w, *wq;
        float *x, *y;
        void* scratch;
        cudaMalloc(reinterpret_cast<void**>(&w), slot * copies);
        cudaMalloc(reinterpret_cast<void**>(&wq), qslot * copies);
        cudaMemset(w, 0, slot * copies);
        for (int k = 0; k < copies; ++k) {
            cudaMemcpy(w + slot * k, host.data(), wbytes, cudaMemcpyHostToDevice);
            flashrt::q3r::repack(w + slot * k, wq + qslot * k, R, K, st);
        }
        cudaMalloc(&x, size_t(4) * K * 4);
        cudaMemset(x, 0, size_t(4) * K * 4);
        cudaMalloc(&y, size_t(4) * R * 4);
        cudaMalloc(&scratch, flashrt::gemv::q8_1_bytes(K, 4));
        float us[4];
        for (int v = 0; v < 4; ++v) {
            const int nt = v < 2 ? 1 : 4;
            auto call = [&](int i) {
                if (v % 2 == 0) flashrt::gemv::matvec(11, w + slot * (i % copies), x, y, K, R, nt, scratch, st);
                else flashrt::q3r::matvec(wq + qslot * (i % copies), x, y, R, K, nt, st);
            };
            for (int i = 0; i < 10; ++i) call(i);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) call(i);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            us[v] = ms * 1000.0f / iters;
        }
        const cudaError_t err = cudaGetLastError();
        char name[32];
        std::snprintf(name, sizeof(name), "%dx%d", K, R);
        std::printf("%-14s %8.2f %9.2f us %9.2f us %9.2f us %9.2f us%s\n", name, wbytes / 1e6, us[0], us[1], us[2], us[3],
                    err == cudaSuccess ? "" : "  CUDA ERROR");
        std::printf("%-14s %8s %9.0f GB/s %6.0f GB/s  (of the Q3_K bytes, 1 token)\n", "", "", wbytes / (us[0] * 1e3), wbytes / (us[1] * 1e3));
        cudaFree(w);
        cudaFree(wq);
        cudaFree(x);
        cudaFree(y);
        cudaFree(scratch);
    }
    return 0;
}
