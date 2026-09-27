// SPDX-License-Identifier: Apache-2.0
// GPU dense mat-vec (kernels/cuda/ggml_gemv) for every weight type qwen4exp's dense tensors
// use. For each type: a random matrix is quantized and dequantized with the llama.cpp
// dev tree's own libggml (ggml_quantize_chunk, type_traits->to_float), so the reference is
// the exact represented weights; the GPU result is compared with a double-precision W x for
// 1-4 tokens. Quantized types also carry the Q8_1 activation rounding (about 1%).
//
// Built only with -DFLASHRT_LLAMA_DIR (for libggml) and CUDA.
#include "ggml.h"
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            return 2;                                                                                \
        }                                                                                            \
    } while (0)

int main() {
    const ggml_type types[] = {GGML_TYPE_F32,  GGML_TYPE_F16,    GGML_TYPE_BF16, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0,
                               GGML_TYPE_Q5_0, GGML_TYPE_Q8_0,   GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K,
                               GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K};
    const int nrows = 384;
    const int shapes[] = {2560, 6144};
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    int fail = 0;

    for (int ncols : shapes) {
        std::vector<float> wf(size_t(nrows) * ncols), x(size_t(4) * ncols);
        for (auto& v : wf) v = 0.05f * nd(rng);
        for (auto& v : x) v = nd(rng);
        float *dx = nullptr, *dy = nullptr;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dy, size_t(4) * nrows * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        void* scratch = nullptr;
        CK(cudaMalloc(&scratch, flashrt::gemv::q8_1_bytes(ncols, 4)));

        for (ggml_type t : types) {
            const size_t rb = ggml_row_size(t, ncols);
            if (size_t(flashrt::gemv::row_bytes(t, ncols)) != rb) {
                std::printf("%-7s ncols %d: row_bytes %lld != ggml_row_size %zu FAIL\n", ggml_type_name(t), ncols,
                            (long long) flashrt::gemv::row_bytes(t, ncols), rb);
                ++fail;
                continue;
            }
            std::vector<uint8_t> wq(rb * nrows);
            if (t == GGML_TYPE_F32) std::memcpy(wq.data(), wf.data(), wq.size());
            else ggml_quantize_chunk(t, wf.data(), wq.data(), 0, nrows, ncols, nullptr);
            std::vector<float> wd(size_t(nrows) * ncols);
            if (t == GGML_TYPE_F32) wd = wf;
            else
                for (int r = 0; r < nrows; ++r)
                    ggml_get_type_traits(t)->to_float(wq.data() + size_t(r) * rb, wd.data() + size_t(r) * ncols, ncols);

            void* dw = nullptr;
            CK(cudaMalloc(&dw, wq.size() + flashrt::gemv::kWeightTailPad));
            CK(cudaMemset(dw, 0, wq.size() + flashrt::gemv::kWeightTailPad));
            CK(cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice));

            for (int nt = 1; nt <= 4; ++nt) {
                CK(cudaMemset(dy, 0xff, size_t(4) * nrows * 4));
                flashrt::gemv::matvec(t, dw, dx, dy, ncols, nrows, nt, scratch, st);
                CK(cudaStreamSynchronize(st));
                CK(cudaGetLastError());
                std::vector<float> y(size_t(nt) * nrows);
                CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (int k = 0; k < nt; ++k)
                    for (int r = 0; r < nrows; ++r) {
                        double ref = 0;
                        for (int c = 0; c < ncols; ++c) ref += double(wd[size_t(r) * ncols + c]) * x[size_t(k) * ncols + c];
                        const double d = y[size_t(k) * nrows + r] - ref;
                        num += d * d;
                        den += ref * ref;
                    }
                const double rel = std::sqrt(num / den);
                const double tol = t == GGML_TYPE_F32 ? 1e-5 : (t == GGML_TYPE_F16 || t == GGML_TYPE_BF16) ? 1e-3 : 3e-2;
                const bool ok = std::isfinite(rel) && rel < tol;
                fail += !ok;
                if (!ok || nt == 1 || nt == 4)
                    std::printf("%-7s ncols %5d, %d tok: rel L2 %.2e (tol %.0e) %s\n", ggml_type_name(t), ncols, nt, rel,
                                tol, ok ? "ok" : "FAIL");
            }
            CK(cudaFree(dw));
        }
        CK(cudaFree(dx));
        CK(cudaFree(dy));
        CK(cudaFree(scratch));
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
