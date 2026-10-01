// SPDX-License-Identifier: Apache-2.0
// linear_multi (several BF16 mat-vecs of one input in one launch) against separate ggml MMVF
// calls, on decode's shapes: router + shared-expert gate (512 + 1 rows), GDN alpha + beta
// (48 + 48), indexer q + k (512 + 128), K = 2560, for 1..8 tokens. Checks the outputs (no
// epilogue) and times both, weights rotated through > 256 MB so they come from VRAM. For alpha +
// beta, also the epilogues decode uses: the GDN decay softplus(z + dt_bias) * a, and sigmoid,
// against the same applied in double to the separate calls' outputs.
//
//   test_linear_multi
#include "arch/qwen4exp/blocks.hpp"
#include "tests/cuda_check.hpp"
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

int main() {
    constexpr int K = 2560;
    struct Shape {
        const char* name;
        int r0, r1;
    };
    const Shape shapes[] = {{"router + shexp gate", 512, 1}, {"alpha + beta", 48, 48}, {"indexer q + k", 512, 128}};
    Spec s;
    GpuWeights w;
    BlockScratch bs;
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    BlockCtx c{s, w, bs, st};
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* x;
    CUDA_CHECK(cudaMalloc(&x, size_t(8) * K * 4));
    {
        std::vector<float> h(size_t(8) * K);
        for (float& v : h) v = nd(rng);
        CUDA_CHECK(cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    int fail = 0;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (const Shape& sh : shapes) {
        const int rows = sh.r0 + sh.r1;
        const size_t mat = size_t(rows) * K * 2;
        const int copies = int(std::max<size_t>(1, (256u << 20) / mat));
        std::vector<__nv_bfloat16> hw(size_t(rows) * K);
        for (auto& v : hw) v = __float2bfloat16(0.02f * nd(rng));
        std::vector<void*> wd(copies);
        for (auto& p : wd) {
            CUDA_CHECK(cudaMalloc(&p, mat + gemv::kWeightTailPad));
            CUDA_CHECK(cudaMemcpy(p, hw.data(), mat, cudaMemcpyHostToDevice));
        }
        float *y0, *y1, *z0, *z1;
        CUDA_CHECK(cudaMalloc(&y0, size_t(8) * sh.r0 * 4));
        CUDA_CHECK(cudaMalloc(&y1, size_t(8) * sh.r1 * 4));
        CUDA_CHECK(cudaMalloc(&z0, size_t(8) * sh.r0 * 4));
        CUDA_CHECK(cudaMalloc(&z1, size_t(8) * sh.r1 * 4));
        auto tensors = [&](int i, GpuTensor& a, GpuTensor& b) {
            a.dev = wd[i];
            a.type = 30;
            a.dims = {K, sh.r0};
            b.dev = static_cast<char*>(wd[i]) + size_t(sh.r0) * K * 2;
            b.type = 30;
            b.dims = {K, sh.r1};
        };
        for (int T : {1, 2, 3, 4, 8}) {
            GpuTensor a, b;
            tensors(0, a, b);
            const LinearOut outs[2] = {{&a, y0}, {&b, y1}};
            if (!linear_multi_ok(outs, 2, T)) {
                std::printf("not applicable\n");
                return 1;
            }
            linear_multi(c, outs, 2, x, T);
            gemv::matvec(30, a.dev, x, z0, K, sh.r0, T, nullptr, st);
            gemv::matvec(30, b.dev, x, z1, K, sh.r1, T, nullptr, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<float> a0(size_t(T) * sh.r0), a1(size_t(T) * sh.r1), b0(a0.size()), b1(a1.size());
            CUDA_CHECK(cudaMemcpy(a0.data(), y0, a0.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(a1.data(), y1, a1.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b0.data(), z0, b0.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b1.data(), z1, b1.size() * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (size_t i = 0; i < a0.size(); ++i) num += (a0[i] - b0[i]) * (a0[i] - b0[i]), den += b0[i] * b0[i];
            for (size_t i = 0; i < a1.size(); ++i) num += (a1[i] - b1[i]) * (a1[i] - b1[i]), den += b1[i] * b1[i];
            const double err = std::sqrt(num / den);
            const bool ok = err < 1e-5;
            fail += !ok;
            float ms[2];
            const int iters = 400;
            for (int k = 0; k < 2; ++k) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                for (int it = 0; it < iters; ++it) {
                    tensors(it % copies, a, b);
                    if (k == 0) {
                        const LinearOut o2[2] = {{&a, y0}, {&b, y1}};
                        linear_multi(c, o2, 2, x, T);
                    } else {
                        gemv::matvec(30, a.dev, x, z0, K, sh.r0, T, nullptr, st);
                        gemv::matvec(30, b.dev, x, z1, K, sh.r1, T, nullptr, st);
                    }
                }
                CUDA_CHECK(cudaEventRecord(e1, st));
                CUDA_CHECK(cudaEventSynchronize(e1));
                CUDA_CHECK(cudaEventElapsedTime(&ms[k], e0, e1));
            }
            std::printf("%-20s T %d: error %.1e %s; fused %.2f us, separate %.2f us\n", sh.name, T, err, ok ? "ok" : "FAIL",
                        1e3 * ms[0] / iters, 1e3 * ms[1] / iters);
            if (sh.r0 == 48 && sh.r1 == 48) {   // the GDN gates' epilogues
                std::vector<float> bias(48), aa(48);
                for (int h = 0; h < 48; ++h) {
                    bias[h] = 0.5f * nd(rng) + (h % 7 == 0 ? 25.0f : 0.0f);   // some past softplus's linear threshold
                    aa[h] = -std::exp(0.5f * nd(rng));
                }
                float *pb, *pa;
                CUDA_CHECK(cudaMalloc(&pb, 48 * 4));
                CUDA_CHECK(cudaMalloc(&pa, 48 * 4));
                CUDA_CHECK(cudaMemcpy(pb, bias.data(), 48 * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(pa, aa.data(), 48 * 4, cudaMemcpyHostToDevice));
                tensors(0, a, b);
                LinearOut oe[2] = {{&a, y0, 1, pb, pa}, {&b, y1, 2}};
                linear_multi(c, oe, 2, x, T);
                CUDA_CHECK(cudaStreamSynchronize(st));
                std::vector<float> g(size_t(T) * 48), be(size_t(T) * 48);
                CUDA_CHECK(cudaMemcpy(g.data(), y0, g.size() * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(be.data(), y1, be.size() * 4, cudaMemcpyDeviceToHost));
                double emax = 0;
                for (size_t i = 0; i < g.size(); ++i) {
                    const int h = int(i % 48);
                    const double z = double(b0[i]) + bias[h];
                    const double gr = (z > 20.0 ? z : std::log1p(std::exp(z))) * aa[h], br = 1.0 / (1.0 + std::exp(-double(b1[i])));
                    emax = std::max({emax, std::fabs(g[i] - gr) / std::max(1e-3, std::fabs(gr)), std::fabs(be[i] - br)});
                }
                const bool eok = emax < 2e-5;
                fail += !eok;
                std::printf("%-20s T %d: epilogues (decay, sigmoid) max error %.1e %s\n", sh.name, T, emax, eok ? "ok" : "FAIL");
                cudaFree(pb);
                cudaFree(pa);
            }
        }
        for (void* p : wd) cudaFree(p);
        for (float* p : {y0, y1, z0, z1}) cudaFree(p);
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
