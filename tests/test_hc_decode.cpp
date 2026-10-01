// SPDX-License-Identifier: Apache-2.0
// The decode hyper-connection mix (hc_decode_raw: norm, down + inject, up, silu gate, gated mean)
// against a double-precision CPU reference, at qwen4exp's shape (4 streams of 2,560, rank 320;
// down Q8P, inject and up BF16), for T = 1..4; then its time with the weights rotated through
// > 256 MB of copies, so they come from VRAM as in a decode step, against the bytes it must read.
//
//   test_hc_decode
#include "arch/qwen4exp/blocks.hpp"
#include "tests/cuda_check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

namespace {
float bf2f(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

int main() {
    constexpr int HC = 4, n = 2560, rank = 320, NI = 4, W4 = HC * n;
    const float eps = 1e-6f;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // weights: down Q8P [rank][W4] + fp16 scales per 32; inject BF16 [NI][W4]; up BF16 [W4][rank]
    std::vector<int8_t> dq(size_t(rank) * W4);
    std::vector<__half> ds(dq.size() / 32);
    for (auto& v : dq) v = int8_t(int(nd(rng) * 40.0f) % 127);
    for (auto& v : ds) v = __float2half(0.0005f * (1.0f + std::fabs(nd(rng))));
    std::vector<uint16_t> wi(size_t(NI) * W4), wu(size_t(W4) * rank);
    for (auto& v : wi) v = f2bf(0.01f * nd(rng));
    for (auto& v : wu) v = f2bf(0.05f * nd(rng));
    std::vector<float> wn(W4), x(size_t(4) * W4);
    for (auto& v : wn) v = 1.0f + 0.1f * nd(rng);
    for (auto& v : x) v = nd(rng);
    const size_t down_bytes = dq.size() + ds.size() * 2, inj_bytes = wi.size() * 2, up_bytes = wu.size() * 2;
    const size_t set_bytes = down_bytes + inj_bytes + up_bytes;
    const int copies = int((256u << 20) / set_bytes) + 1;
    std::vector<char*> sets(copies);
    for (auto& p : sets) {
        CUDA_CHECK(cudaMalloc(&p, set_bytes));
        CUDA_CHECK(cudaMemcpy(p, dq.data(), dq.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p + dq.size(), ds.data(), ds.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p + down_bytes, wi.data(), inj_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p + down_bytes + inj_bytes, wu.data(), up_bytes, cudaMemcpyHostToDevice));
    }
    float *dx, *dwn, *xn, *part, *mixed, *inject;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
    CUDA_CHECK(cudaMalloc(&dwn, wn.size() * 4));
    CUDA_CHECK(cudaMalloc(&xn, size_t(4) * W4 * 4));
    CUDA_CHECK(cudaMalloc(&part, size_t(4) * HC * (rank + NI) * 4));
    CUDA_CHECK(cudaMalloc(&mixed, size_t(4) * n * 4));
    CUDA_CHECK(cudaMalloc(&inject, size_t(4) * NI * 4));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    auto run = [&](int T, int set) {
        char* p = sets[set];
        hc_decode_raw(T, dx, dwn, p, p + down_bytes, p + down_bytes + inj_bytes, xn, part, mixed, inject, n, rank, eps, st);
    };
    int fail = 0;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int T = 1; T <= 4; ++T) {
        run(T, 0);
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::vector<float> gm(size_t(T) * n), gi(size_t(T) * NI);
        CUDA_CHECK(cudaMemcpy(gm.data(), mixed, gm.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gi.data(), inject, gi.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, inum = 0, iden = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<double> xnr(W4), xs(rank);
            for (int g = 0; g < HC; ++g) {
                double ss = 0;
                for (int i = 0; i < n; ++i) ss += double(x[size_t(t) * W4 + g * n + i]) * x[size_t(t) * W4 + g * n + i];
                const double inv = 1.0 / std::sqrt(ss / n + eps);
                for (int i = 0; i < n; ++i) xnr[g * n + i] = x[size_t(t) * W4 + g * n + i] * inv * wn[g * n + i];
            }
            for (int j = 0; j < rank; ++j) {
                double a = 0;
                for (int e = 0; e < W4; ++e) a += double(dq[size_t(j) * W4 + e]) * __half2float(ds[(size_t(j) * W4 + e) / 32]) * xnr[e];
                const double v = a / HC;
                xs[j] = v / (1.0 + std::exp(-v));
            }
            for (int k = 0; k < NI; ++k) {
                double a = 0;
                for (int e = 0; e < W4; ++e) a += bf2f(wi[size_t(k) * W4 + e]) * xnr[e];
                inum += (gi[t * NI + k] - a) * (gi[t * NI + k] - a);
                iden += a * a;
            }
            for (int i = 0; i < n; ++i) {
                double m = 0;
                for (int s2 = 0; s2 < HC; ++s2) {
                    double gt = 0;
                    for (int j = 0; j < rank; ++j) gt += bf2f(wu[(size_t(s2) * n + i) * rank + j]) * xs[j];
                    m += xnr[s2 * n + i] / (1.0 + std::exp(-gt));
                }
                m /= HC;
                num += (gm[size_t(t) * n + i] - m) * (gm[size_t(t) * n + i] - m);
                den += m * m;
            }
        }
        const double em = std::sqrt(num / den), ei = std::sqrt(inum / iden);
        const bool ok = em < 1e-4 && ei < 1e-4;
        fail += !ok;
        const int iters = 300;
        float ms = 0;
        CUDA_CHECK(cudaEventRecord(e0, st));
        for (int it = 0; it < iters; ++it) run(T, it % copies);
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / iters;
        std::printf("T %d: mixed %.1e, inject %.1e relative %s; %.2f us per mix, %.0f GB/s of weights\n", T, em, ei, ok ? "ok" : "FAIL", us,
                    set_bytes / (us * 1e3));
    }
    // with the combine folded in: x += out * 2 sigmoid(inject / 4) first (the residual is updated in
    // place); checked against the same reference on the combined x
    {
        float *dco, *dci;
        CUDA_CHECK(cudaMalloc(&dco, size_t(4) * n * 4));
        CUDA_CHECK(cudaMalloc(&dci, size_t(4) * HC * 4));
        std::vector<float> co(size_t(4) * n), ci(size_t(4) * HC);
        for (auto& v : co) v = nd(rng);
        for (auto& v : ci) v = nd(rng);
        CUDA_CHECK(cudaMemcpy(dco, co.data(), co.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dci, ci.data(), ci.size() * 4, cudaMemcpyHostToDevice));
        for (int T = 1; T <= 4; ++T) {
            CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            char* p = sets[0];
            hc_decode_raw(T, dx, dwn, p, p + down_bytes, p + down_bytes + inj_bytes, xn, part, mixed, inject, n, rank, eps, st, dco, dci);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<float> gx(size_t(T) * W4), gm(size_t(T) * n), gi(size_t(T) * NI);
            CUDA_CHECK(cudaMemcpy(gx.data(), dx, gx.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gm.data(), mixed, gm.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gi.data(), inject, gi.size() * 4, cudaMemcpyDeviceToHost));
            double xnum = 0, xden = 0, num = 0, den = 0, inum = 0, iden = 0;
            for (int t = 0; t < T; ++t) {
                std::vector<double> xc(W4), xnr(W4), xs(rank);
                for (int g = 0; g < HC; ++g) {
                    const double wv = 2.0 / (1.0 + std::exp(-double(ci[t * HC + g]) / HC));
                    for (int i = 0; i < n; ++i) xc[g * n + i] = x[size_t(t) * W4 + g * n + i] + co[size_t(t) * n + i] * wv;
                }
                for (int e = 0; e < W4; ++e) {
                    xnum += (gx[size_t(t) * W4 + e] - xc[e]) * (gx[size_t(t) * W4 + e] - xc[e]);
                    xden += xc[e] * xc[e];
                }
                for (int g = 0; g < HC; ++g) {
                    double ss = 0;
                    for (int i = 0; i < n; ++i) ss += xc[g * n + i] * xc[g * n + i];
                    const double inv = 1.0 / std::sqrt(ss / n + eps);
                    for (int i = 0; i < n; ++i) xnr[g * n + i] = xc[g * n + i] * inv * wn[g * n + i];
                }
                for (int j = 0; j < rank; ++j) {
                    double a = 0;
                    for (int e = 0; e < W4; ++e) a += double(dq[size_t(j) * W4 + e]) * __half2float(ds[(size_t(j) * W4 + e) / 32]) * xnr[e];
                    const double v = a / HC;
                    xs[j] = v / (1.0 + std::exp(-v));
                }
                for (int k = 0; k < NI; ++k) {
                    double a = 0;
                    for (int e = 0; e < W4; ++e) a += bf2f(wi[size_t(k) * W4 + e]) * xnr[e];
                    inum += (gi[t * NI + k] - a) * (gi[t * NI + k] - a);
                    iden += a * a;
                }
                for (int i = 0; i < n; ++i) {
                    double m = 0;
                    for (int s2 = 0; s2 < HC; ++s2) {
                        double gt = 0;
                        for (int j = 0; j < rank; ++j) gt += bf2f(wu[(size_t(s2) * n + i) * rank + j]) * xs[j];
                        m += xnr[s2 * n + i] / (1.0 + std::exp(-gt));
                    }
                    m /= HC;
                    num += (gm[size_t(t) * n + i] - m) * (gm[size_t(t) * n + i] - m);
                    den += m * m;
                }
            }
            const double ex = std::sqrt(xnum / xden), em = std::sqrt(num / den), ei = std::sqrt(inum / iden);
            const bool ok = ex < 1e-6 && em < 1e-4 && ei < 1e-4;
            fail += !ok;
            const int iters = 300;
            float ms = 0;
            CUDA_CHECK(cudaEventRecord(e0, st));
            for (int it = 0; it < iters; ++it) {
                char* q = sets[it % copies];
                hc_decode_raw(T, dx, dwn, q, q + down_bytes, q + down_bytes + inj_bytes, xn, part, mixed, inject, n, rank, eps, st, dco, dci);
            }
            CUDA_CHECK(cudaEventRecord(e1, st));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("T %d with the combine: x %.1e, mixed %.1e, inject %.1e relative %s; %.2f us per combine + mix\n", T, ex, em, ei,
                        ok ? "ok" : "FAIL", 1e3 * ms / iters);
        }
        cudaFree(dco);
        cudaFree(dci);
    }
    std::printf("weights %.2f MB per mix\n%s\n", set_bytes / 1e6, fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
