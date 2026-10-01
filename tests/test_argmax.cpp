// SPDX-License-Identifier: Apache-2.0
// argmax_dev (greedy decoding's argmax over a logits row) against a CPU reference with the same
// rule: the largest value, the lowest index on ties, NaN never taken (all -inf or NaN: INT32_MAX).
// The target's vocabulary (262,144) and the drafter's (an odd length, from an unaligned pointer),
// many ties, the maximum at either end; then its time at 262K. FLASHRT_ARGMAX_CLUSTER=0 runs the
// one-CTA kernel.
//
//   test_argmax
#include "arch/qwen4exp/blocks.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace flashrt::qwen4exp;

namespace {
int32_t ref_argmax(const float* x, int n) {
    float v = -INFINITY;
    int32_t idx = 0x7fffffff;
    for (int i = 0; i < n; ++i)
        if (x[i] > v) { v = x[i]; idx = i; }
    return idx;
}
}  // namespace

int main() {
    constexpr int NV = 262144;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 4.0f);
    float* dx;
    int32_t* dout;
    cudaMalloc(&dx, (NV + 4) * 4);
    cudaMalloc(&dout, 4);
    cudaStream_t st;
    cudaStreamCreate(&st);
    int fail = 0;
    auto check = [&](const char* what, const std::vector<float>& x, int off) {
        const int n = int(x.size());
        cudaMemcpy(dx + off, x.data(), size_t(n) * 4, cudaMemcpyHostToDevice);
        argmax_dev(st, dx + off, n, dout);
        int32_t got = -1;
        cudaMemcpyAsync(&got, dout, 4, cudaMemcpyDeviceToHost, st);
        if (cudaStreamSynchronize(st) != cudaSuccess) {
            std::printf("%s: CUDA error\n", what);
            ++fail;
            return;
        }
        const int32_t want = ref_argmax(x.data(), n);
        const bool ok = got == want;
        fail += !ok;
        std::printf("%-44s n %6d: %d (reference %d) %s\n", what, n, got, want, ok ? "ok" : "FAIL");
    };
    for (int n : {NV, 40001, 33, 1}) {
        std::vector<float> x(n);
        for (auto& v : x) v = nd(rng);
        check("random", x, n == NV ? 0 : 1);
        for (auto& v : x) v = std::round(v * 0.25f);   // many ties at the maximum
        check("rounded (ties)", x, n == NV ? 0 : 1);
        x[n - 1] = 1e30f;
        check("maximum last", x, 0);
        x[0] = 1e30f;
        check("maximum first and last", x, 1);
        for (int i = 0; i < n; i += 7) x[i] = NAN;
        check("NaN every 7th", x, 0);
        for (auto& v : x) v = -INFINITY;
        check("all -inf", x, 1);
    }
    {
        std::vector<float> x(NV);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), size_t(NV) * 4, cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int iters = 1000;
        argmax_dev(st, dx, NV, dout);
        cudaEventRecord(e0, st);
        for (int it = 0; it < iters; ++it) argmax_dev(st, dx, NV, dout);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1e3 * ms / iters;
        std::printf("262K: %.2f us per call, %.0f GB/s\n", us, NV * 4.0 / (us * 1e3));
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
