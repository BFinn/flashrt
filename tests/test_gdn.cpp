// SPDX-License-Identifier: Apache-2.0
// The chunked (WY) delta rule against the column kernel (the recurrence token by token), on
// synthetic GDN inputs of the model's shape: 48 heads, 16 key groups, state 128; q and k
// L2-normed, log decays g in about -3..0, beta in 0..1, a nonzero start state. T = 1,000 (a
// partial last chunk) and 3,000 (several slabs). The column kernel is fp32;
// the chunked form runs its products in fp16 (fp32 accumulation, fp32 state): outputs and final
// states agree to about 1e-3 relative (tolerance 1e-2).
//
//   test_gdn
#include "arch/qwen4exp/blocks.hpp"
#include "tests/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

namespace {
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}
}  // namespace

int main() {
    Spec s;
    s.ssm_heads = 48;
    s.ssm_state = 128;
    s.ssm_groups = 16;
    const int H = s.ssm_heads, DK = s.ssm_state, G = s.ssm_groups, ch = 2 * G * DK + H * DK;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    int fail = 0;
    for (const int T : {1000, 3000}) {
        std::vector<float> conv(size_t(T) * ch), g(size_t(T) * H), beta(size_t(T) * H), S0(size_t(H) * DK * DK);
        for (int t = 0; t < T; ++t) {
            float* row = conv.data() + size_t(t) * ch;
            for (int c = 0; c < ch; ++c) row[c] = nd(rng);
            for (int hh = 0; hh < 2 * G; ++hh) {   // L2-normed q and k heads
                double ss = 0;
                for (int d = 0; d < DK; ++d) ss += double(row[hh * DK + d]) * row[hh * DK + d];
                const float inv = float(1.0 / std::sqrt(ss + 1e-6));
                for (int d = 0; d < DK; ++d) row[hh * DK + d] *= inv;
            }
            for (int h = 0; h < H; ++h) {
                g[size_t(t) * H + h] = std::log(0.05f + 0.95f * ud(rng));
                beta[size_t(t) * H + h] = ud(rng);
            }
        }
        for (float& v : S0) v = 0.05f * nd(rng);
        float *dconv, *dg, *db, *dS1, *dS2, *do1, *do2;
        void* ws;
        CUDA_CHECK(cudaMalloc(&dconv, conv.size() * 4));
        CUDA_CHECK(cudaMalloc(&dg, g.size() * 4));
        CUDA_CHECK(cudaMalloc(&db, beta.size() * 4));
        CUDA_CHECK(cudaMalloc(&dS1, S0.size() * 4));
        CUDA_CHECK(cudaMalloc(&dS2, S0.size() * 4));
        CUDA_CHECK(cudaMalloc(&do1, size_t(T) * H * DK * 4));
        CUDA_CHECK(cudaMalloc(&do2, size_t(T) * H * DK * 4));
        CUDA_CHECK(cudaMalloc(&ws, gdn_chunk_ws_bytes(s)));
        CUDA_CHECK(cudaMemcpy(dconv, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dS1, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dS2, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice));
        gdn_delta_prefill(s, dS1, dconv, dg, db, do1, T, false, nullptr, nullptr);
        gdn_delta_prefill(s, dS2, dconv, dg, db, do2, T, true, ws, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 2;
        }
        std::vector<float> o1(size_t(T) * H * DK), o2(o1.size()), s1(S0.size()), s2(S0.size());
        CUDA_CHECK(cudaMemcpy(o1.data(), do1, o1.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o2.data(), do2, o2.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s1.data(), dS1, s1.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s2.data(), dS2, s2.size() * 4, cudaMemcpyDeviceToHost));
        const double eo = rel(o2, o1), es = rel(s2, s1);
        const bool ok = std::isfinite(eo) && std::isfinite(es) && eo < 1e-2 && es < 1e-2;
        fail += !ok;
        // timing at this T
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        float ms[2];
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < 5; ++r) gdn_delta_prefill(s, k ? dS2 : dS1, dconv, dg, db, k ? do2 : do1, T, k == 1, ws, nullptr);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms[k], e0, e1));
        }
        std::printf("T %5d: outputs %.2e, final state %.2e relative %s; column %.2f ms, chunked %.2f ms\n", T, eo, es, ok ? "ok" : "FAIL",
                    ms[0] / 5, ms[1] / 5);
        for (void* p : {static_cast<void*>(dconv), static_cast<void*>(dg), static_cast<void*>(db), static_cast<void*>(dS1),
                        static_cast<void*>(dS2), static_cast<void*>(do1), static_cast<void*>(do2), ws})
            cudaFree(p);
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
