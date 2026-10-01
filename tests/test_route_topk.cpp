// SPDX-License-Identifier: Apache-2.0
// moe_route_topk (prefill routing: softmax, top-k by probability, ties by lower index, weights
// renormalised) against a CPU reference, 512 experts, top-10, 16,384 tokens of random logits
// with many exact ties; and its time.
//
//   test_route_topk
#include "arch/qwen4exp/blocks.hpp"
#include "tests/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

int main() {
    const int T = 16384, E = 512, K = 10;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> lg(size_t(T) * E);
    for (auto& v : lg) v = std::round(nd(rng) * 4.0f) / 4.0f;   // coarse: many ties
    float* d_lg;
    int32_t* d_ids;
    float* d_w;
    uint32_t* d_cnt;
    CUDA_CHECK(cudaMalloc(&d_lg, lg.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, size_t(T) * K * 4));
    CUDA_CHECK(cudaMalloc(&d_w, size_t(T) * K * 4));
    CUDA_CHECK(cudaMalloc(&d_cnt, E * 4));
    CUDA_CHECK(cudaMemcpy(d_lg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cnt, 0, E * 4));
    moe_route_topk(nullptr, d_lg, T, E, K, d_ids, d_w, d_cnt);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> ids(size_t(T) * K);
    std::vector<float> w(size_t(T) * K);
    std::vector<uint32_t> cnt(E);
    CUDA_CHECK(cudaMemcpy(ids.data(), d_ids, ids.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.data(), d_w, w.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt, E * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    double werr = 0;
    std::vector<uint32_t> rc(E, 0);
    std::vector<int> idx(E);
    for (int t = 0; t < T; ++t) {
        const float* l = lg.data() + size_t(t) * E;
        // the kernel ranks by its own softmax values; equal logits give equal probabilities, and
        // the softmax is monotonic, so ranking by logit (ties by index) is the same order
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        const float mx = l[idx[0]];
        double sum = 0;
        for (int e = 0; e < E; ++e) sum += std::exp(double(l[e]) - mx);
        double ws = 0;
        for (int k = 0; k < K; ++k) ws += std::exp(double(l[idx[k]]) - mx) / sum;
        for (int k = 0; k < K; ++k) {
            bad += ids[size_t(t) * K + k] != idx[k];
            ++rc[idx[k]];
            werr = std::max(werr, std::fabs(w[size_t(t) * K + k] - std::exp(double(l[idx[k]]) - mx) / sum / ws));
        }
    }
    int cbad = 0;
    for (int e = 0; e < E; ++e) cbad += cnt[e] != rc[e];
    const bool ok = bad == 0 && cbad == 0 && werr < 1e-5;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < 20; ++r) moe_route_topk(nullptr, d_lg, T, E, K, d_ids, d_w, nullptr);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("T %d, E %d, top-%d: %d wrong ids, %d wrong counts, max weight error %.1e %s; %.3f ms per call\n", T, E, K, bad, cbad, werr,
                ok ? "ok" : "FAIL", ms / 20);
    return ok ? 0 : 1;
}
