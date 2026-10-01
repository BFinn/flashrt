// SPDX-License-Identifier: Apache-2.0
// qsa_select (the indexer's block selection) against a CPU reference: the top M blocks by score,
// ties taken in block order, listed in block order, then the incomplete tail's cells; dense below
// the width. Positions from within the width to 250K, T = 1 and 3 (decode and verify windows) and
// a prefill sub-batch, scores rounded so that many tie, with 8 CTAs per token and with 1; then
// its time for decode and prefill sub-batches.
//
//   test_idx_select
#include "arch/qwen4exp/blocks.hpp"
#include "tests/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace flashrt::qwen4exp;

namespace {
// the kernel's rule on the host
std::vector<int32_t> reference(const float* sc, int q, int r, int nsel, int width, int& count) {
    if (q + 1 <= width) {
        count = -1;
        return {};
    }
    const int nb = (q + 1) / r, tail = (q + 1) - nb * r, M = nsel - (tail > 0 ? 1 : 0);
    std::vector<int> blocks;
    if (nb <= M) {
        blocks.resize(nb);
        std::iota(blocks.begin(), blocks.end(), 0);
    } else {
        std::vector<int> idx(nb);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[a] > sc[b]; });   // ties: lower index first
        blocks.assign(idx.begin(), idx.begin() + M);
        std::sort(blocks.begin(), blocks.end());
    }
    std::vector<int32_t> out;
    for (int b : blocks)
        for (int k = 0; k < r; ++k) out.push_back(b * r + k);
    for (int k = 0; k < tail; ++k) out.push_back(nb * r + k);
    count = int(out.size());
    return out;
}
}  // namespace

int main() {
    const int r = 4, top_k = 2048, width = top_k + r - 1, nsel = (width + r - 1) / r, ldc = nsel * r;
    const int max_q = 250000, ld = max_q / r + 1;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* d_sc;
    int32_t *d_cells, *d_counts;
    const int Tmax = 128;
    CUDA_CHECK(cudaMalloc(&d_sc, size_t(Tmax) * ld * 4));
    CUDA_CHECK(cudaMalloc(&d_cells, size_t(Tmax) * ldc * 4));
    CUDA_CHECK(cudaMalloc(&d_counts, Tmax * 4));
    int fails = 0, cases = 0;
    struct Case { int pos0, T; float step; };
    const Case cs[] = {{100, 1, 0.25f}, {2040, 3, 0.25f}, {2051, 1, 0.25f}, {2100, 3, 0.1f}, {4096, 1, 0.5f}, {32768, 1, 0.25f},
                       {32769, 3, 0.25f}, {131070, 3, 0.05f}, {245760, 1, 0.25f}, {245761, 3, 1.0f}, {249990, 3, 0.25f},
                       {60000, 16, 0.25f}, {245000, 1, 0.0f}};
    for (int cl : {8, 4, 2, 1})   // CTAs per token: decode's cluster, prefill's single CTA (sw126)
    for (const Case& c : cs) {
        std::vector<float> sc(size_t(c.T) * ld);
        for (auto& v : sc) {
            v = c.step > 0 ? std::round(nd(rng) / c.step) * c.step : 1.0f;   // step 0: every score equal
            if (v == 0.0f) v = 0.0f;   // no -0: the kernel orders it below +0 (bit order), a float compare does not
        }
        CUDA_CHECK(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_cells, 0xff, size_t(Tmax) * ldc * 4));
        qsa_select(d_sc, ld, d_cells, d_counts, ldc, c.pos0, c.T, r, nsel, width, nullptr, cl);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("CUDA error at pos0 %d\n", c.pos0);
            return 1;
        }
        std::vector<int32_t> cells(size_t(c.T) * ldc), counts(c.T);
        CUDA_CHECK(cudaMemcpy(cells.data(), d_cells, cells.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, counts.size() * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < c.T; ++t) {
            int want_n = 0;
            const std::vector<int32_t> want = reference(sc.data() + size_t(t) * ld, c.pos0 + t, r, nsel, width, want_n);
            bool ok = counts[t] == want_n;
            for (int i = 0; ok && i < want_n; ++i) ok = cells[size_t(t) * ldc + i] == want[size_t(i)];
            ++cases;
            if (!ok) {
                ++fails;
                std::printf("MISMATCH (%d CTAs) pos %d (t %d): count %d, want %d\n", cl, c.pos0 + t, t, counts[t], want_n);
            }
        }
    }
    std::printf("%d of %d token selections match the reference\n", cases - fails, cases);
    // time: decode (T = 1, 3 at 245K) and a prefill sub-batch (T = 128 at 32K and 245K), each with
    // 8 CTAs per token and with 1
    struct Timing { int pos0, T; };
    for (const Timing tm : {Timing{245760, 1}, Timing{245760, 3}, Timing{32768, 128}, Timing{65536, 128}, Timing{131072, 128}, Timing{196608, 128}, Timing{245760, 128}})
    for (int cl : {8, 4, 2, 1}) {
        const int T = tm.T, p0 = tm.pos0;
        std::vector<float> sc(size_t(T) * ld);
        for (auto& v : sc) v = nd(rng);
        CUDA_CHECK(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < 3; ++i) qsa_select(d_sc, ld, d_cells, d_counts, ldc, p0, T, r, nsel, width, nullptr, cl);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        const int n = T > 3 ? 12 : 120;
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < n; ++i) qsa_select(d_sc, ld, d_cells, d_counts, ldc, p0, T, r, nsel, width, nullptr, cl);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("pos %d, T = %d, %d CTA(s) per token: %.1f us per call\n", p0, T, cl, 1000.0f * ms / n);
    }
    cudaFree(d_sc);
    cudaFree(d_cells);
    cudaFree(d_counts);
    return fails ? 1 : 0;
}
