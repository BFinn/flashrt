// SPDX-License-Identifier: Apache-2.0
// The GPU sampler (kernels/cuda/sample) against the host: (1) for random rows and draws, the
// kernel's token equals the host chain (exact top-k by sorting, then sample::choose); (2) over
// many draws of one row, the empirical frequencies match the chain's exact probabilities
// (computed in double) within 5 standard deviations; (3) greedy returns the argmax.
#include "kernels/cuda/sample.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <vector>

using namespace flashrt;

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            return 2;                                                                                \
        }                                                                                            \
    } while (0)

namespace {
// the top k of a row, (value desc, index asc)
void topk(const float* x, int V, int k, std::vector<float>& v, std::vector<int32_t>& id) {
    std::vector<int32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    v.resize(k);
    id.assign(idx.begin(), idx.begin() + k);
    for (int i = 0; i < k; ++i) v[i] = x[id[i]];
}
// the chain's exact probabilities in double
std::map<int32_t, double> chain_probs(const std::vector<float>& v, const std::vector<int32_t>& id, const sample::Params& p) {
    const int k = int(v.size());
    std::vector<double> pr(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += pr[i] = std::exp(double(v[i]) - v[0]);
    int n = k;
    double cum = 0;
    for (int i = 0; i < k && p.top_p < 1.0f; ++i) {
        cum += pr[i] / sum;
        if (cum >= p.top_p) { n = i + 1; break; }
    }
    for (int i = 1; i < n && p.min_p > 0; ++i)
        if (pr[i] < p.min_p) { n = i; break; }
    std::map<int32_t, double> out;
    double qs = 0;
    for (int i = 0; i < n; ++i) qs += std::exp((double(v[i]) - v[0]) / p.temperature);
    for (int i = 0; i < n; ++i) out[id[i]] = std::exp((double(v[i]) - v[0]) / p.temperature) / qs;
    return out;
}
}  // namespace

int main() {
    const int V = 248320, R = 8;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> rows(size_t(R) * V);
    for (int r = 0; r < R; ++r) {
        float* x = rows.data() + size_t(r) * V;
        for (int i = 0; i < V; ++i) x[i] = nd(rng);
        for (int j = 0; j < 30; ++j) x[rng() % V] = 9.0f + 0.4f * float(j % 7) + 0.01f * float(r);   // a peaked head, with ties
    }
    float* dl;
    int32_t* dout;
    CK(cudaMalloc(&dl, rows.size() * 4));
    CK(cudaMalloc(&dout, R * 4));
    CK(cudaMemcpy(dl, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
    sample::Params p;   // temperature 1, top_k 20, top_p 0.95
    bool ok = true;

    // (1) kernel == host chain, row by row
    {
        long same = 0, n = 0;
        std::vector<float> v;
        std::vector<int32_t> id;
        std::vector<int32_t> out(R);
        for (int trial = 0; trial < 256; ++trial) {
            const uint64_t seed = 1000 + trial;
            sample::sample_rows(dl, R, V, p, seed, 77 * trial, dout, nullptr);
            CK(cudaMemcpy(out.data(), dout, R * 4, cudaMemcpyDeviceToHost));
            for (int r = 0; r < R; ++r) {
                topk(rows.data() + size_t(r) * V, V, p.top_k, v, id);
                same += out[r] == sample::choose(v.data(), id.data(), p.top_k, p, sample::draw(seed, 77 * trial + r));
                ++n;
            }
        }
        const bool pass = same >= n - n / 1000;   // host and device exp may differ in the last bit
        std::printf("kernel vs host chain: %ld of %ld draws equal: %s\n", same, n, pass ? "ok" : "FAIL");
        ok &= pass;
    }
    // (2) empirical frequencies vs exact probabilities, row 0 replicated
    {
        std::vector<float> rep(size_t(R) * V);
        for (int r = 0; r < R; ++r) std::copy(rows.begin(), rows.begin() + V, rep.begin() + size_t(r) * V);
        CK(cudaMemcpy(dl, rep.data(), rep.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> v;
        std::vector<int32_t> id;
        topk(rep.data(), V, p.top_k, v, id);
        const auto exact = chain_probs(v, id, p);
        std::map<int32_t, long> cnt;
        const int launches = 12000;
        std::vector<int32_t> out(R);
        for (int l = 0; l < launches; ++l) {
            sample::sample_rows(dl, R, V, p, 42, int64_t(l) * R, dout, nullptr);
            CK(cudaMemcpy(out.data(), dout, R * 4, cudaMemcpyDeviceToHost));
            for (int32_t t : out) ++cnt[t];
        }
        const double N = double(launches) * R;
        double worst = 0;
        bool outside = false;
        for (const auto& [t, c] : cnt) outside |= !exact.count(t);
        for (const auto& [t, q] : exact) {
            const double f = cnt.count(t) ? cnt[t] / N : 0.0, sd = std::sqrt(q * (1 - q) / N);
            worst = std::max(worst, std::fabs(f - q) / std::max(sd, 1e-12));
        }
        const bool pass = worst < 5.0 && !outside;
        std::printf("distribution: %zu tokens kept by the chain, %.0f draws, worst deviation %.2f sd%s: %s\n", exact.size(), N, worst,
                    outside ? ", a token outside the chain" : "", pass ? "ok" : "FAIL");
        ok &= pass;
    }
    // (3) greedy
    {
        sample::Params g;
        g.temperature = 0.0f;
        CK(cudaMemcpy(dl, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
        std::vector<int32_t> out(R);
        sample::sample_rows(dl, R, V, g, 1, 0, dout, nullptr);
        CK(cudaMemcpy(out.data(), dout, R * 4, cudaMemcpyDeviceToHost));
        bool pass = true;
        for (int r = 0; r < R; ++r) {
            const float* x = rows.data() + size_t(r) * V;
            pass &= out[r] == int32_t(std::max_element(x, x + V) - x);
        }
        std::printf("greedy: %s\n", pass ? "ok" : "FAIL");
        ok &= pass;
    }
    std::printf("%s\n", ok ? "all passed" : "FAILED");
    return ok ? 0 : 1;
}
