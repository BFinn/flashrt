// SPDX-License-Identifier: Apache-2.0
// Speculative sampling with sampled drafts (sample::draft_row + sample::spec_verify) is exact in
// distribution: over many seeds, the token emitted at the draft's position must follow the
// target's own sampling distribution p (the sampler chain: top-k 20, top-p 0.95, temperature 1),
// and drafts must be accepted at rate sum min(p, q). Synthetic rows: target logits with a few
// strong tokens, and drafter logits that are the target's plus noise (so q is close to p but not
// equal), over 4,096 entries.
// Second case: a window of two drafts (three verify rows, as --spec 2), the drafter over a trimmed
// vocabulary of 2,048 entries mapped to token ids (as the MTP head's trimmed LM head). The token
// emitted at each position, given the window reached it, must follow that row's p, and each draft
// must be accepted at rate sum min(p, q) with q over the remapped ids.
//
//   test_spec_sample
#include "kernels/cuda/sample.h"
#include "tests/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <vector>

using namespace flashrt;

namespace {
// the chain's distribution of a logits row, as token -> probability
std::map<int, double> chain(const std::vector<float>& lg, const sample::Params& p) {
    std::vector<int> ix(lg.size());
    std::iota(ix.begin(), ix.end(), 0);
    const int k = p.top_k;
    std::partial_sort(ix.begin(), ix.begin() + k, ix.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    float cl[sample::kMaxTopK], pr[sample::kMaxTopK];
    for (int i = 0; i < k; ++i) cl[i] = lg[ix[i]];
    const int n = sample::chain_probs(cl, k, p, pr);
    std::map<int, double> m;
    for (int i = 0; i < n; ++i) m[ix[i]] = pr[i];
    return m;
}
}  // namespace

int main() {
    const int V = 4096, N = 200000;
    const int64_t pos = 1234;
    sample::Params sp;
    sp.temperature = 1.0f;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> tl(V), dl(V);
    for (int i = 0; i < V; ++i) tl[i] = nd(rng);
    for (int i = 0; i < 12; ++i) tl[size_t(rng() % V)] += 4.0f + 1.5f * nd(rng);   // a few strong tokens
    for (int i = 0; i < V; ++i) dl[i] = tl[i] + 0.7f * nd(rng);
    const std::map<int, double> P = chain(tl, sp), Q = chain(dl, sp);
    double overlap = 0, p_argmax_q = 0;
    for (auto& [t, v] : P) overlap += std::min(v, Q.count(t) ? Q.at(t) : 0.0);
    {
        int am = 0;
        for (int i = 1; i < V; ++i)
            if (dl[i] > dl[am]) am = i;
        p_argmax_q = P.count(am) ? P.at(am) : 0.0;
    }

    float *d_t, *d_d;
    CUDA_CHECK(cudaMalloc(&d_t, size_t(2) * V * 4));   // rows: the draft's position, then the last row (the same logits)
    CUDA_CHECK(cudaMalloc(&d_d, size_t(V) * 4));
    CUDA_CHECK(cudaMemcpy(d_t, tl.data(), size_t(V) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t + V, tl.data(), size_t(V) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, dl.data(), size_t(V) * 4, cudaMemcpyHostToDevice));
    std::vector<sample::DraftCfg> cfg(N);
    for (int i = 0; i < N; ++i) cfg[size_t(i)] = sample::DraftCfg{sp, 1000003ull * uint64_t(i) + 17};
    sample::DraftCfg* d_cfg;
    CUDA_CHECK(cudaMalloc(&d_cfg, size_t(N) * sizeof(sample::DraftCfg)));
    CUDA_CHECK(cudaMemcpy(d_cfg, cfg.data(), size_t(N) * sizeof(sample::DraftCfg), cudaMemcpyHostToDevice));
    const int32_t dp_h[4] = {0, int32_t(pos - 1), 0, 0};
    int32_t *d_dp, *d_v, *d_qi, *d_qn, *d_out;
    float* d_qp;
    CUDA_CHECK(cudaMalloc(&d_dp, 16));
    CUDA_CHECK(cudaMemcpy(d_dp, dp_h, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_v, 16));
    CUDA_CHECK(cudaMalloc(&d_qi, sample::kMaxTopK * 4));
    CUDA_CHECK(cudaMalloc(&d_qp, sample::kMaxTopK * 4));
    CUDA_CHECK(cudaMalloc(&d_qn, 4));
    CUDA_CHECK(cudaMalloc(&d_out, size_t(N) * 3 * 4));
    for (int i = 0; i < N; ++i) {
        sample::draft_row(d_d, V, d_cfg + i, d_dp, nullptr, d_v, d_qi, d_qp, d_qn, nullptr);
        sample::spec_verify(d_t, 2, V, sp, cfg[size_t(i)].seed, pos, d_v + 1, d_qi, d_qp, d_qn, d_out + size_t(i) * 3, nullptr);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::vector<int32_t> out(size_t(N) * 3);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
    std::map<int, long> h0, h1;
    long acc = 0;
    for (int i = 0; i < N; ++i) {
        ++h0[out[size_t(i) * 3]];
        ++h1[out[size_t(i) * 3 + 1]];
        acc += out[size_t(i) * 3 + 2];
    }
    auto tv = [&](const std::map<int, long>& h) {
        double d = 0;
        for (auto& [t, v] : P) d += std::fabs(double(h.count(t) ? h.at(t) : 0) / N - v);
        for (auto& [t, c] : h)
            if (!P.count(t)) d += double(c) / N;
        return d / 2;
    };
    const double tv0 = tv(h0), tv1 = tv(h1), ar = double(acc) / N;
    // sampling noise: TV over ~|P| bins with N draws is about sqrt(|P| / (2 pi N)); allow 3x
    const double tol = 3.0 * std::sqrt(double(P.size()) / (2.0 * M_PI * N));
    bool ok = tv0 < tol && tv1 < tol && std::fabs(ar - overlap) < 0.005;
    std::printf("p over %zu tokens, q over %zu: sum min(p, q) %.4f, p(argmax q) %.4f\n", P.size(), Q.size(), overlap, p_argmax_q);
    std::printf("%d seeds: acceptance %.4f; emitted token at the draft's position TV %.4f, last row TV %.4f (noise bound %.4f) %s\n", N, ar,
                tv0, tv1, tol, ok ? "ok" : "FAIL");

    // ---- two drafts, trimmed drafter vocabulary
    const int VD = 2048, R = 3;
    std::vector<int32_t> ids(V);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(VD);   // the drafter's entries: half the vocabulary, in shuffled order
    std::vector<std::vector<float>> trow(R, std::vector<float>(V)), drow(R - 1, std::vector<float>(VD));
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < V; ++i) trow[r][i] = nd(rng);
        for (int i = 0; i < 10; ++i) trow[r][ids[size_t(rng() % VD)]] += 4.0f + 1.5f * nd(rng);   // strong tokens the drafter has
        trow[r][size_t(rng() % V)] += 4.0f;                                                      // and one it may not have
        if (r < R - 1)
            for (int e = 0; e < VD; ++e) drow[r][e] = trow[r][ids[e]] + 0.7f * nd(rng);
    }
    std::vector<std::map<int, double>> Pr(R), Qr(R - 1);
    std::vector<double> ov(R - 1, 0.0);
    for (int r = 0; r < R; ++r) Pr[r] = chain(trow[r], sp);
    for (int r = 0; r < R - 1; ++r) {
        for (auto& [e, v] : chain(drow[r], sp)) Qr[r][ids[e]] = v;   // q over token ids
        for (auto& [t, v] : Pr[r]) ov[r] += std::min(v, Qr[r].count(t) ? Qr[r].at(t) : 0.0);
    }
    float *d_rows, *d_drafter;
    int32_t *d_ids, *d_drafts, *d_dp2;
    CUDA_CHECK(cudaMalloc(&d_rows, size_t(R) * V * 4));
    CUDA_CHECK(cudaMalloc(&d_drafter, size_t(R - 1) * VD * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, size_t(VD) * 4));
    CUDA_CHECK(cudaMalloc(&d_drafts, size_t(R - 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_dp2, size_t(R - 1) * 16));
    for (int r = 0; r < R; ++r) CUDA_CHECK(cudaMemcpy(d_rows + size_t(r) * V, trow[r].data(), size_t(V) * 4, cudaMemcpyHostToDevice));
    for (int r = 0; r < R - 1; ++r) {
        CUDA_CHECK(cudaMemcpy(d_drafter + size_t(r) * VD, drow[r].data(), size_t(VD) * 4, cudaMemcpyHostToDevice));
        const int32_t dp[4] = {0, int32_t(pos - 1 + r), 0, r};   // draft r is for position pos + r, step r
        CUDA_CHECK(cudaMemcpy(d_dp2 + 4 * r, dp, 16, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), size_t(VD) * 4, cudaMemcpyHostToDevice));
    int32_t *d_qi2, *d_qn2, *d_out2;
    float* d_qp2;
    CUDA_CHECK(cudaMalloc(&d_qi2, size_t(R - 1) * sample::kMaxTopK * 4));
    CUDA_CHECK(cudaMalloc(&d_qp2, size_t(R - 1) * sample::kMaxTopK * 4));
    CUDA_CHECK(cudaMalloc(&d_qn2, size_t(R - 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_out2, size_t(N) * (2 * R - 1) * 4));
    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < R - 1; ++r) {
            sample::draft_row(d_drafter + size_t(r) * VD, VD, d_cfg + i, d_dp2 + 4 * r, d_ids, d_v, d_qi2, d_qp2, d_qn2, nullptr);
            CUDA_CHECK(cudaMemcpyAsync(d_drafts + r, d_v + 1, 4, cudaMemcpyDeviceToDevice, nullptr));
        }
        sample::spec_verify(d_rows, R, V, sp, cfg[size_t(i)].seed, pos, d_drafts, d_qi2, d_qp2, d_qn2, d_out2 + size_t(i) * (2 * R - 1), nullptr);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::vector<int32_t> out2(size_t(N) * (2 * R - 1));
    CUDA_CHECK(cudaMemcpy(out2.data(), d_out2, out2.size() * 4, cudaMemcpyDeviceToHost));
    // position r is reached when drafts 0 .. r-1 were accepted; its token is the emitted one there
    std::vector<std::map<int, long>> hist(R);
    std::vector<long> reached(R, 0), accepted(R - 1, 0);
    for (int i = 0; i < N; ++i) {
        const int32_t* o = out2.data() + size_t(i) * (2 * R - 1);
        for (int r = 0; r < R; ++r) {
            ++reached[r];
            ++hist[r][o[r]];
            if (r == R - 1 || !o[R + r]) break;
            ++accepted[r];
        }
    }
    for (int r = 0; r < R; ++r) {
        double d = 0;
        const double n = double(reached[r]);
        for (auto& [t, v] : Pr[r]) d += std::fabs(double(hist[r].count(t) ? hist[r].at(t) : 0) / n - v);
        for (auto& [t, c] : hist[r])
            if (!Pr[r].count(t)) d += double(c) / n;
        d /= 2;
        const double tol_r = 3.0 * std::sqrt(double(Pr[r].size()) / (2.0 * M_PI * n));
        bool row_ok = d < tol_r;
        std::printf("window row %d: reached %ld times, emitted token TV %.4f (noise bound %.4f)", r, reached[r], d, tol_r);
        if (r < R - 1) {
            const double a = double(accepted[r]) / n, noise = 3.0 * std::sqrt(ov[r] * (1 - ov[r]) / n);
            row_ok = row_ok && std::fabs(a - ov[r]) < std::max(noise, 0.003);
            std::printf(", draft acceptance %.4f against sum min(p, q) %.4f", a, ov[r]);
        }
        std::printf(" %s\n", row_ok ? "ok" : "FAIL");
        ok = ok && row_ok;
    }
    return ok ? 0 : 1;
}
