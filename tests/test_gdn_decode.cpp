// SPDX-License-Identifier: Apache-2.0
// The decode delta rule (gdn_delta_decode: the state in registers, as decode steps and verify
// windows run it) against a double-precision CPU recurrence, at the model's shape (48 heads, 16
// key groups, state 128), for T = 1..8. Then the property a speculative rewind rests on
// (gdn_rewind): replaying n tokens from the backup the window call saved gives the state of a
// fresh call over those n tokens, bit for bit.
//
//   test_gdn_decode
#include "arch/qwen4exp/blocks.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

namespace {
double rel(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

// S[h][j][i] *= exp(g); d_i = beta (v_i - sum_j S[j][i] k_j); S[j][i] += d_i k_j;
// o_i = sum_j S[j][i] q_j / sqrt(DK); key head h % G
void reference(std::vector<double>& S, const std::vector<float>& conv, const std::vector<float>& g, const std::vector<float>& beta,
               std::vector<double>& o, int T, int H, int G, int DK) {
    const int ch = 2 * G * DK + H * DK;
    const double scale = 1.0 / std::sqrt(double(DK));
    for (int t = 0; t < T; ++t) {
        const float* row = conv.data() + size_t(t) * ch;
        for (int h = 0; h < H; ++h) {
            const int hk = h % G;
            const float *q = row + size_t(hk) * DK, *k = row + size_t(G + hk) * DK, *v = row + size_t(2 * G) * DK + size_t(h) * DK;
            double* Sh = S.data() + size_t(h) * DK * DK;
            const double decay = std::exp(double(g[size_t(t) * H + h]));
            for (int x = 0; x < DK * DK; ++x) Sh[x] *= decay;
            for (int i = 0; i < DK; ++i) {
                double sk = 0;
                for (int j = 0; j < DK; ++j) sk += Sh[size_t(j) * DK + i] * k[j];
                const double d = beta[size_t(t) * H + h] * (v[i] - sk);
                double oi = 0;
                for (int j = 0; j < DK; ++j) {
                    Sh[size_t(j) * DK + i] += d * k[j];
                    oi += Sh[size_t(j) * DK + i] * q[j];
                }
                o[(size_t(t) * H + h) * DK + i] = oi * scale;
            }
        }
    }
}
}  // namespace

int main() {
    Spec s;
    s.ssm_heads = 48;
    s.ssm_state = 128;
    s.ssm_groups = 16;
    const int H = s.ssm_heads, DK = s.ssm_state, G = s.ssm_groups, ch = 2 * G * DK + H * DK, TMAX = 8;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> conv(size_t(TMAX) * ch), g(size_t(TMAX) * H), beta(size_t(TMAX) * H), S0(size_t(H) * DK * DK);
    for (int t = 0; t < TMAX; ++t) {
        float* row = conv.data() + size_t(t) * ch;
        for (int c = 0; c < ch; ++c) row[c] = nd(rng);
        for (int hh = 0; hh < 2 * G; ++hh) {   // L2-normed q and k heads, as the conv kernel leaves them
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
    const size_t sb = S0.size() * 4;
    float *dconv, *dg, *db, *dS, *dS2, *dbak, *dout;
    cudaMalloc(&dconv, conv.size() * 4);
    cudaMalloc(&dg, g.size() * 4);
    cudaMalloc(&db, beta.size() * 4);
    cudaMalloc(&dS, sb);
    cudaMalloc(&dS2, sb);
    cudaMalloc(&dbak, sb);
    cudaMalloc(&dout, size_t(TMAX) * H * DK * 4);
    cudaMemcpy(dconv, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice);
    int fail = 0;
    auto sync = [&] {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::exit(2);
        }
    };

    // 1. against the reference, T = 1..8, the state updated in place (a decode step or window)
    for (int T = 1; T <= TMAX; ++T) {
        cudaMemcpy(dS, S0.data(), sb, cudaMemcpyHostToDevice);
        gdn_delta_decode(s, dS, dS, nullptr, dconv, dg, db, dout, T, nullptr);
        sync();
        std::vector<float> o(size_t(T) * H * DK), S(S0.size());
        cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(S.data(), dS, sb, cudaMemcpyDeviceToHost);
        std::vector<double> Sr(S0.begin(), S0.end()), orf(o.size());
        reference(Sr, conv, g, beta, orf, T, H, G, DK);
        const double eo = rel(o, orf), es = rel(S, Sr);
        const bool ok = eo < 1e-5 && es < 1e-5;
        fail += !ok;
        std::printf("T %d: outputs %.2e, final state %.2e relative to double %s\n", T, eo, es, ok ? "ok" : "FAIL");
    }

    // 2. a window of T saves the state before it; replaying n of its tokens from that backup (as
    // gdn_rewind does) gives exactly the state of a fresh call over n tokens
    for (int T = 2; T <= 4; ++T)
        for (int n = 0; n < T; ++n) {
            cudaMemcpy(dS, S0.data(), sb, cudaMemcpyHostToDevice);
            gdn_delta_decode(s, dS, dS, dbak, dconv, dg, db, dout, T, nullptr);    // the window
            gdn_delta_decode(s, dbak, dS, nullptr, dconv, dg, db, dout, n, nullptr);   // the rewind to n
            cudaMemcpy(dS2, S0.data(), sb, cudaMemcpyHostToDevice);
            gdn_delta_decode(s, dS2, dS2, nullptr, dconv, dg, db, dout, n, nullptr);   // n tokens, fresh
            sync();
            std::vector<float> a(S0.size()), b(S0.size()), bak(S0.size());
            cudaMemcpy(a.data(), dS, sb, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dS2, sb, cudaMemcpyDeviceToHost);
            cudaMemcpy(bak.data(), dbak, sb, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(a.data(), b.data(), sb) == 0, bak_ok = std::memcmp(bak.data(), S0.data(), sb) == 0;
            if (!same || !bak_ok) {
                std::printf("window %d rewound to %d: %s%s\n", T, n, same ? "" : "state differs from a fresh run ",
                            bak_ok ? "" : "backup is not the state before the window");
                ++fail;
            }
        }
    std::printf("rewinds: windows of 2..4 rewound to every n, bit-identical to fresh runs: %s\n", fail ? "see above" : "ok");
    for (void* p : {static_cast<void*>(dconv), static_cast<void*>(dg), static_cast<void*>(db), static_cast<void*>(dS),
                    static_cast<void*>(dS2), static_cast<void*>(dbak), static_cast<void*>(dout)})
        cudaFree(p);
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
