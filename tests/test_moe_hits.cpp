// SPDX-License-Identifier: Apache-2.0
// The GPU cache-hit expert path (arch/qwen4exp/moe_fast: moe_hits) against a double-precision
// reference on the same planar Q2_0 experts: random codes and scales, K experts in shuffled
// slots, of which only the first hit_n are computed. The int8 activation rounding (per 64
// values, twice: x and the hidden rows) gives about 1% relative error; the tolerance is 3%.
#include "arch/qwen4exp/moe_fast.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
// dequantized planar matrix row r, element j
double wval(const uint8_t* mat, int rows, int cols, int r, int j) {
    const int nb = cols / 64, b = j / 64, e = j % 64;
    const uint8_t code = (mat[(size_t(r) * nb + b) * 16 + e % 16] >> (2 * (e / 16))) & 3;
    const uint16_t hbits = reinterpret_cast<const uint16_t*>(mat + size_t(rows) * nb * 16)[size_t(r) * nb + b];
    __half_raw hr;
    hr.x = hbits;
    return (int(code) - 1) * double(__half2float(__half(hr)));
}
}  // namespace

int main() {
    const int n = 2560, ff = 640, K = 10, n_slots = 16;
    const q2_0::ExpertShape es{n, ff};
    const size_t eb = q2_0::expert_bytes(es), gu = q2_0::mat_bytes(ff, n);
    std::mt19937 rng(7);
    std::vector<uint8_t> host(eb * n_slots);
    for (int sl = 0; sl < n_slots; ++sl) {
        uint8_t* e = host.data() + eb * sl;
        const size_t mats[3][2] = {{0, size_t(ff)}, {gu, size_t(ff)}, {2 * gu, size_t(n)}};
        for (auto& m : mats) {
            const int rows = int(m[1]), cols = rows == ff ? n : ff, nb = cols / 64;
            uint8_t* p = e + m[0];
            for (size_t i = 0; i < size_t(rows) * nb * 16; ++i) p[i] = uint8_t(rng());
            uint16_t* sc = reinterpret_cast<uint16_t*>(p + size_t(rows) * nb * 16);
            std::uniform_real_distribution<float> ud(0.005f, 0.02f);
            for (size_t i = 0; i < size_t(rows) * nb; ++i) {
                const __half_raw hr = __half(__float2half(ud(rng)));
                sc[i] = hr.x;
            }
        }
    }
    std::vector<float> x(n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng);
    const int32_t slot_of[K] = {3, 11, 0, 7, 15, 5, 9, 2, 13, 3};   // the last is a pad entry
    const int hit_n = 9;

    uint8_t* d_slots;
    float *d_x, *d_yh;
    const uint8_t** d_ptr;
    int32_t* d_n;
    void* d_scr;
    CK(cudaMalloc(&d_slots, host.size()));
    CK(cudaMemcpy(d_slots, host.data(), host.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, n * 4));
    CK(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_yh, size_t(K) * n * 4));
    CK(cudaMemset(d_yh, 0xff, size_t(K) * n * 4));   // NaN: rows that must not be written stay NaN
    std::vector<const uint8_t*> ptrs(K);
    for (int k = 0; k < K; ++k) ptrs[k] = d_slots + eb * slot_of[k];
    CK(cudaMalloc(&d_ptr, K * sizeof(void*)));
    CK(cudaMemcpy(d_ptr, ptrs.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_n, 4));
    CK(cudaMemcpy(d_n, &hit_n, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_scr, qwen4exp::moe_hits_scratch_bytes(K, ff)));
    qwen4exp::moe_hits(d_ptr, d_n, K, d_x, n, ff, d_scr, d_yh, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> yh(size_t(K) * n);
    CK(cudaMemcpy(yh.data(), d_yh, yh.size() * 4, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int k = 0; k < K; ++k) {
        if (k >= hit_n) {
            bool untouched = true;
            for (int r = 0; r < n; ++r) untouched &= std::isnan(yh[size_t(k) * n + r]);
            std::printf("expert %d (pad): %s\n", k, untouched ? "not written, ok" : "WRITTEN, FAIL");
            fail += !untouched;
            continue;
        }
        const uint8_t* e = host.data() + eb * slot_of[k];
        std::vector<double> h(ff);
        for (int r = 0; r < ff; ++r) {
            double g = 0, u = 0;
            for (int j = 0; j < n; ++j) {
                g += wval(e, ff, n, r, j) * x[j];
                u += wval(e + gu, ff, n, r, j) * x[j];
            }
            h[r] = g / (1 + std::exp(-g)) * u;
        }
        double num = 0, den = 0;
        for (int r = 0; r < n; ++r) {
            double y = 0;
            for (int j = 0; j < ff; ++j) y += wval(e + 2 * gu, n, ff, r, j) * h[j];
            const double d = yh[size_t(k) * n + r] - y;
            num += d * d;
            den += y * y;
        }
        const double rel = std::sqrt(num / den);
        const bool ok = std::isfinite(rel) && rel < 3e-2;
        fail += !ok;
        std::printf("expert %d (slot %2d): rel L2 %.2e %s\n", k, slot_of[k], rel, ok ? "ok" : "FAIL");
    }

    // a window of 3 tokens with overlapping hit lists (grouped by expert inside moe_hits) must
    // give each (token, entry) exactly what a one-token call gives
    {
        const int T = 3;
        const int32_t win_slot[T][K] = {{3, 11, 0, 7, 15, 5, 9, 2, 13, 3},
                                        {11, 3, 6, 7, 1, 5, 9, 12, 14, 8},
                                        {0, 11, 4, 10, 15, 2, 6, 3, 13, 1}};
        const int32_t win_n[T] = {9, 10, 7};
        std::vector<float> xw(size_t(T) * n);
        for (auto& v : xw) v = nd(rng);
        std::vector<const uint8_t*> wptr(size_t(T) * K);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) wptr[size_t(t) * K + k] = d_slots + eb * win_slot[t][k];
        float *d_xw, *d_yw, *d_y1;
        const uint8_t** d_wptr;
        int32_t* d_wn;
        void* d_wscr;
        CK(cudaMalloc(&d_xw, xw.size() * 4));
        CK(cudaMemcpy(d_xw, xw.data(), xw.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_yw, size_t(T) * K * n * 4));
        CK(cudaMalloc(&d_y1, size_t(K) * n * 4));
        CK(cudaMalloc(&d_wptr, wptr.size() * sizeof(void*)));
        CK(cudaMemcpy(d_wptr, wptr.data(), wptr.size() * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_wn, T * 4));
        CK(cudaMemcpy(d_wn, win_n, T * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_wscr, qwen4exp::moe_hits_scratch_bytes(K, ff, T)));
        qwen4exp::moe_hits(d_wptr, d_wn, K, d_xw, n, ff, d_wscr, d_yw, nullptr, T);
        std::vector<float> yw(size_t(T) * K * n), y1(size_t(K) * n);
        CK(cudaMemcpy(yw.data(), d_yw, yw.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            qwen4exp::moe_hits(d_wptr + size_t(t) * K, d_wn + t, K, d_xw + size_t(t) * n, n, ff, d_scr, d_y1, nullptr);
            CK(cudaMemcpy(y1.data(), d_y1, y1.size() * 4, cudaMemcpyDeviceToHost));
            for (int k = 0; k < win_n[t]; ++k)
                for (int r = 0; r < n; ++r)
                    worst = std::max(worst, double(std::fabs(yw[(size_t(t) * K + k) * n + r] - y1[size_t(k) * n + r])));
        }
        const bool ok = worst == 0.0;
        fail += !ok;
        std::printf("window of %d tokens, grouped by expert: max difference to one-token calls %.3g %s\n", T, worst, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
