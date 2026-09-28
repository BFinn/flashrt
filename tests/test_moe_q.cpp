// SPDX-License-Identifier: Apache-2.0
// gemv::moe_q (grouped expert mat-vec over a back-to-back expert tensor) against gemv::matvec on
// each expert's slice: Q8_0 experts with random codes, 3 tokens x 4 experts of 8, the gate+up
// pass with fused SwiGLU (one activation per token) and the down pass (one activation per
// token and expert). Both paths quantize the activations to Q8_1 the same way, so they agree to
// float rounding; the tolerance is 1e-4 relative.
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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
constexpr uint32_t kQ8_0 = 8;
// random Q8_0 rows: blocks of an fp16 scale and 32 int8 codes
std::vector<uint8_t> random_q8_0(size_t rows, size_t cols, std::mt19937& rng) {
    std::vector<uint8_t> v(rows * cols / 32 * 34);
    std::uniform_real_distribution<float> ud(0.002f, 0.01f);
    for (size_t b = 0; b < v.size() / 34; ++b) {
        const __half_raw hr = __half(__float2half(ud(rng)));
        v[b * 34] = uint8_t(hr.x & 0xff);
        v[b * 34 + 1] = uint8_t(hr.x >> 8);
        for (int i = 0; i < 32; ++i) v[b * 34 + 2 + i] = uint8_t(rng());
    }
    return v;
}
double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}
}  // namespace

int main() {
    const int n = 2560, ff = 640, E = 8, K = 4, T = 3;
    std::mt19937 rng(11);
    const auto gate = random_q8_0(size_t(E) * ff, n, rng), up = random_q8_0(size_t(E) * ff, n, rng),
               down = random_q8_0(size_t(E) * n, ff, rng);
    const int32_t ids_h[T * K] = {3, 0, 7, 5, 1, 1, 6, 2, 4, 3, 0, 7};   // a repeated id on purpose
    std::vector<float> x_h(size_t(T) * n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x_h) v = nd(rng);

    void *dg, *du, *dd, *xq, *hq, *sc;
    float *dx, *hid, *y, *ref;
    int32_t* ids;
    const size_t pad = gemv::kWeightTailPad;
    CK(cudaMalloc(&dg, gate.size() + pad));
    CK(cudaMalloc(&du, up.size() + pad));
    CK(cudaMalloc(&dd, down.size() + pad));
    CK(cudaMemset(dg, 0, gate.size() + pad));
    CK(cudaMemset(du, 0, up.size() + pad));
    CK(cudaMemset(dd, 0, down.size() + pad));
    CK(cudaMemcpy(dg, gate.data(), gate.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du, up.data(), up.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, down.data(), down.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, x_h.size() * 4));
    CK(cudaMemcpy(dx, x_h.data(), x_h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ids, sizeof(ids_h)));
    CK(cudaMemcpy(ids, ids_h, sizeof(ids_h), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xq, gemv::q8_1_bytes(n, T)));
    CK(cudaMalloc(&hq, gemv::q8_1_bytes(ff, T * K)));
    CK(cudaMalloc(&sc, gemv::q8_1_bytes(n, 1)));
    CK(cudaMalloc(&hid, size_t(T) * K * ff * 4));
    CK(cudaMalloc(&y, size_t(T) * K * n * 4));
    CK(cudaMalloc(&ref, size_t(T) * K * n * 4));

    const int64_t gu_stride = gemv::row_bytes(kQ8_0, n) * ff, d_stride = gemv::row_bytes(kQ8_0, ff) * n;
    gemv::quantize_q8_1(dx, n, T, xq, nullptr);
    gemv::moe_q(kQ8_0, du, dg, xq, ids, hid, T, K, n, ff, gu_stride, false, nullptr);
    gemv::quantize_q8_1(hid, ff, T * K, hq, nullptr);
    gemv::moe_q(kQ8_0, dd, nullptr, hq, ids, y, T, K, ff, n, d_stride, true, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> hid_g(size_t(T) * K * ff), y_g(size_t(T) * K * n);
    CK(cudaMemcpy(hid_g.data(), hid, hid_g.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_g.data(), y, y_g.size() * 4, cudaMemcpyDeviceToHost));

    // reference: per (token, expert) matvec on the expert's slice, SwiGLU on the host
    std::vector<float> hid_r(hid_g.size()), gt(ff), ut(ff);
    float *dgt, *dut;
    CK(cudaMalloc(&dgt, ff * 4));
    CK(cudaMalloc(&dut, ff * 4));
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const int e = ids_h[t * K + k];
            gemv::matvec(kQ8_0, static_cast<uint8_t*>(dg) + e * gu_stride, dx + size_t(t) * n, dgt, n, ff, 1, sc, nullptr);
            gemv::matvec(kQ8_0, static_cast<uint8_t*>(du) + e * gu_stride, dx + size_t(t) * n, dut, n, ff, 1, sc, nullptr);
            CK(cudaMemcpy(gt.data(), dgt, ff * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ut.data(), dut, ff * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < ff; ++i) hid_r[(size_t(t) * K + k) * ff + i] = gt[i] / (1.0f + std::exp(-gt[i])) * ut[i];
        }
    // down from the GPU hidden rows, so this pass is checked on its own
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const int e = ids_h[t * K + k];
            gemv::matvec(kQ8_0, static_cast<uint8_t*>(dd) + e * d_stride, hid + (size_t(t) * K + k) * ff, ref + (size_t(t) * K + k) * n, ff,
                         n, 1, sc, nullptr);
        }
    CK(cudaDeviceSynchronize());
    std::vector<float> y_r(y_g.size());
    CK(cudaMemcpy(y_r.data(), ref, y_r.size() * 4, cudaMemcpyDeviceToHost));
    const double e1 = rel_err(hid_g, hid_r), e2 = rel_err(y_g, y_r);
    const bool ok = e1 < 1e-4 && e2 < 1e-4;
    std::printf("moe_q Q8_0: gate+up relative error %.2e, down %.2e: %s\n", e1, e2, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
