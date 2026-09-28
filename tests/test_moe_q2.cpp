// SPDX-License-Identifier: Apache-2.0
// moe_q2::run (int8 tensor cores on the arena's planar Q2_0 layout) against the MMQ path
// (gemm::moe on ggml's layout for gate and up, SwiGLU, gemm::moe for down), on layer 0's real
// experts with random activations and routing: 96 tokens over 24 experts, 1,200 and 8,192 over
// all 512. Both quantize the activations to 8 bits per 32, in different orders, so they agree to
// about 1e-2 relative (tolerance 3e-2). Also times both at 8,192 tokens.
//
//   test_moe_q2 MODEL.gguf
#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/moe_q2.h"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace flashrt;

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

namespace {
std::vector<uint8_t> read(const Gguf& g, const GgufTensor& t) {
    std::vector<uint8_t> h(t.bytes);
    const int fd = open(g.shards[t.shard].c_str(), O_RDONLY);
    for (size_t r = 0; r < t.bytes;) {
        const ssize_t n = pread(fd, h.data() + r, t.bytes - r, off_t(t.file_offset + r));
        if (n <= 0) { std::fprintf(stderr, "read failed\n"); std::exit(2); }
        r += size_t(n);
    }
    close(fd);
    return h;
}
void* upload(const void* h, size_t bytes, size_t pad) {
    void* d = nullptr;
    CK(cudaMalloc(&d, bytes + pad));
    CK(cudaMemset(d, 0, bytes + pad));
    CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    return d;
}
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_moe_q2 MODEL.gguf\n"); return 2; }
    const Gguf g = Gguf::open(argv[1]);
    const GgufTensor* tg = g.tensor("blk.0.ffn_gate_exps.weight");
    const GgufTensor* tu = g.tensor("blk.0.ffn_up_exps.weight");
    const GgufTensor* td = g.tensor("blk.0.ffn_down_exps.weight");
    if (!tg || !tu || !td) { std::fprintf(stderr, "expert tensors missing\n"); return 2; }
    const int n = int(tg->dims[0]), ff = int(tg->dims[1]), E = int(tg->dims[2]), K = 10;
    const std::vector<uint8_t> hg = read(g, *tg), hu = read(g, *tu), hd = read(g, *td);
    // the arena's layout: one planar blob per expert
    const q2_0::ExpertShape shape{n, ff};
    const size_t blob = q2_0::expert_bytes(shape), stride = (blob + 4095) / 4096 * 4096;
    const size_t gu_bytes = q2_0::mat_bytes(ff, n), d_bytes = q2_0::mat_bytes(n, ff);
    std::vector<uint8_t> planar(stride * E);
    for (int e = 0; e < E; ++e)
        q2_0::repack_expert(hg.data() + e * gu_bytes, hu.data() + e * gu_bytes, hd.data() + e * d_bytes, shape, planar.data() + e * stride);
    void* Wg = upload(hg.data(), hg.size(), gemv::kWeightTailPad);
    void* Wu = upload(hu.data(), hu.size(), gemv::kWeightTailPad);
    void* Wd = upload(hd.data(), hd.size(), gemv::kWeightTailPad);
    auto* Wp = static_cast<uint8_t*>(upload(planar.data(), planar.size(), 0));

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fail = 0;
    for (const int T : {96, 1200, 8192}) {
        const int n_used = T == 96 ? 24 : E;
        std::vector<int32_t> ids(size_t(T) * K);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                int e;
                bool dup;
                do {
                    e = int(rng() % n_used);
                    dup = false;
                    for (int kk = 0; kk < k; ++kk) dup |= ids[size_t(t) * K + kk] == e;
                } while (dup);
                ids[size_t(t) * K + k] = e;
            }
        std::vector<float> x(size_t(T) * n);
        for (float& v : x) v = nd(rng);
        float *dx, *dhg, *dhu, *dy1, *dy2;
        int32_t* dids;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dids, ids.size() * 4));
        CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        const size_t S = size_t(T) * K;
        CK(cudaMalloc(&dhg, S * ff * 4));
        CK(cudaMalloc(&dhu, S * ff * 4));
        CK(cudaMalloc(&dy1, S * n * 4));
        CK(cudaMalloc(&dy2, S * n * 4));
        const size_t ws1_bytes = gemm::workspace_bytes(n, int64_t(S)), ws2_bytes = moe_q2::workspace_bytes(T, K, n, ff, E);
        void *ws1, *ws2;
        CK(cudaMalloc(&ws1, ws1_bytes));
        CK(cudaMalloc(&ws2, ws2_bytes));
        const uint32_t kQ2 = 42;
        const int64_t sgu = int64_t(gu_bytes), sd = int64_t(d_bytes);
        auto gate_up = [&] {
            gemm::moe(kQ2, Wg, sgu, E, dx, false, dids, T, K, dhg, n, ff, ws1, ws1_bytes, nullptr);
            gemm::moe(kQ2, Wu, sgu, E, dx, false, dids, T, K, dhu, n, ff, ws1, ws1_bytes, nullptr);
        };
        // reference: MMQ gate and up, SwiGLU on the host, MMQ down
        gate_up();
        std::vector<float> a(S * ff), b(S * ff);
        CK(cudaMemcpy(a.data(), dhg, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dhu, b.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); ++i) a[i] = a[i] / (1.0f + std::exp(-a[i])) * b[i];
        CK(cudaMemcpy(dhg, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        gemm::moe(kQ2, Wd, sd, E, dhg, true, dids, T, K, dy1, ff, n, ws1, ws1_bytes, nullptr);
        moe_q2::run(Wp, stride, E, n, ff, dx, dids, T, K, dy2, ws2, ws2_bytes, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> y1(S * n), y2(S * n);
        CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost));
        const double err = rel(y2, y1);
        const bool ok = std::isfinite(err) && err < 3e-2;
        fail += !ok;
        std::printf("T %5d over %3d experts: relative error %.2e %s\n", T, n_used, err, ok ? "ok" : "FAIL");
        if (T == 8192) {   // timing: the three MMQ products (with their grouping and quantization) against run()
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            const int reps = 10;
            float ms_ref = 0, ms_new = 0;
            CK(cudaEventRecord(e0));
            for (int r = 0; r < reps; ++r) {
                gate_up();
                gemm::moe(kQ2, Wd, sd, E, dhg, true, dids, T, K, dy1, ff, n, ws1, ws1_bytes, nullptr);
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms_ref, e0, e1));
            CK(cudaEventRecord(e0));
            for (int r = 0; r < reps; ++r) moe_q2::run(Wp, stride, E, n, ff, dx, dids, T, K, dy2, ws2, ws2_bytes, nullptr);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms_new, e0, e1));
            const double gmac = double(S) * 3 * n * ff / 1e9;
            std::printf("T 8192: MMQ path %.2f ms (%.0f TOPS), moe_q2 %.2f ms (%.0f TOPS)\n", ms_ref / reps, 2 * gmac / ms_ref * reps,
                        ms_new / reps, 2 * gmac / ms_new * reps);
        }
        for (void* p : {static_cast<void*>(dx), static_cast<void*>(dids), static_cast<void*>(dhg), static_cast<void*>(dhu),
                        static_cast<void*>(dy1), static_cast<void*>(dy2), ws1, ws2})
            cudaFree(p);
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
