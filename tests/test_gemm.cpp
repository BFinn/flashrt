// SPDX-License-Identifier: Apache-2.0
// gemm (MMQ / cuBLAS matrix-matrix) against gemv (MMVQ / MMVF mat-vec) on real weights from the
// model: one tensor of each type the model uses, T = 200 tokens of random activations. Both
// quantize the activations to 8 bits, but per 32 or per 128 values and in another order, so they
// agree to about 1e-3 relative; the tolerance is 1e-2. Also: gemm::moe on a real Q2_0 expert
// tensor against gemv::moe_q, and Q3R -> Q3_K (q3r::unpack) byte-exact against the original.
//
//   test_gemm MODEL.gguf
#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemm.h"
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/q3r.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <map>
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
void* upload(const Gguf& g, const GgufTensor& t) {
    std::vector<uint8_t> h(t.bytes);
    const int fd = open(g.shards[t.shard].c_str(), O_RDONLY);
    for (size_t r = 0; r < t.bytes;) {
        const ssize_t n = pread(fd, h.data() + r, t.bytes - r, off_t(t.file_offset + r));
        if (n <= 0) { std::fprintf(stderr, "read failed\n"); std::exit(2); }
        r += size_t(n);
    }
    close(fd);
    void* d = nullptr;
    CK(cudaMalloc(&d, t.bytes + gemv::kWeightTailPad));
    CK(cudaMemset(d, 0, t.bytes + gemv::kWeightTailPad));
    CK(cudaMemcpy(d, h.data(), t.bytes, cudaMemcpyHostToDevice));
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
    if (argc < 2) { std::fprintf(stderr, "usage: test_gemm MODEL.gguf\n"); return 2; }
    const Gguf g = Gguf::open(argv[1]);
    const int T = 200;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fail = 0;
    size_t ws_bytes = gemm::workspace_bytes(12288, T * 10) + (64 << 20);
    void* ws;
    CK(cudaMalloc(&ws, ws_bytes));
    void* q8;
    CK(cudaMalloc(&q8, gemv::q8_1_bytes(12288, 8)));

    // one 2-D tensor of each type (the first in file order)
    std::map<uint32_t, const GgufTensor*> pick;
    for (const GgufTensor& t : g.tensors)
        if (t.dims.size() == 2 && t.dims[1] > 1 && t.name != "token_embd.weight" && t.name != "per_layer_token_embd.weight" &&
            t.name != "output.weight" && gemm::supported(t.type) && !pick.count(t.type))
            pick[t.type] = &t;
    for (const auto& [type, t] : pick) {
        const int64_t K = t->dims[0], R = t->dims[1];
        void* W = upload(g, *t);
        std::vector<float> x(size_t(T) * K);
        for (float& v : x) v = nd(rng);
        float *dx, *y1, *y2;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y1, size_t(T) * R * 4));
        CK(cudaMalloc(&y2, size_t(T) * R * 4));
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int n = std::min(8, T - t0);
            gemv::matvec(type, W, dx + size_t(t0) * K, y1 + size_t(t0) * R, K, R, n, q8, nullptr);
        }
        gemm::gemm(type, W, dx, y2, K, R, T, ws, ws_bytes, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> a(size_t(T) * R), b(size_t(T) * R);
        CK(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
        const double e = rel(b, a);
        const bool ok = std::isfinite(e) && e < 1e-2;
        fail += !ok;
        std::printf("gemm %-7s %-28s %6lld x %6lld: relative error %.2e %s\n", ggml_type_name(type), t->name.c_str(), (long long)R,
                    (long long)K, e, ok ? "ok" : "FAIL");
        // Q3_K: the Q3R round trip must give the same bytes
        if (type == 11 && K % 256 == 0) {
            void *r, *back;
            CK(cudaMalloc(&r, q3r::bytes(R, K)));
            CK(cudaMalloc(&back, t->bytes));
            q3r::repack(W, r, R, K, nullptr);
            q3r::unpack(r, back, R, K, nullptr);
            CK(cudaDeviceSynchronize());
            std::vector<uint8_t> o(t->bytes), p(t->bytes);
            CK(cudaMemcpy(o.data(), W, t->bytes, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(p.data(), back, t->bytes, cudaMemcpyDeviceToHost));
            const bool same = o == p;
            fail += !same;
            std::printf("q3r round trip %s: %s\n", t->name.c_str(), same ? "identical, ok" : "DIFFERENT, FAIL");
            cudaFree(r);
            cudaFree(back);
        }
        cudaFree(W);
        cudaFree(dx);
        cudaFree(y1);
        cudaFree(y2);
    }

    // grouped experts: layer 0's gate (with up: fused SwiGLU is not in gemm, so gate alone) and down
    for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}) {
        const GgufTensor* t = g.tensor(name);
        if (!t) continue;
        const int64_t K = t->dims[0], R = t->dims[1], E = t->dims[2];
        const int k = 10, TT = 96;
        const bool per_slot = std::string(name).find("down") != std::string::npos;
        void* W = upload(g, *t);
        const int64_t stride = gemv::row_bytes(t->type, K) * R;
        std::vector<int32_t> ids(size_t(TT) * k);
        for (int i = 0; i < TT; ++i)
            for (int j = 0; j < k; ++j) {
                int e;
                bool dup;
                do {
                    e = int(rng() % 24);   // few experts, so they are shared across tokens
                    dup = false;
                    for (int jj = 0; jj < j; ++jj) dup |= ids[size_t(i) * k + jj] == e;
                } while (dup);
                ids[size_t(i) * k + j] = e;
            }
        (void)E;
        const size_t xrows = per_slot ? size_t(TT) * k : size_t(TT);
        std::vector<float> x(xrows * K);
        for (float& v : x) v = nd(rng);
        float *dx, *y1, *y2;
        int32_t* dids;
        void* xq;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dids, ids.size() * 4));
        CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y1, size_t(TT) * k * R * 4));
        CK(cudaMalloc(&y2, size_t(TT) * k * R * 4));
        CK(cudaMalloc(&xq, gemv::q8_1_bytes(K, 8 * k)));
        for (int t0 = 0; t0 < TT; t0 += 8) {
            const int n = std::min(8, TT - t0);
            gemv::quantize_q8_1(dx + (per_slot ? size_t(t0) * k : size_t(t0)) * K, K, per_slot ? n * k : n, xq, nullptr);
            gemv::moe_q(t->type, W, nullptr, xq, dids + size_t(t0) * k, y1 + size_t(t0) * k * R, n, k, K, R, stride, per_slot, nullptr);
        }
        gemm::moe(t->type, W, stride, int(E), dx, per_slot, dids, TT, k, y2, K, R, ws, ws_bytes, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> a(size_t(TT) * k * R), b(a.size());
        CK(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
        const double e = rel(b, a);
        const bool ok = std::isfinite(e) && e < 1e-2;
        fail += !ok;
        std::printf("moe  %-7s %-28s %lld experts, %d tokens x %d: relative error %.2e %s\n", ggml_type_name(t->type), name, (long long)E, TT,
                    k, e, ok ? "ok" : "FAIL");
        cudaFree(W);
        cudaFree(dx);
        cudaFree(dids);
        cudaFree(y1);
        cudaFree(y2);
        cudaFree(xq);
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
