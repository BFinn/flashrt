// SPDX-License-Identifier: Apache-2.0
// Q2_0 pack parity:
//   1. matvec_ref on the repacked layout == ggml's dot on ggml blocks, bit for bit
//   2. matvec_avx512 matches matvec_ref to float-reordering error, for 1-4 tokens
//   3. expert_ffn (AVX-512) matches expert_ffn (reference), and both track an fp32 SwiGLU
//      on dequantized weights within quantization error
#include "core/fp16.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::q2_0;

namespace {

int failures = 0;
void check(bool ok, const char* what, double v) {
    std::printf("%-60s %s (%.3g)\n", what, ok ? "ok" : "FAIL", v);
    if (!ok) ++failures;
}

std::vector<GgmlBlock> random_blocks(std::mt19937& rng, int rows, int cols) {
    std::vector<GgmlBlock> v(size_t(rows) * (cols / kBlock));
    std::normal_distribution<float> nd(0.0f, 0.01f);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : v) {
        b.d = fp32_to_fp16(std::fabs(nd(rng)) + 1e-4f);
        for (auto& q : b.qs) q = uint8_t(byte(rng));
    }
    return v;
}

std::vector<float> random_vec(std::mt19937& rng, int n, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

struct Act {
    std::vector<uint8_t> mem;
    Q8Act a;
    Act(const float* x, int n) : mem(q8_bytes(n) + 64) {
        auto p = (reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63);
        a = q8_view(reinterpret_cast<void*>(p), n);
        quantize_q8(x, a);
    }
};

float dequant(const GgmlBlock* blocks, int cols, int r, int j) {
    const GgmlBlock& b = blocks[size_t(r) * (cols / kBlock) + j / kBlock];
    const int e = j % kBlock;
    return float(((b.qs[e / 4] >> (2 * (e % 4))) & 3) - 1) * fp16_to_fp32(b.d);
}

}  // namespace

int main() {
    std::mt19937 rng(20260927);
    std::printf("AVX-512 VNNI kernel: %s\n", have_avx512() ? "available" : "NOT available (reference fallback)");

    // fp16 helpers against known values
    check(fp32_to_fp16(1.0f) == 0x3c00 && fp32_to_fp16(-2.0f) == 0xc000 && fp32_to_fp16(65504.0f) == 0x7bff &&
              fp16_to_fp32(0x3555) == 0.333251953125f && fp16_to_fp32(0x0001) == 5.960464477539063e-08f,
          "fp16 conversions", 0);

    const int d_model = 2560, d_ff = 640;
    for (auto [rows, cols] : {std::pair{96, d_model}, std::pair{160, d_ff}}) {
        auto blocks = random_blocks(rng, rows, cols);
        std::vector<uint8_t> packed(mat_bytes(rows, cols));
        const Mat w = repack(blocks.data(), rows, cols, packed.data());

        std::vector<Act> acts;
        for (int t = 0; t < 4; ++t) acts.emplace_back(random_vec(rng, cols, 1.0f).data(), cols);
        std::vector<Q8Act> av;
        for (auto& x : acts) av.push_back(x.a);

        // 1. reference on the repacked layout == ggml semantics on ggml blocks
        std::vector<float> yr(4 * rows), ya(4 * rows);
        matvec_ref(w, av.data(), 4, 0, rows, yr.data(), rows);
        double maxd = 0;
        for (int t = 0; t < 4; ++t)
            for (int r = 0; r < rows; ++r)
                maxd = std::max(maxd, double(std::fabs(yr[t * rows + r] -
                                                       dot_ggml(blocks.data() + size_t(r) * (cols / kBlock), av[t]))));
        char name[96];
        std::snprintf(name, sizeof name, "ref == ggml dot, %dx%d (max abs diff)", rows, cols);
        check(maxd == 0.0, name, maxd);

        // 2. AVX-512 vs reference for each token count
        for (int nt = 1; nt <= 4; ++nt) {
            std::fill(ya.begin(), ya.end(), 0.0f);
            matvec_avx512(w, av.data(), nt, 0, rows, ya.data(), rows);
            // max abs error over the RMS of the outputs: robust to outputs near zero
            double worst = 0, ss = 0;
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) {
                    ss += double(yr[t * rows + r]) * yr[t * rows + r];
                    worst = std::max(worst, double(std::fabs(ya[t * rows + r] - yr[t * rows + r])));
                }
            worst /= std::sqrt(ss / (nt * rows));
            std::snprintf(name, sizeof name, "avx512 vs ref, %dx%d, %d tok (max err / rms)", rows, cols, nt);
            check(worst < 1e-5, name, worst);
        }
        // a row sub-range must touch only its rows
        std::vector<float> ys(rows, -7.0f);
        matvec_avx512(w, av.data(), 1, 10, 20, ys.data(), rows);
        bool clean = ys[9] == -7.0f && ys[20] == -7.0f && std::fabs(ys[15] - yr[15]) <= 1e-4 * (std::fabs(yr[15]) + 1e-2);
        std::snprintf(name, sizeof name, "avx512 row range [10,20) of %dx%d", rows, cols);
        check(clean, name, 0);
    }

    // 3. whole expert
    const ExpertShape s{d_model, d_ff};
    auto g = random_blocks(rng, d_ff, d_model), u = random_blocks(rng, d_ff, d_model), dn = random_blocks(rng, d_model, d_ff);
    std::vector<uint8_t> blob(expert_bytes(s));
    const Expert e = repack_expert(g.data(), u.data(), dn.data(), s, blob.data());
    check(expert_bytes(s) == 1382400, "expert blob is 1,382,400 bytes", double(expert_bytes(s)));
    std::vector<uint8_t> scratch(expert_scratch_bytes(s));
    for (int nt : {1, 3}) {
        std::vector<std::vector<float>> xs;
        std::vector<Act> acts;
        for (int t = 0; t < nt; ++t) {
            xs.push_back(random_vec(rng, d_model, 1.0f));
            acts.emplace_back(xs.back().data(), d_model);
        }
        std::vector<Q8Act> av;
        for (auto& x : acts) av.push_back(x.a);
        std::vector<float> o_ref(nt * d_model), o_avx(nt * d_model);
        expert_ffn(e, av.data(), nt, o_ref.data(), d_model, scratch.data(), false);
        expert_ffn(e, av.data(), nt, o_avx.data(), d_model, scratch.data(), true);

        double num = 0, den = 0, num32 = 0;
        for (int t = 0; t < nt; ++t) {
            // fp32 SwiGLU on dequantized weights, unquantized activations
            std::vector<float> h(d_ff);
            for (int i = 0; i < d_ff; ++i) {
                double gs = 0, us = 0;
                for (int j = 0; j < d_model; ++j) {
                    gs += dequant(g.data(), d_model, i, j) * xs[t][j];
                    us += dequant(u.data(), d_model, i, j) * xs[t][j];
                }
                h[i] = float(gs / (1.0 + std::exp(-gs)) * us);
            }
            for (int r = 0; r < d_model; ++r) {
                double o = 0;
                for (int i = 0; i < d_ff; ++i) o += dequant(dn.data(), d_ff, r, i) * h[i];
                const double a = o_avx[t * d_model + r], rf = o_ref[t * d_model + r];
                num += (a - rf) * (a - rf);
                num32 += (a - o) * (a - o);
                den += o * o;
            }
        }
        char name[96];
        std::snprintf(name, sizeof name, "expert %d tok: avx512 vs ref (rel L2)", nt);
        check(std::sqrt(num / den) < 1e-4, name, std::sqrt(num / den));
        std::snprintf(name, sizeof name, "expert %d tok: avx512 vs fp32 SwiGLU (rel L2, q8 error)", nt);
        check(std::sqrt(num32 / den) < 0.05, name, std::sqrt(num32 / den));
    }

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
