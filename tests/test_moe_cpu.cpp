// SPDX-License-Identifier: Apache-2.0
// moe_cpu against a plain loop of expert_ffn calls: the weighted sum over misses per token,
// for pools of 1, 3 and 6 workers, a 3-token window and misses that cover some tokens only.
#include "core/cpu_pool.hpp"
#include "core/fp16.hpp"
#include "quant/q2_0/moe_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace flashrt;
using namespace flashrt::q2_0;

int main() {
    std::mt19937 rng(11);
    const ExpertShape s{2560, 640};
    const int n_exp = 6, window = 3;

    // random experts
    std::vector<std::vector<uint8_t>> blobs(n_exp, std::vector<uint8_t>(expert_bytes(s)));
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : blobs) {
        for (auto& v : b) v = uint8_t(byte(rng));
        const Expert e = expert_view(b.data(), s);
        for (const Mat* m : {&e.gate, &e.up, &e.down})
            for (size_t k = 0; k < size_t(m->rows) * m->nb(); ++k)
                const_cast<uint16_t*>(m->scales)[k] = fp32_to_fp16(0.005f + 0.01f * float(byte(rng)) / 255.0f);
    }

    // window activations
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<uint8_t>> amem(window);
    std::vector<Q8Act> x(window);
    for (int t = 0; t < window; ++t) {
        std::vector<float> v(s.d_model);
        for (auto& f : v) f = nd(rng);
        amem[t].resize(q8_bytes(s.d_model) + 64);
        auto p = (reinterpret_cast<uintptr_t>(amem[t].data()) + 63) & ~uintptr_t(63);
        x[t] = q8_view(reinterpret_cast<void*>(p), s.d_model);
        quantize_q8(v.data(), x[t]);
    }

    // misses: expert i routed to a subset of the window's tokens
    std::vector<Miss> miss;
    const int toks[6][3] = {{0, 1, 2}, {0, -1, -1}, {1, 2, -1}, {2, -1, -1}, {0, 2, -1}, {1, -1, -1}};
    for (int i = 0; i < n_exp; ++i) {
        Miss m{blobs[i].data(), 0, {}, {}};
        for (int j = 0; j < 3 && toks[i][j] >= 0; ++j) {
            m.tok[m.n_tok] = toks[i][j];
            m.w[m.n_tok] = 0.05f + 0.1f * float(i + j);
            ++m.n_tok;
        }
        miss.push_back(m);
    }

    // reference: expert_ffn per miss and token, weighted and summed
    std::vector<float> ref(size_t(window) * s.d_model, 0.0f), o(s.d_model);
    std::vector<uint8_t> scratch(expert_scratch_bytes(s) + 64);
    for (const Miss& m : miss) {
        const Expert e = expert_view(m.blob, s);
        for (int j = 0; j < m.n_tok; ++j) {
            expert_ffn(e, &x[m.tok[j]], 1, o.data(), s.d_model, scratch.data(), have_avx512());
            for (int r = 0; r < s.d_model; ++r) ref[size_t(m.tok[j]) * s.d_model + r] += m.w[j] * o[r];
        }
    }

    int fail = 0;
    for (int workers : {1, 3, 6}) {
        CpuPool pool(workers);
        std::vector<uint8_t> ms(moe_cpu_scratch_bytes(s, int(miss.size()), workers));
        std::vector<float> out(size_t(window) * s.d_model, -1.0f);
        for (int rep = 0; rep < 3; ++rep)   // repeated runs reuse the pool and scratch
            moe_cpu(pool, s, miss.data(), int(miss.size()), x.data(), window, out.data(), s.d_model, ms.data());
        double num = 0, den = 0;
        for (size_t k = 0; k < out.size(); ++k) {
            num += double(out[k] - ref[k]) * (out[k] - ref[k]);
            den += double(ref[k]) * ref[k];
        }
        const double rel = std::sqrt(num / den);
        const bool ok = rel < 1e-5;
        fail += !ok;
        std::printf("moe_cpu %d workers vs expert_ffn loop: rel L2 %.2e %s\n", workers, rel, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
