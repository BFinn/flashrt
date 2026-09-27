// SPDX-License-Identifier: Apache-2.0
// bench_moe_cpu: latency of one layer's CPU misses (moe_cpu on a CpuPool) from DRAM.
//
//   bench_moe_cpu [--experts E] [--workers 4,6,8,11] [--misses 1,2,4,8] [--tokens N] [--seconds S]
//
// Each call routes `misses` experts picked at random from an arena of E random experts
// (default 2,400, 3.3 GB, far above the L3) to N tokens, like one decode layer. Prints mean,
// p50 and p99 latency per call and the weight bytes per second it implies.
#include "core/cpu_pool.hpp"
#include "core/fp16.hpp"
#include "core/platform.hpp"
#include "quant/q2_0/moe_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace flashrt;
using namespace flashrt::q2_0;
using Clock = std::chrono::steady_clock;

static std::vector<int> ints(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(std::atoi(x.c_str()));
    return v;
}

int main(int argc, char** argv) {
    int n_exp = 2400, n_tok = 1;
    double seconds = 1.5;
    std::vector<int> workers{4, 6, 8, 11}, misses{1, 2, 4, 8};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--experts")) n_exp = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--workers")) workers = ints(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--misses")) misses = ints(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--tokens")) n_tok = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--seconds")) seconds = std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    const ExpertShape s{2560, 640};
    const size_t eb = expert_bytes(s), stride = (eb + 4095) & ~size_t(4095);
    HostBuffer arena = host_alloc(stride * n_exp, PageMode::THP, 12);
    {
        std::mt19937_64 rng(1);
        auto* p = static_cast<uint64_t*>(arena.ptr);
        for (size_t i = 0; i < stride * n_exp / 8; ++i) p[i] = rng();
        const uint16_t sc = fp32_to_fp16(0.01f);
        for (int x = 0; x < n_exp; ++x) {
            const Expert e = expert_view(static_cast<uint8_t*>(arena.ptr) + stride * x, s);
            for (const Mat* m : {&e.gate, &e.up, &e.down})
                for (size_t k = 0; k < size_t(m->rows) * m->nb(); ++k) const_cast<uint16_t*>(m->scales)[k] = sc;
        }
    }
    std::vector<std::vector<uint8_t>> amem(n_tok);
    std::vector<Q8Act> x(n_tok);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int t = 0; t < n_tok; ++t) {
        std::vector<float> v(s.d_model);
        for (auto& f : v) f = nd(rng);
        amem[t].resize(q8_bytes(s.d_model) + 64);
        auto p = (reinterpret_cast<uintptr_t>(amem[t].data()) + 63) & ~uintptr_t(63);
        x[t] = q8_view(reinterpret_cast<void*>(p), s.d_model);
        quantize_q8(v.data(), x[t]);
    }
    std::vector<float> out(size_t(n_tok) * s.d_model);
    const auto cpus = physical_cpus();
    std::printf("bench_moe_cpu: arena %.1f GB, %d token(s) per miss\n", stride * double(n_exp) / 1e9, n_tok);
    std::printf("%8s %7s %10s %9s %9s %9s\n", "workers", "misses", "calls", "mean us", "p50 us", "p99 us");
    for (int w : workers) {
        CpuPool pool(w, cpus);
        for (int k : misses) {
            std::vector<uint8_t> scratch(moe_cpu_scratch_bytes(s, k, w));
            std::vector<Miss> ms(k);
            std::vector<double> lat;
            std::uniform_int_distribution<int> pick(0, n_exp - 1);
            const auto t_end = Clock::now() + std::chrono::duration<double>(seconds);
            while (Clock::now() < t_end) {
                for (int i = 0; i < k; ++i) {
                    ms[i] = Miss{static_cast<uint8_t*>(arena.ptr) + stride * pick(rng), n_tok, {0, 1, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f}};
                }
                const auto a = Clock::now();
                moe_cpu(pool, s, ms.data(), k, x.data(), n_tok, out.data(), s.d_model, scratch.data());
                lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
            }
            std::sort(lat.begin(), lat.end());
            double mean = 0;
            for (double l : lat) mean += l;
            mean /= lat.size();
            std::printf("%8d %7d %10zu %9.1f %9.1f %9.1f   %.1f GB/s\n", w, k, lat.size(), mean, lat[lat.size() / 2],
                        lat[lat.size() * 99 / 100], k * double(eb) / (mean * 1e3));
        }
    }
    host_free(arena);
    return 0;
}
