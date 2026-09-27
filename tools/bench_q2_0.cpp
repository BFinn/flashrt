// SPDX-License-Identifier: Apache-2.0
// bench_q2_0: CPU expert throughput from DRAM, the rate of the CPU miss path.
//
//   bench_q2_0 [--experts E] [--threads T] [--tokens N] [--seconds S] [--pages 4k|thp] [--ref]
//
// Fills an arena of E repacked Q2_0 experts (1.38 MB each; the default 2,400 is 3.3 GB,
// far above the L3), then T pinned threads each run whole experts (gate, up, SwiGLU, down)
// for N tokens on randomly chosen experts. Prints experts/s and GB/s of weights read.
// Each expert here runs on one thread; splitting one expert's rows across the pool is
// the engine's job.
#include "core/fp16.hpp"
#include "core/platform.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace flashrt;
using namespace flashrt::q2_0;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    int n_exp = 2400, threads = 1, n_tok = 1;
    double seconds = 3.0;
    PageMode mode = PageMode::THP;
    bool ref = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--experts") n_exp = std::atoi(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--tokens") n_tok = std::atoi(next());
        else if (a == "--seconds") seconds = std::atof(next());
        else if (a == "--pages") mode = parse_page_mode(next());
        else if (a == "--ref") ref = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!ref && !have_avx512()) { std::fprintf(stderr, "no AVX-512 VNNI on this CPU\n"); return 1; }
    const ExpertShape s{2560, 640};
    const size_t eb = expert_bytes(s);

    // random codes with small positive scales (the scales are the last 2/18 of each matrix)
    HostBuffer arena = host_alloc(eb * n_exp, mode, 12);
    if (!arena.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    {
        std::mt19937_64 rng(1);
        auto* p = static_cast<uint64_t*>(arena.ptr);
        for (size_t i = 0; i < eb * n_exp / 8; ++i) p[i] = rng();
        const uint16_t sc = fp32_to_fp16(0.01f);
        for (int x = 0; x < n_exp; ++x) {
            const Expert e = expert_view(static_cast<char*>(arena.ptr) + eb * x, s);
            for (const Mat* m : {&e.gate, &e.up, &e.down}) {
                auto* scales = const_cast<uint16_t*>(m->scales);
                for (size_t k = 0; k < size_t(m->rows) * m->nb(); ++k) scales[k] = sc;
            }
        }
    }

    const auto cpus = physical_cpus();
    std::atomic<bool> go{false}, stop{false};
    std::vector<long> done(threads, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&, t] {
            if (!cpus.empty()) pin_current_thread(cpus[t % cpus.size()]);
            std::vector<uint8_t> scratch(expert_scratch_bytes(s) + 64);
            std::vector<std::vector<uint8_t>> amem(n_tok);
            std::vector<Q8Act> acts(n_tok);
            std::mt19937 rng(100 + t);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> x(s.d_model), out(size_t(n_tok) * s.d_model);
            for (int k = 0; k < n_tok; ++k) {
                amem[k].resize(q8_bytes(s.d_model) + 64);
                auto p = (reinterpret_cast<uintptr_t>(amem[k].data()) + 63) & ~uintptr_t(63);
                acts[k] = q8_view(reinterpret_cast<void*>(p), s.d_model);
                for (auto& v : x) v = nd(rng);
                quantize_q8(x.data(), acts[k]);
            }
            std::uniform_int_distribution<int> pick(0, n_exp - 1);
            while (!go.load(std::memory_order_acquire)) {}
            long n = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const Expert e = expert_view(static_cast<char*>(arena.ptr) + eb * pick(rng), s);
                expert_ffn(e, acts.data(), n_tok, out.data(), s.d_model, scratch.data(), !ref);
                ++n;
            }
            done[t] = n;
        });
    }
    const auto t0 = Clock::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : ts) t.join();
    const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    long total = 0;
    for (long n : done) total += n;
    std::printf("bench_q2_0 %s threads=%d tokens=%d arena=%.1fGB pages=%s: %.0f experts/s, %.2f GB/s of weights, "
                "%.1f us per expert per thread\n",
                ref ? "ref" : "avx512", threads, n_tok, eb * double(n_exp) / 1e9, page_mode_name(mode), total / dt,
                total * double(eb) / dt / 1e9, threads * dt / total * 1e6);
    host_free(arena);
    return 0;
}
