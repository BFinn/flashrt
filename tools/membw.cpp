// SPDX-License-Identifier: Apache-2.0
// membw: host DRAM read bandwidth, the ceiling for CPU-computed expert misses.
//
//   membw [--threads N] [--gb G] [--pages 4k|thp|hugetlb] [--seconds S] [--no-pin]
//
// Each thread streams its own slice with 64-byte vector loads (AVX-512 when compiled for
// it, else AVX2), for S seconds. Prints GB/s over all threads. Use --threads 0 for one
// thread per physical core.
#include "core/platform.hpp"

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace flashrt;
using Clock = std::chrono::steady_clock;

namespace {

// Sum-of-XOR over [p, p+bytes): one pass, returns a value so the loads cannot be elided.
std::uint64_t read_pass(const char* p, std::size_t bytes) {
#if defined(__AVX512F__)
    __m512i acc0 = _mm512_setzero_si512(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (std::size_t i = 0; i + 256 <= bytes; i += 256) {
        acc0 = _mm512_xor_si512(acc0, _mm512_load_si512(p + i));
        acc1 = _mm512_xor_si512(acc1, _mm512_load_si512(p + i + 64));
        acc2 = _mm512_xor_si512(acc2, _mm512_load_si512(p + i + 128));
        acc3 = _mm512_xor_si512(acc3, _mm512_load_si512(p + i + 192));
    }
    acc0 = _mm512_xor_si512(_mm512_xor_si512(acc0, acc1), _mm512_xor_si512(acc2, acc3));
    return std::uint64_t(_mm512_reduce_add_epi64(acc0));
#elif defined(__AVX2__)
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    for (std::size_t i = 0; i + 64 <= bytes; i += 64) {
        acc0 = _mm256_xor_si256(acc0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i)));
        acc1 = _mm256_xor_si256(acc1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
    }
    acc0 = _mm256_xor_si256(acc0, acc1);
    alignas(32) std::uint64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), acc0);
    return v[0] ^ v[1] ^ v[2] ^ v[3];
#else   // portable (a build without -march=native): the compiler vectorises what it can
    std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const std::uint64_t* q = reinterpret_cast<const std::uint64_t*>(p);
    for (std::size_t i = 0; i + 32 <= bytes; i += 32, q += 4) {
        a0 ^= q[0];
        a1 ^= q[1];
        a2 ^= q[2];
        a3 ^= q[3];
    }
    return a0 ^ a1 ^ a2 ^ a3;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    int threads = 0;
    double gb = 8.0, seconds = 5.0;
    PageMode mode = PageMode::Default;
    bool pin = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--threads") threads = std::atoi(next());
        else if (a == "--gb") gb = std::atof(next());
        else if (a == "--seconds") seconds = std::atof(next());
        else if (a == "--pages") mode = parse_page_mode(next());
        else if (a == "--no-pin") pin = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    const std::vector<int> cpus = physical_cpus();
    if (threads <= 0) threads = int(cpus.size());

    HostBuffer buf = host_alloc(std::size_t(gb * double(1ull << 30)), mode, threads);
    if (!buf.ptr) {
        std::fprintf(stderr, "allocation of %.1f GiB with %s pages failed\n", gb, page_mode_name(mode));
        return 1;
    }

    const std::size_t slice = (buf.bytes / threads) & ~std::size_t(255);
    std::atomic<bool> go{false}, stop{false};
    std::vector<double> bytes_done(threads, 0.0);
    std::vector<std::uint64_t> sink(threads, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&, t] {
            if (pin && !cpus.empty()) pin_current_thread(cpus[t % cpus.size()]);
            const char* p = static_cast<const char*>(buf.ptr) + std::size_t(t) * slice;
            while (!go.load(std::memory_order_acquire)) {}
            double done = 0;
            std::uint64_t s = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                s ^= read_pass(p, slice);
                done += double(slice);
            }
            bytes_done[t] = done;
            sink[t] = s;
        });
    }

    const auto t0 = Clock::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : ts) t.join();
    const double dt = std::chrono::duration<double>(Clock::now() - t0).count();

    double total = 0;
    std::uint64_t x = 0;
    for (int t = 0; t < threads; ++t) { total += bytes_done[t]; x ^= sink[t]; }
    std::printf("membw pages=%s huge_frac=%.2f threads=%d buffer=%.1fGiB seconds=%.2f read=%.2f GB/s (sink %llx)\n",
                page_mode_name(mode), huge_page_fraction(buf), threads, double(buf.bytes) / double(1ull << 30),
                dt, total / dt / 1e9, (unsigned long long) (x & 0xff));
    host_free(buf);
    return 0;
}
