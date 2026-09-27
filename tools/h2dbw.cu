// SPDX-License-Identifier: Apache-2.0
// h2dbw: host->device bandwidth for expert-shaped transfers.
//
//   h2dbw [--gb G] [--pages 4k|thp|hugetlb] [--seconds S] [--cpu-threads N]
//
// Measures, over a G-GiB host buffer:
//   copy-alloc   cudaMemcpyAsync from cudaHostAlloc'd memory
//   copy-reg     cudaMemcpyAsync from mmap'd memory registered with cudaHostRegister
//   zero-copy    a kernel reading registered host memory directly (mapped pointer)
// each at chunk sizes of 1.38 MB (one Q2_0 expert), 16 MB and 64 MB.
// --cpu-threads N runs N DRAM-reading threads at the same time, to see how much of the
// link survives when the CPU is also computing expert misses.
#include "core/platform.hpp"

#include <cuda_runtime.h>

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

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

__global__ void zero_copy_read(const uint4* __restrict__ src, size_t n16, unsigned long long* out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) {
        const uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) atomicAdd(out, 1ull);   // keeps loads live
}

// Copy `chunk`-sized pieces from host to one device buffer, round-robin, for `seconds`.
double copy_rate(const char* host, size_t host_bytes, void* dev, size_t chunk, double seconds, cudaStream_t st) {
    const size_t n = host_bytes / chunk;
    size_t i = 0;
    double bytes = 0;
    const auto t0 = Clock::now();
    while (true) {
        for (int k = 0; k < 16; ++k, ++i) {   // 16 async copies per sync
            CK(cudaMemcpyAsync(dev, host + (i % n) * chunk, chunk, cudaMemcpyHostToDevice, st));
            bytes += double(chunk);
        }
        CK(cudaStreamSynchronize(st));
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= seconds) break;
    }
    return bytes / std::chrono::duration<double>(Clock::now() - t0).count() / 1e9;
}

double zero_copy_rate(const char* dev_mapped, size_t host_bytes, size_t chunk, double seconds, cudaStream_t st,
                      unsigned long long* d_out) {
    const size_t n = host_bytes / chunk;
    size_t i = 0;
    double bytes = 0;
    const auto t0 = Clock::now();
    while (true) {
        for (int k = 0; k < 16; ++k, ++i) {
            const uint4* src = reinterpret_cast<const uint4*>(dev_mapped + (i % n) * chunk);
            zero_copy_read<<<160, 256, 0, st>>>(src, chunk / 16, d_out);
            bytes += double(chunk);
        }
        CK(cudaStreamSynchronize(st));
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= seconds) break;
    }
    CK(cudaGetLastError());
    return bytes / std::chrono::duration<double>(Clock::now() - t0).count() / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    double gb = 4.0, seconds = 3.0;
    PageMode mode = PageMode::Default;
    int cpu_threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--gb") gb = std::atof(next());
        else if (a == "--seconds") seconds = std::atof(next());
        else if (a == "--pages") mode = parse_page_mode(next());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const size_t bytes = size_t(gb * double(1ull << 30)) & ~size_t((2u << 20) - 1);
    const size_t chunks[] = {1382400, 16u << 20, 64u << 20};   // one Q2_0 expert blob, 16 MB, 64 MB

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    void* dev = nullptr;
    CK(cudaMalloc(&dev, 64u << 20));
    unsigned long long* d_out = nullptr;
    CK(cudaMalloc(&d_out, sizeof(*d_out)));

    // optional concurrent DRAM readers, pinned to the physical cores
    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    HostBuffer rbuf;
    if (cpu_threads > 0) {
        rbuf = host_alloc(size_t(4) << 30, PageMode::THP, cpu_threads);
        const auto cpus = physical_cpus();
        const size_t slice = (rbuf.bytes / cpu_threads) & ~size_t(63);
        for (int t = 0; t < cpu_threads; ++t) {
            readers.emplace_back([&, t] {
                if (!cpus.empty()) pin_current_thread(cpus[t % cpus.size()]);
                const volatile uint64_t* p = reinterpret_cast<const uint64_t*>(static_cast<char*>(rbuf.ptr) + t * slice);
                uint64_t s = 0;
                while (!stop.load(std::memory_order_relaxed))
                    for (size_t i = 0; i < slice / 8; i += 8) s ^= p[i];   // one load per cache line
                if (s == 42) std::printf(" ");
            });
        }
    }

    // 1. cudaHostAlloc
    char* pinned = nullptr;
    CK(cudaHostAlloc(reinterpret_cast<void**>(&pinned), bytes, cudaHostAllocDefault));
    std::memset(pinned, 1, bytes);
    for (size_t c : chunks)
        std::printf("h2dbw copy-alloc   chunk=%8.2fMB  %6.2f GB/s\n", c / 1e6, copy_rate(pinned, bytes, dev, c, seconds, st));
    CK(cudaFreeHost(pinned));

    // 2./3. mmap (+ optional huge pages) registered and mapped
    HostBuffer hb = host_alloc(bytes, mode, 8);
    if (!hb.ptr) { std::fprintf(stderr, "host_alloc failed for %s pages\n", page_mode_name(mode)); return 1; }
    CK(cudaHostRegister(hb.ptr, hb.bytes, cudaHostRegisterMapped));
    char* mapped = nullptr;
    CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), hb.ptr, 0));
    std::printf("h2dbw registered buffer: pages=%s huge_frac=%.2f\n", page_mode_name(mode), huge_page_fraction(hb));
    for (size_t c : chunks)
        std::printf("h2dbw copy-reg     chunk=%8.2fMB  %6.2f GB/s\n", c / 1e6,
                    copy_rate(static_cast<const char*>(hb.ptr), hb.bytes, dev, c, seconds, st));
    for (size_t c : chunks)
        std::printf("h2dbw zero-copy    chunk=%8.2fMB  %6.2f GB/s\n", c / 1e6,
                    zero_copy_rate(mapped, hb.bytes, c, seconds, st, d_out));
    CK(cudaHostUnregister(hb.ptr));
    host_free(hb);

    stop.store(true);
    for (auto& t : readers) t.join();
    if (rbuf.ptr) host_free(rbuf);
    std::printf("h2dbw done (cpu_threads=%d)\n", cpu_threads);
    return 0;
}
