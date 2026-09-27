// SPDX-License-Identifier: Apache-2.0
// ssdrand: random-read IOPS from one file, the ceiling for the PLE (n-gram table) path.
//
//   ssdrand FILE [--threads N] [--depth D] [--bytes B] [--seconds S]
//
// N*D blocking pread loops, each on its own O_DIRECT descriptor, read B bytes (default
// 4096) at uniformly random aligned offsets, so N*D reads are in flight. Prints IOPS,
// MB/s and latency percentiles. The page cache is bypassed.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ssdrand FILE [--threads N] [--depth D] [--bytes B] [--seconds S]\n");
        return 2;
    }
    const std::string path = argv[1];
    int threads = 8, depth = 8;
    size_t bytes = 4096;
    double seconds = 5.0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--threads") threads = std::atoi(next());
        else if (a == "--depth") depth = std::atoi(next());
        else if (a == "--bytes") bytes = std::strtoull(next(), nullptr, 10);
        else if (a == "--seconds") seconds = std::atof(next());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) { std::perror(path.c_str()); return 1; }
    const uint64_t blocks = uint64_t(st.st_size) / bytes;

    const int workers = threads * depth;          // one blocking pread loop per in-flight read
    std::atomic<bool> stop{false};
    std::vector<std::vector<float>> lat(workers);
    std::vector<std::thread> ts;
    for (int w = 0; w < workers; ++w) {
        ts.emplace_back([&, w] {
            const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) { std::perror("open O_DIRECT"); std::exit(1); }
            void* buf = nullptr;
            if (posix_memalign(&buf, 4096, bytes) != 0) std::exit(1);
            std::mt19937_64 rng(0x9e3779b97f4a7c15ull * (w + 1));
            std::uniform_int_distribution<uint64_t> pick(0, blocks - 1);
            auto& l = lat[w];
            l.reserve(1 << 16);
            while (!stop.load(std::memory_order_relaxed)) {
                const auto t0 = Clock::now();
                if (pread(fd, buf, bytes, off_t(pick(rng) * bytes)) != ssize_t(bytes)) { std::perror("pread"); std::exit(1); }
                l.push_back(std::chrono::duration<float, std::micro>(Clock::now() - t0).count());
            }
            free(buf);
            close(fd);
        });
    }
    const auto t0 = Clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : ts) t.join();
    const double dt = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<float> all;
    for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
    std::sort(all.begin(), all.end());
    auto pct = [&](double p) { return all.empty() ? 0.0f : all[size_t(p * (all.size() - 1))]; };
    std::printf("ssdrand qd=%d (threads %d x depth %d) bytes=%zu: %.0f IOPS, %.1f MB/s, p50 %.0f us, p99 %.0f us\n",
                workers, threads, depth, bytes, all.size() / dt, all.size() * double(bytes) / dt / 1e6, pct(0.5), pct(0.99));
    return 0;
}
