// SPDX-License-Identifier: Apache-2.0
#include "core/cpu_pool.hpp"

#include "core/platform.hpp"

#include <chrono>

#if defined(__x86_64__)
#include <immintrin.h>
#define FLASHRT_PAUSE() _mm_pause()
#else
#define FLASHRT_PAUSE() std::this_thread::yield()
#endif

namespace flashrt {

namespace {
using Clock = std::chrono::steady_clock;
}

CpuPool::CpuPool(int n_workers, std::vector<int> cpus, int spin_us)
    : n_(n_workers < 1 ? 1 : n_workers), spin_us_(spin_us), cpus_(std::move(cpus)) {
    if (!cpus_.empty()) pin_current_thread(cpus_[0]);
    // Workers start from the generation as of now: a thread that only gets scheduled after
    // the first run() must still see that run as new.
    const uint64_t start = gen_.load();
    for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i, start] { worker(i, start); });
}

CpuPool::~CpuPool() {
    quit_.store(true);
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
    for (auto& t : threads_) t.join();
}

void CpuPool::run(Fn fn, void* ctx) {
    fn_ = fn;
    ctx_ = ctx;
    done_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
    fn(ctx, 0, n_);
    while (done_.load(std::memory_order_acquire) != n_ - 1) FLASHRT_PAUSE();
}

void CpuPool::barrier() {
    const uint64_t g = bar_gen_.load(std::memory_order_acquire);
    if (bar_count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
        bar_count_.store(0, std::memory_order_relaxed);
        bar_gen_.fetch_add(1, std::memory_order_release);
    } else {
        while (bar_gen_.load(std::memory_order_acquire) == g) FLASHRT_PAUSE();
    }
}

void CpuPool::worker(int id, uint64_t start_gen) {
    if (!cpus_.empty()) pin_current_thread(cpus_[id % cpus_.size()]);
    uint64_t seen = start_gen;
    for (;;) {
        // spin, then sleep on the futex until the generation changes
        const auto t0 = Clock::now();
        uint64_t g;
        int k = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            FLASHRT_PAUSE();
            if (++k == 256) {
                k = 0;
                if (std::chrono::duration<double, std::micro>(Clock::now() - t0).count() > spin_us_.load(std::memory_order_relaxed)) {
                    gen_.wait(seen, std::memory_order_acquire);
                }
            }
        }
        seen = g;
        if (quit_.load()) return;
        fn_(ctx_, id, n_);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

}  // namespace flashrt
