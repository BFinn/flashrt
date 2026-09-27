// SPDX-License-Identifier: Apache-2.0
// CpuPool: a fixed set of pinned workers for short, latency-critical parallel phases (the CPU
// expert misses of one layer). The calling thread takes part as worker 0, so a pool of N
// runs N-1 extra threads. Idle workers spin for `spin_us`, then sleep on a futex, so a
// decode loop pays no wake-up latency while an idle engine does not burn cores.
#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace flashrt {

class CpuPool {
public:
    using Fn = void (*)(void* ctx, int worker, int n_workers);

    // cpus: logical CPUs to pin to (worker i -> cpus[i % size]); empty = no pinning.
    explicit CpuPool(int n_workers, std::vector<int> cpus = {}, int spin_us = 300);
    ~CpuPool();
    CpuPool(const CpuPool&) = delete;
    CpuPool& operator=(const CpuPool&) = delete;

    // Runs fn(ctx, w, size()) on every worker, the caller as w = 0; returns when all finish.
    void run(Fn fn, void* ctx);
    // A barrier for all workers of the current run.
    void barrier();
    int size() const { return n_; }
    // How long idle workers spin before they sleep (decode sets it above the layer interval).
    void set_spin_us(int us) { spin_us_.store(us, std::memory_order_relaxed); }

private:
    void worker(int id, uint64_t start_gen);

    int n_;
    std::atomic<int> spin_us_;
    std::vector<int> cpus_;
    std::vector<std::thread> threads_;
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<int> bar_count_{0};
    alignas(64) std::atomic<uint64_t> bar_gen_{0};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    std::atomic<bool> quit_{false};
};

}  // namespace flashrt
