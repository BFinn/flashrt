// SPDX-License-Identifier: Apache-2.0
// CpuPool stress: many runs separated by random idle gaps (so workers fall past the spin window
// into the futex sleep and wake late), most runs doing no work (as layers with no expert
// misses), some using the barrier. Each run's context carries its generation; every worker
// must execute every run exactly once with that run's context, and no worker may touch a
// context after run() returned.
#include "core/cpu_pool.hpp"
#include "core/platform.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace flashrt;

namespace {
struct Ctx {
    long gen;
    bool use_barrier;
    std::atomic<int> visits{0};
    std::atomic<int> after_barrier{0};
    std::atomic<bool> closed{false};   // set after run() returns
    std::atomic<int> bad{0};
    std::vector<long>* last_gen;       // per worker: the last generation it ran
    CpuPool* pool;
};

void work(void* p, int w, int n) {
    Ctx& c = *static_cast<Ctx*>(p);
    if (c.closed.load()) c.bad.fetch_add(1);
    if ((*c.last_gen)[w] != c.gen - 1) c.bad.fetch_add(1);   // skipped or repeated a generation
    (*c.last_gen)[w] = c.gen;
    c.visits.fetch_add(1);
    if (c.use_barrier) {
        c.pool->barrier();
        if (c.visits.load() != n) c.bad.fetch_add(1);   // everyone arrived before anyone left
        c.after_barrier.fetch_add(1);
    }
}
}  // namespace

int main() {
    const int workers = 8, runs = 20000;
    CpuPool pool(workers, physical_cpus(), 200);   // short spin window: workers sleep often
    std::vector<long> last_gen(workers, 0);
    std::mt19937 rng(3);
    long bad = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long g = 1; g <= runs; ++g) {
        Ctx c;
        c.gen = g;
        c.use_barrier = rng() % 4 == 0;
        c.last_gen = &last_gen;
        c.pool = &pool;
        pool.run(work, &c);
        c.closed.store(true);
        if (c.visits.load() != workers || (c.use_barrier && c.after_barrier.load() != workers)) ++bad;
        bad += c.bad.load();
        const int gap = int(rng() % 16);
        if (gap < 3) std::this_thread::sleep_for(std::chrono::microseconds(300 + 200 * gap));   // past the spin window
        else if (gap < 6) std::this_thread::sleep_for(std::chrono::microseconds(150));          // around it
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("cpu_pool: %d runs, %d workers, %.1f s, %ld violations: %s\n", runs, workers, s, bad, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
