// SPDX-License-Identifier: Apache-2.0
// fr_bench: decode speed of flashrt at a given depth.
//
//   fr_bench MODEL.gguf --ids PROMPT.txt --n-prompt N --gen G [--slots S] [--reserve-mib R]
//            [--reference] [--workers W]
//
// Prefills N prompt tokens in 64-token batches (reference path; its routing counts pick the
// cache contents), fills the VRAM expert cache with the most-routed experts (S slots, or all
// free VRAM minus R MiB), then decodes G tokens greedily on the fast path and reports tok/s and
// the cache hit rate. --reference decodes on the reference path instead (no cache), to compare
// the generated tokens.
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fr_bench MODEL.gguf --ids PROMPT.txt --n-prompt N --gen G [--slots S] [--reserve-mib R] [--reference]\n");
        return 2;
    }
    std::string ids_path;
    int n_prompt = 1024, gen = 128, slots = 0, reserve_mib = 1024, workers = 8;
    bool reference = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ids") ids_path = next();
        else if (a == "--n-prompt") n_prompt = std::atoi(next());
        else if (a == "--gen") gen = std::atoi(next());
        else if (a == "--slots") slots = std::atoi(next());
        else if (a == "--reserve-mib") reserve_mib = std::atoi(next());
        else if (a == "--workers") workers = std::atoi(next());
        else if (a == "--reference") reference = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<int32_t> seq;
    {
        std::ifstream f(ids_path);
        long v;
        while (f >> v && int(seq.size()) < n_prompt) seq.push_back(int32_t(v));
    }
    if (int(seq.size()) < n_prompt) { std::fprintf(stderr, "prompt file has only %zu tokens\n", seq.size()); return 1; }

    const Gguf g = Gguf::open(argv[1]);
    const Spec s = parse(g);
    const WeightPlan plan = qwen4exp::plan(g, s);
    GpuWeights w;
    w.load(g, plan);
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    load_experts(g, s, arena, 12);
    CpuPool pool(workers, physical_cpus());
    ForwardRef fwd(g, s, w, arena, pool, n_prompt + gen + 16, 64);

    float* logits_dev = nullptr;
    cudaMalloc(&logits_dev, size_t(s.n_vocab) * 4);
    std::vector<float> logits(s.n_vocab);
    auto argmax = [&]() {
        cudaMemcpy(logits.data(), logits_dev, logits.size() * 4, cudaMemcpyDeviceToHost);
        return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };

    // prefill
    const auto tp = Clock::now();
    for (int p = 0; p < n_prompt; p += 64) {
        const int T = std::min(64, n_prompt - p);
        const bool last = p + T >= n_prompt;
        fwd.forward(seq.data(), T, last ? T - 1 : T, last ? logits_dev : nullptr);
    }
    const double prefill_s = std::chrono::duration<double>(Clock::now() - tp).count();
    std::printf("prefill: %d tokens in %.1f s (%.1f tok/s, reference path)\n", n_prompt, prefill_s, n_prompt / prefill_s);

    // expert cache from the prompt's routing counts
    ExpertCache cache;
    MoeFastHost host;
    if (!reference) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t eb = q2_0::expert_bytes({s.d_model, s.d_ff_expert});
        if (slots <= 0) slots = int((free_b - size_t(reserve_mib) * 1048576) / eb);
        std::vector<uint32_t>& cnt = fwd.counts();
        std::vector<int> idx(cnt.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        std::vector<std::pair<int, int>> order;
        for (int i : idx) order.push_back({i / s.n_expert, i % s.n_expert});
        const auto tf = Clock::now();
        cache = alloc_expert_cache(s, slots);
        expert_cache_fill(s, cache, arena, order, fwd.stream());
        std::printf("expert cache: %d slots (%.1f%% of experts), filled in %.1f s from %.0f MiB free\n", slots,
                    100.0 * slots / (s.n_layer * s.n_expert), std::chrono::duration<double>(Clock::now() - tf).count(),
                    free_b / 1048576.0);
        host = alloc_moe_fast_host(s);
        host.arena = &arena;
        host.pool = &pool;
        fwd.set_fast_moe(&cache, &host);
    }

    // decode
    std::vector<int32_t> out;
    seq.push_back(argmax());
    out.push_back(seq.back());
    cudaProfilerStart();   // nsys --capture-range=cudaProfilerApi profiles the decode loop only
    const auto td = Clock::now();
    for (int i = 0; i < gen; ++i) {
        fwd.forward(seq.data(), 1, 0, logits_dev);
        seq.push_back(argmax());
        out.push_back(seq.back());
    }
    const double dec_s = std::chrono::duration<double>(Clock::now() - td).count();
    cudaProfilerStop();
    std::printf("decode: %d tokens at depth %d in %.2f s: %.2f tok/s (%s)\n", gen, n_prompt, dec_s, gen / dec_s,
                reference ? "reference path" : "fast path");
    if (!reference)
        std::printf("expert cache hit rate %.2f%% (%ld hits, %ld misses)\n", 100.0 * host.hits / std::max(1L, host.hits + host.misses),
                    host.hits, host.misses);
    std::printf("tokens:");
    for (int i = 0; i < std::min<int>(24, int(out.size())); ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    if (!reference) {
        free_moe_fast_host(host);
        free_expert_cache(cache);
    }
    cudaFree(logits_dev);
    arena_free(arena);
    return 0;
}
