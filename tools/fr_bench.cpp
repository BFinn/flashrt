// SPDX-License-Identifier: Apache-2.0
// fr_bench: decode speed of flashrt at a given depth.
//
//   fr_bench MODEL.gguf --ids PROMPT.txt --n-prompt N --gen G [--slots S] [--reserve-mib R]
//            [--reference] [--workers W] [--no-doorbell] [--spin-us U] [--windows N] [--trace FILE]
//            [--static-cache] [--swap-budget B] [--pcie-frac F] [--save-state FILE | --load-state FILE] [--no-q3r]
//            [--no-graphs] [--kv q8] [--kv-hot BLOCKS] [--count-half-life N] [--mtp DRAFT.gguf [--draft K]]
//
// Prefills N prompt tokens in 64-token batches (reference path; its routing counts pick the
// cache contents), fills the VRAM expert cache with the most-routed experts (S slots, or all
// free VRAM minus R MiB), then decodes G tokens greedily on the fast path and reports tok/s and
// the cache hit rate. --reference decodes on the reference path instead (no cache), to compare
// the generated tokens. The fast path runs in doorbell mode (the whole token enqueued at once, a
// miss-server thread on the pool's first CPU) unless --no-doorbell. --windows N decodes N
// consecutive windows of G tokens after the one prefill and reports each (deep prompts).
// --trace writes the decode's routing, int16 [tokens][n_layer][top_k], for tools/cache_sim.py.
// The expert cache adapts during decode (decayed LFU, up to B uploads in flight, default 8)
// unless --static-cache (up to B uploads in flight, default 8). --pcie-frac F (default 0 = off; it did not help at 2K or 245K, sw15) lets the GPU read floor(F * misses)
// of each layer's misses straight from host memory, up to 4 per layer. --save-state writes the
// state after the prefill; --load-state restores it instead of prefilling (the prompt file is
// still read, for the n-gram context): decode at 250K without the 35-minute prefill.
// --mtp loads the MTP draft head (it runs over the prompt too, to fill its KV cache) and probes
// it: before each decode token, it drafts K tokens ahead (default 4), and the report gives the
// acceptance a greedy verifier would see (the drafts' matching prefix against the tokens the
// target then decodes). The decode tok/s includes the drafting.
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/mtp.hpp"
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
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // keep the log if the process dies
    if (argc < 2) {
        std::fprintf(stderr, "usage: fr_bench MODEL.gguf --ids PROMPT.txt --n-prompt N --gen G [--slots S] [--reserve-mib R] [--reference]\n");
        return 2;
    }
    std::string ids_path, trace_path, save_state, load_state, mtp_path;
    int draft_k = 4;
    int n_prompt = 1024, gen = 128, slots = 0, reserve_mib = 1024, workers = 8, windows = 1;
    bool reference = false, doorbell = true, adaptive = true;
    int swap_budget = 8;
    float pcie_frac = 0.0f;
    bool q3r = true, graphs = true, kv_q8 = false;
    int kv_hot = 0, half_life = 4096;
    int spin_us = 2000;
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
        else if (a == "--no-doorbell") doorbell = false;
        else if (a == "--spin-us") spin_us = std::atoi(next());
        else if (a == "--windows") windows = std::max(1, std::atoi(next()));
        else if (a == "--trace") trace_path = next();
        else if (a == "--static-cache") adaptive = false;
        else if (a == "--swap-budget") swap_budget = std::atoi(next());
        else if (a == "--pcie-frac") pcie_frac = float(std::atof(next()));
        else if (a == "--save-state") save_state = next();
        else if (a == "--no-q3r") q3r = false;
        else if (a == "--no-graphs") graphs = false;
        else if (a == "--kv") kv_q8 = std::string(next()) == "q8";
        else if (a == "--kv-hot") kv_hot = std::atoi(next());
        else if (a == "--count-half-life") half_life = std::atoi(next());
        else if (a == "--load-state") load_state = next();
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--draft") draft_k = std::max(1, std::atoi(next()));
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
    w.load(g, plan, q3r);
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    load_experts(g, s, arena, 12);
    const std::vector<int> cpus = physical_cpus();
    CpuPool pool(workers, cpus);   // pins this thread to cpus[0]
    ForwardRef fwd(g, s, w, arena, pool, n_prompt + windows * gen + 16, 64, kv_q8 || kv_hot > 0, kv_hot);
    fwd.set_count_half_life(half_life);
    std::printf("KV cache: %s%s\n", kv_q8 || kv_hot > 0 ? "q8_0" : "fp16",
                kv_hot > 0 ? (", host-resident, hot set of " + std::to_string(kv_hot) + " blocks per layer").c_str() : "");

    float* logits_dev = nullptr;
    cudaMalloc(&logits_dev, size_t(s.n_vocab) * 4);
    auto argmax = [&]() { return fwd.argmax(logits_dev); };

    // MTP draft head: it runs over every prompt batch after the target, taking the target's
    // streams shifted by one position (h_{p-1} with x_p)
    std::unique_ptr<Gguf> g_mtp;
    std::unique_ptr<MtpHead> mtp;
    const size_t hrow = size_t(s.hc_count) * s.d_model;
    float *h_carry = nullptr, *h_buf = nullptr, *mtp_logits = nullptr;
    if (!mtp_path.empty()) {
        g_mtp = std::make_unique<Gguf>(Gguf::open(mtp_path));
        mtp = std::make_unique<MtpHead>(*g_mtp, s, w, fwd.stream(), n_prompt + windows * gen + 16 + draft_k, 64, kv_q8 || kv_hot > 0, kv_hot);
        cudaMalloc(&h_carry, hrow * 4);
        cudaMemset(h_carry, 0, hrow * 4);
        cudaMalloc(&h_buf, 64 * hrow * 4);
        cudaMalloc(&mtp_logits, size_t(s.n_vocab) * 4);
        std::printf("MTP draft head: layer %d, %.0f MiB of weights in VRAM, %d drafts per token (probe)\n", mtp->layer(),
                    mtp->weight_bytes() / 1048576.0, draft_k);
    }
    auto mtp_catchup = [&](int p0, int T) {
        if (!mtp) return;
        cudaStream_t st = fwd.stream();
        cudaMemcpyAsync(h_buf, h_carry, hrow * 4, cudaMemcpyDeviceToDevice, st);
        if (T > 1) cudaMemcpyAsync(h_buf + hrow, fwd.streams(), size_t(T - 1) * hrow * 4, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(h_carry, fwd.streams() + size_t(T - 1) * hrow, hrow * 4, cudaMemcpyDeviceToDevice, st);
        mtp->forward(h_buf, seq.data() + p0, T, p0, T, nullptr);
    };

    // prefill
    const auto tp = Clock::now();
    if (!load_state.empty()) {
        // the state holds positions 0 .. n_prompt-2; the last prompt token runs now, for its logits
        fwd.load_state(load_state);
        if (fwd.pos() != n_prompt - 1) { std::fprintf(stderr, "state holds %d positions, expected %d\n", fwd.pos(), n_prompt - 1); return 1; }
        fwd.forward(seq.data(), 1, 0, logits_dev);
        if (mtp) std::printf("state: warning: the MTP head has no KV for the loaded positions\n");
        mtp_catchup(n_prompt - 1, 1);
        std::printf("state: loaded %s (%d positions) in %.1f s\n", load_state.c_str(), n_prompt - 1,
                    std::chrono::duration<double>(Clock::now() - tp).count());
    } else {
        for (int p = 0; p < n_prompt; p += 64) {
            const int T = std::min(64, n_prompt - p);
            const bool last = p + T >= n_prompt;
            if (last && !save_state.empty()) {   // save before the last token, so a load can re-run it
                if (T > 1) fwd.forward(seq.data(), T - 1, T - 1, nullptr);
                if (T > 1) mtp_catchup(p, T - 1);
                fwd.save_state(save_state);
                std::printf("state: saved %s (%d positions)\n", save_state.c_str(), fwd.pos());
                fwd.forward(seq.data(), 1, 0, logits_dev);
                mtp_catchup(p + T - 1, 1);
                break;
            }
            fwd.forward(seq.data(), T, last ? T - 1 : T, last ? logits_dev : nullptr);
            mtp_catchup(p, T);
        }
        const double prefill_s = std::chrono::duration<double>(Clock::now() - tp).count();
        std::printf("prefill: %d tokens in %.1f s (%.1f tok/s, reference path)\n", n_prompt, prefill_s, n_prompt / prefill_s);
    }

    // expert cache from the prompt's routing counts
    ExpertCache cache;
    MoeFastHost host;
    CacheManager* mgr = nullptr;
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
        if (doorbell) {   // the miss server takes the pool's caller CPU; this thread moves off it
            start_doorbell(s, host, cpus[0]);
            pin_current_thread(cpus[size_t(workers) % cpus.size()]);
        }
        pool.set_spin_us(spin_us);
        fwd.set_graphs(graphs);
        if (pcie_frac > 0) enable_pcie_misses(host, arena, pcie_frac, 4);
        fwd.set_fast_moe(&cache, &host);
        if (adaptive) {
            const auto tr = Clock::now();
            CachePolicyConfig cfg;
            cfg.budget = swap_budget;
            mgr = create_cache_manager(s, cache, arena, cfg, fwd.counts());
            fwd.set_cache_manager(mgr);
            std::printf("adaptive cache: decayed LFU, swap budget %d (arena registered in %.1f s)\n", swap_budget,
                        std::chrono::duration<double>(Clock::now() - tr).count());
        }
    }

    // decode
    std::vector<int32_t> out;
    seq.push_back(argmax());
    out.push_back(seq.back());
    std::vector<int16_t> trace;
    std::vector<int32_t> drafts;       // [token][draft_k]: the drafts made before each decode token
    std::vector<int> draft_pos;        // the position of the newest token when they were made
    double mtp_s = 0;
    cudaProfilerStart();   // nsys --capture-range=cudaProfilerApi profiles the decode loop only
    for (int wi = 0; wi < windows; ++wi) {
        const long hits0 = host.hits, misses0 = host.misses, gmiss0 = host.gpu_misses;
        const int depth = int(seq.size()) - 1;
        const auto td = Clock::now();
        for (int i = 0; i < gen; ++i) {
            if (mtp) {   // draft from (h_{p-1}, x_p): the first step is also position p's catch-up
                const auto tm = Clock::now();
                const int p = int(seq.size()) - 1;
                int32_t d = seq[p];
                draft_pos.push_back(p);
                for (int j = 0; j < draft_k; ++j) {
                    mtp->forward(j == 0 ? h_carry : mtp->h_out(), &d, 1, p + j, 0, mtp_logits);
                    d = fwd.argmax(mtp_logits);
                    drafts.push_back(d);
                }
                mtp_s += std::chrono::duration<double>(Clock::now() - tm).count();
            }
            fwd.forward(seq.data(), 1, 0, logits_dev);
            if (mtp) cudaMemcpyAsync(h_carry, fwd.streams(), hrow * 4, cudaMemcpyDeviceToDevice, fwd.stream());
            seq.push_back(argmax());
            out.push_back(seq.back());
            if (!reference && !trace_path.empty())
                for (int32_t e : host.access) trace.push_back(int16_t(e));
        }
        const double dec_s = std::chrono::duration<double>(Clock::now() - td).count();
        std::printf("decode: %d tokens at depth %d in %.2f s: %.2f tok/s (%s)", gen, depth, dec_s, gen / dec_s,
                    reference ? "reference path" : doorbell ? "fast path, doorbell" : "fast path, host sync per layer");
        if (windows > 1 && !reference)
            std::printf(", window %d, hit rate %.2f%%", wi + 1,
                        100.0 * (host.hits - hits0) /
                            std::max(1L, host.hits - hits0 + host.misses - misses0 + host.gpu_misses - gmiss0));
        if (mgr) std::printf(", swaps %ld", cache_manager_stats(mgr).swaps);
        std::printf("\n");
    }
    cudaProfilerStop();
    if (!reference)
        std::printf("expert cache hit rate %.2f%% (%ld hits, %ld misses: %ld on the CPU, %ld read over PCIe)\n",
                    100.0 * host.hits / std::max(1L, host.hits + host.misses + host.gpu_misses), host.hits,
                    host.misses + host.gpu_misses, host.misses, host.gpu_misses);
    if (!reference)
        std::printf("host per token: %.2f ms waiting for routing, %.2f ms running misses\n", 1e3 * host.wait_s / (gen * windows),
                    1e3 * host.cpu_s / (gen * windows));
    if (!reference) {
        std::printf("misses per layer: count (share of layers) and mean host miss time\n");
        long nl = 0;
        for (long v : host.layers_by_nm) nl += v;
        for (int m = 0; m <= 16; ++m)
            if (host.layers_by_nm[m])
                std::printf("  %2d%s: %5.1f%%  %6.1f us\n", m, m == 16 ? "+" : " ", 100.0 * host.layers_by_nm[m] / nl,
                            1e6 * host.cpu_by_nm[m] / host.layers_by_nm[m]);
    }
    if (!trace_path.empty() && !trace.empty()) {
        std::ofstream tf(trace_path, std::ios::binary);
        tf.write(reinterpret_cast<const char*>(trace.data()), std::streamsize(trace.size() * 2));
        std::printf("trace: %zu tokens x %d layers x %d to %s\n", trace.size() / (size_t(s.n_layer) * s.top_k), s.n_layer, s.top_k,
                    trace_path.c_str());
    }
    std::printf("CUDA graphs: %s (%ld capture(s))\n", graphs ? "on" : "off", fwd.graph_captures());
    if (mtp && !draft_pos.empty()) {   // greedy acceptance: the drafts' matching prefix
        std::vector<long> hist(draft_k + 1, 0);
        long n = 0;
        for (size_t i = 0; i < draft_pos.size(); ++i) {
            const int p = draft_pos[i];
            if (p + draft_k >= int(seq.size())) continue;
            int L = 0;
            while (L < draft_k && drafts[i * draft_k + L] == seq[p + 1 + L]) ++L;
            ++hist[L];
            ++n;
        }
        std::printf("MTP probe: %ld starts, %.2f ms per draft step (%d steps per token)\n", n,
                    1e3 * mtp_s / (double(draft_pos.size()) * draft_k), draft_k);
        std::printf("  accepted drafts L: ");
        for (int L = 0; L <= draft_k; ++L) std::printf(" %d:%.1f%%", L, 100.0 * hist[L] / std::max(1L, n));
        std::printf("\n  per-step acceptance (P(L > j | L >= j)):");
        long ge = n;
        for (int j = 0; j < draft_k; ++j) {
            const long gt = ge - hist[j];
            std::printf(" %.1f%%", 100.0 * gt / std::max(1L, ge));
            ge = gt;
        }
        std::printf("\n  tokens per verify round with k drafts (1 + E[min(L, k)]):");
        for (int k = 1; k <= draft_k; ++k) {
            double e = 0;
            for (int L = 0; L <= draft_k; ++L) e += double(std::min(L, k)) * hist[L];
            std::printf(" k=%d %.3f", k, 1.0 + e / std::max(1L, n));
        }
        std::printf("\n  first drafts vs target:");
        for (size_t i = 0; i < std::min<size_t>(6, draft_pos.size()); ++i) {
            std::printf(" [");
            for (int j = 0; j < draft_k; ++j) std::printf("%s%d", j ? " " : "", drafts[i * draft_k + j]);
            std::printf(" | ");
            for (int j = 0; j < draft_k && draft_pos[i] + 1 + j < int(seq.size()); ++j) std::printf("%s%d", j ? " " : "", seq[draft_pos[i] + 1 + j]);
            std::printf("]");
        }
        std::printf("\n");
    }
    std::printf("tokens:");
    for (int i = 0; i < std::min<int>(24, int(out.size())); ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    if (!reference) {
        fwd.set_cache_manager(nullptr);
        destroy_cache_manager(mgr);
        free_moe_fast_host(host);
        free_expert_cache(cache);
    }
    cudaFree(logits_dev);
    if (mtp) {
        mtp.reset();
        cudaFree(h_carry);
        cudaFree(h_buf);
        cudaFree(mtp_logits);
    }
    arena_free(arena);
    return 0;
}
