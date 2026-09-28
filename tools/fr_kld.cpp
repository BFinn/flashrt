// SPDX-License-Identifier: Apache-2.0
// fr_kld: KL divergence of flashrt's logits against a llama-perplexity --kl-divergence-base
// file (the P1 correctness gate), on the same tokens and chunks.
//
//   fr_kld MODEL.gguf BASE.bin [--ctx N] [--chunks K] [--batch B] [--fast] [--reserve-mib R] [--static-cache]
//          [--pcie-frac F] [--kv q8] [--kv-hot BLOCKS] [--window W]
//
// The base file holds: the magic "_logits_", int32 ctx, int32 n_vocab, int32 n_chunk, the
// tokens of all chunks (n_chunk * ctx int32), then for every chunk the scored positions ctx/2 .. ctx-2, each as a float scale and a
// float minimum log-prob (4 uint16) and n_vocab uint16 log-probs (padded to even). Each chunk
// is run from a fresh state; KLD, same-top-1 and PPL follow llama-perplexity's formulas
// (reference probabilities below e^-16 are ignored).
//
// --fast scores the decode path instead: the unscored first half of each chunk is prefilled in
// batches (reference path), and the scored half runs one token at a time on the fast path
// (VRAM expert cache filled from chunk 0's prefill routing counts, GPU routing, doorbells),
// fed the chunk's own tokens. The cache adapts during decode unless --static-cache.
//
// --window W (with --fast) scores speculative verify windows instead: each step runs W tokens
// in one window, of which only the first j (random, 1..W) are the chunk's and the rest random
// tokens (rejected drafts), scores the j real rows and commits them, so every rewind path
// (GDN state, conv and PLE histories, the KV caches and indexer ring) is exercised.
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // keep the log if the process dies
    if (argc < 3) {
        std::fprintf(stderr, "usage: fr_kld MODEL.gguf BASE.bin --ctx N [--chunks K] [--batch B]\n");
        return 2;
    }
    int ctx = 0, chunks = 0, batch = 64, reserve_mib = 1024;
    bool fast = false, adaptive = true;
    float pcie_frac = 0.0f;
    bool kv_q8 = false;
    int kv_hot = 0, window = 0;
    for (int i = 3; i < argc; ++i) {
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : "0"; };
        if (!std::strcmp(argv[i], "--ctx")) ctx = std::atoi(next());
        else if (!std::strcmp(argv[i], "--chunks")) chunks = std::atoi(next());
        else if (!std::strcmp(argv[i], "--batch")) batch = std::atoi(next());
        else if (!std::strcmp(argv[i], "--fast")) fast = true;
        else if (!std::strcmp(argv[i], "--reserve-mib")) reserve_mib = std::atoi(next());
        else if (!std::strcmp(argv[i], "--static-cache")) adaptive = false;
        else if (!std::strcmp(argv[i], "--pcie-frac")) pcie_frac = float(std::atof(next()));
        else if (!std::strcmp(argv[i], "--kv")) kv_q8 = !std::strcmp(next(), "q8");
        else if (!std::strcmp(argv[i], "--kv-hot")) kv_hot = std::atoi(next());
        else if (!std::strcmp(argv[i], "--window")) window = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    FILE* bf = std::fopen(argv[2], "rb");
    if (!bf) { std::perror(argv[2]); return 1; }
    char magic[8];
    int32_t file_ctx = 0, n_vocab = 0, n_chunk = 0;
    if (std::fread(magic, 1, 8, bf) != 8 || std::memcmp(magic, "_logits_", 8) != 0 || std::fread(&file_ctx, 4, 1, bf) != 1 ||
        std::fread(&n_vocab, 4, 1, bf) != 1 || std::fread(&n_chunk, 4, 1, bf) != 1) {
        std::fprintf(stderr, "not a llama-perplexity KL base file\n");
        return 1;
    }
    if (ctx == 0) ctx = file_ctx;
    if (ctx != file_ctx) { std::fprintf(stderr, "--ctx %d does not match the base file's %d\n", ctx, file_ctx); return 1; }
    std::vector<int32_t> tokens(size_t(n_chunk) * ctx);
    if (std::fread(tokens.data(), 4, tokens.size(), bf) != tokens.size()) { std::fprintf(stderr, "base file too short\n"); return 1; }
    if (chunks <= 0 || chunks > n_chunk) chunks = n_chunk;
    const int first = ctx / 2, n_scored = ctx - 1 - first;
    const size_t nv = size_t(2 * ((n_vocab + 1) / 2) + 4);
    const long data0 = std::ftell(bf);

    const Gguf g = Gguf::open(argv[1]);
    const Spec s = parse(g);
    if (s.n_vocab != n_vocab) { std::fprintf(stderr, "vocab mismatch: model %d, base %d\n", s.n_vocab, n_vocab); return 1; }
    const WeightPlan plan = qwen4exp::plan(g, s);
    GpuWeights w;
    w.load(g, plan);
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    load_experts(g, s, arena, 12);
    const std::vector<int> cpus = physical_cpus();
    CpuPool pool(8, cpus);
    if (window > 0 && (!fast || window > 8 || window > batch)) { std::fprintf(stderr, "--window needs --fast and 1..8\n"); return 2; }
    ForwardRef fwd(g, s, w, arena, pool, ctx + 8, batch, kv_q8 || kv_hot > 0, kv_hot);
    if (window > 0) fwd.enable_windows(window);
    std::printf("KV cache: %s, hot set %d blocks per layer\n", kv_q8 || kv_hot > 0 ? "q8_0" : "fp16", kv_hot);

    float* logits_dev = nullptr;
    cudaMalloc(&logits_dev, size_t(batch) * n_vocab * 4);
    std::vector<float> logits(size_t(batch) * n_vocab);
    std::vector<uint16_t> base(nv);
    std::vector<double> klds;
    double sum_nll = 0, sum_nll_base = 0;
    long same_top = 0, count = 0;
    const auto t0 = std::chrono::steady_clock::now();
    ExpertCache cache;
    MoeFastHost host;
    CacheManager* mgr = nullptr;
    std::mt19937 rng(1234);
    long window_steps = 0, window_rewinds = 0;

    for (int ch = 0; ch < chunks; ++ch) {
        const int32_t* seq = tokens.data() + size_t(ch) * ctx;
        fwd.reset();
        // steps (p0, T): batches, or in --fast mode batches up to `first` and then single tokens
        std::vector<std::pair<int, int>> steps;
        for (int p0 = 0; p0 < (fast ? first : ctx); p0 += batch) steps.push_back({p0, std::min(batch, (fast ? first : ctx) - p0)});
        if (fast && window == 0)
            for (int p0 = first; p0 < ctx - 1; ++p0) steps.push_back({p0, 1});
        if (fast && window > 0)   // windows: (p0, -j), j real tokens, the step's length is `window`
            for (int p0 = first; p0 < ctx - 1;) {
                const int j = std::min<int>(1 + int(rng() % unsigned(window)), ctx - 1 - p0);
                steps.push_back({p0, -j});
                p0 += j;
            }
        std::vector<int32_t> seqw;
        for (size_t si = 0; si < steps.size(); ++si) {
            const int p0 = steps[si].first, j_real = steps[si].second < 0 ? -steps[si].second : 0;
            const int T = j_real ? j_real : steps[si].second;
            if (fast && p0 == first && !host.doorbell) {   // chunk 0's prefill is done: fill the cache, start the fast path
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                const size_t eb = q2_0::expert_bytes({s.d_model, s.d_ff_expert});
                const int slots = int((free_b - size_t(reserve_mib) * 1048576) / eb);
                std::vector<uint32_t>& cnt = fwd.counts();
                std::vector<int> idx(cnt.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
                std::vector<std::pair<int, int>> order;
                for (int i : idx) order.push_back({i / s.n_expert, i % s.n_expert});
                cache = alloc_expert_cache(s, slots);
                expert_cache_fill(s, cache, arena, order, fwd.stream());
                host = alloc_moe_fast_host(s, std::max(1, window));
                host.arena = &arena;
                host.pool = &pool;
                start_doorbell(s, host, cpus[0]);
                pin_current_thread(cpus[8 % cpus.size()]);
                pool.set_spin_us(2000);
                if (pcie_frac > 0) enable_pcie_misses(host, arena, pcie_frac, 4);
                fwd.set_fast_moe(&cache, &host);
                if (adaptive) {
                    mgr = create_cache_manager(s, cache, arena, CachePolicyConfig{}, fwd.counts());
                    fwd.set_cache_manager(mgr);
                }
                std::printf("fast path: %d cache slots (%.1f%% of experts), %s cache\n", slots, 100.0 * slots / (s.n_layer * s.n_expert),
                            adaptive ? "adaptive" : "static");
            }
            // rows whose logits are scored: positions first .. ctx-2
            const int lo = std::max(first, p0), hi = std::min(ctx - 2, p0 + T - 1);
            const int out_from = lo <= hi ? lo - p0 : T;
            if (j_real) {   // a window of `window` tokens: j_real real ones, then random ones
                seqw.assign(seq, seq + p0 + j_real);
                for (int k = j_real; k < window; ++k) seqw.push_back(int32_t(rng() % unsigned(n_vocab)));
                fwd.forward_window(seqw.data(), window, logits_dev);
                fwd.commit(j_real);
                ++window_steps;
                window_rewinds += j_real < window;
            } else {
                fwd.forward(seq, T, out_from, lo <= hi ? logits_dev : nullptr);
            }
            if (lo > hi) continue;
            const int R = T - out_from;
            cudaMemcpy(logits.data(), logits_dev, size_t(R) * n_vocab * 4, cudaMemcpyDeviceToHost);
            for (int p = lo; p <= hi; ++p) {
                const float* lg = logits.data() + size_t(p - lo) * n_vocab;
                const long rec = long(ch) * n_scored + (p - first);
                std::fseek(bf, data0 + rec * long(nv * 2), SEEK_SET);
                if (std::fread(base.data(), 2, nv, bf) != nv) { std::fprintf(stderr, "base record %ld missing\n", rec); return 1; }
                float scale, min_lp;
                std::memcpy(&scale, &base[0], 4);
                std::memcpy(&min_lp, &base[2], 4);
                const uint16_t* bl = base.data() + 4;
                // ours: log-softmax
                float mx = lg[0];
                int imax = 0;
                for (int v = 1; v < n_vocab; ++v)
                    if (lg[v] > mx) { mx = lg[v]; imax = v; }
                double se = 0;
                for (int v = 0; v < n_vocab; ++v) se += std::exp(double(lg[v] - mx));
                const double lse = std::log(se) + mx;
                const int tok = seq[p + 1];
                sum_nll += lse - lg[tok];
                sum_nll_base += -(scale * bl[tok] + min_lp);
                double kl = 0;
                int imax_b = 0;
                float lp_max = -INFINITY;
                for (int v = 0; v < n_vocab; ++v) {
                    const float lpb = scale * bl[v] + min_lp;
                    if (lpb > lp_max) { lp_max = lpb; imax_b = v; }
                    if (lpb > -16.0f) kl += std::exp(double(lpb)) * (lpb - (lg[v] - lse));
                }
                klds.push_back(kl);
                same_top += imax == imax_b;
                ++count;
            }
            if ((fast && (T == 1 || j_real)) ? ((p0 + T) / 1024 != p0 / 1024 || p0 + T + 1 >= ctx) : ((p0 / batch) % 16 == 15 || p0 + T >= ctx)) {
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double m = 0;
                for (double k : klds) m += k;
                std::printf("chunk %d pos %5d: %ld scored, mean KLD %.5f, same top %.2f%%, %.1f tok/s\n", ch, p0 + T, count,
                            count ? m / count : 0.0, count ? 100.0 * same_top / count : 0.0,
                            (double(ch) * ctx + p0 + T) / el);
                std::fflush(stdout);
            }
        }
    }
    std::vector<double> sorted = klds;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0;
    for (double k : klds) mean += k;
    mean /= std::max<size_t>(1, klds.size());
    auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted[size_t(q * (sorted.size() - 1))]; };
    std::printf("\nfr_kld: %ld tokens over %d chunk(s) of %d\n", count, chunks, ctx);
    std::printf("  PPL flashrt %.4f, base %.4f, ratio %.5f\n", std::exp(sum_nll / count), std::exp(sum_nll_base / count),
                std::exp((sum_nll - sum_nll_base) / count));
    std::printf("  KLD mean %.6f, median %.6f, p99 %.6f, p99.9 %.6f, max %.6f\n", mean, pct(0.5), pct(0.99), pct(0.999),
                sorted.empty() ? 0.0 : sorted.back());
    std::printf("  same top token %.3f%%\n", 100.0 * same_top / std::max(1L, count));
    if (fast) {
        std::printf("  fast path: expert cache hit rate %.2f%%, %ld misses read over PCIe", 100.0 * host.hits /
                    std::max(1L, host.hits + host.misses + host.gpu_misses), host.gpu_misses);
        if (mgr) std::printf(", %ld swaps", cache_manager_stats(mgr).swaps);
        std::printf("\n");
        if (window > 0) std::printf("  windows of %d: %ld steps, %ld rewound\n", window, window_steps, window_rewinds);
        fwd.set_cache_manager(nullptr);
        destroy_cache_manager(mgr);
        free_moe_fast_host(host);
        free_expert_cache(cache);
    }
    std::fclose(bf);
    cudaFree(logits_dev);
    arena_free(arena);
    return 0;
}
