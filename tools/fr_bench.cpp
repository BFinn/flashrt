// SPDX-License-Identifier: Apache-2.0
// fr_bench: decode speed of flashrt at a given depth.
//
//   fr_bench MODEL.gguf --ids PROMPT.txt --n-prompt N --gen G [--slots S] [--reserve-mib R]
//            [--reference] [--workers W] [--no-doorbell] [--spin-us U] [--windows N] [--trace FILE]
//            [--static-cache] [--swap-budget B] [--pcie-frac F] [--save-state FILE | --load-state FILE] [--no-q3r]
//            [--no-graphs] [--kv q8] [--kv-hot BLOCKS] [--count-half-life N] [--mtp DRAFT.gguf [--draft K | --spec K]]
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
// target then decodes). The decode tok/s includes the drafting. --spec K decodes speculatively
// instead (greedy): each round the head drafts K tokens, the target verifies the window of K + 1
// in one step, and the matching prefix plus the target's next token are kept. The head's experts
// are Q2_0 (requantized at load; --mtp-bits 8 keeps the GGUF's Q8_0, 4 uses Q4_0). --draft-vocab RANKS (bench/mtp_vocab.py) trims
// the drafter's LM head to the top --draft-vocab-n ranked tokens (default 32768) plus the
// prompt's distinct tokens, at most 65536 rows. --draft-pmin P stops a round's drafting at the
// first draft whose probability under the head is below P (then fewer than K are verified;
// none if the first is below P).
// --prefill-chunk C prefills C tokens per call (default 64: the CPU reference path; above 64 the
// chunk path with the experts streamed to the GPU); "auto" picks the longest chunk the free VRAM
// holds (up to 16,384), as flashrt-engine does.
// --teacher decodes the ids file's own continuation instead of the sampled tokens (speculation:
// a draft is kept when it equals the file's token), so configurations can be compared on the
// same routing: the generated text otherwise moves the hit rate more than most changes do.
// --save-counts FILE writes the prefill's routing counts (use --count-half-life 0 for a whole
// corpus): a cache prior for flashrt-engine --cache-prior.
// --temp T [--top-k K] [--top-p P] [--min-p M] [--seed S] samples instead of greedy decoding
// (defaults 20, 0.95, 0, 1; the draw for a position depends only on the seed and the position).
// Speculative rounds then sample every verified row and keep drafts while the sample equals the
// draft (exact). --dist-test N (with --spec) checks that at the decode start: N seeds, each a
// speculative round and plain steps from the same state (rewound after each), comparing the
// tokens sampled at the first two positions.
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/mtp.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "kernels/cuda/sample.h"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
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
    std::string ids_path, trace_path, save_state, load_state, mtp_path, vocab_path, save_counts;
    int draft_k = 4, spec_k = 0, vocab_n = 32768, chunk = 64;
    float draft_pmin = 0.0f;
    sample::Params sp;
    sp.temperature = 0.0f;
    uint64_t seed = 1;
    int dist_test = 0;
    int mtp_bits = 2;
    int n_prompt = 1024, gen = 128, slots = 0, reserve_mib = 256, workers = 8, windows = 1;
    bool teacher = false;
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
        else if (a == "--spec") spec_k = std::max(1, std::atoi(next()));
        else if (a == "--mtp-bits") mtp_bits = std::atoi(next());
        else if (a == "--draft-vocab") vocab_path = next();
        else if (a == "--draft-vocab-n") vocab_n = std::atoi(next());
        else if (a == "--draft-pmin") draft_pmin = float(std::atof(next()));
        else if (a == "--temp") sp.temperature = float(std::atof(next()));
        else if (a == "--top-k") sp.top_k = std::atoi(next());
        else if (a == "--top-p") sp.top_p = float(std::atof(next()));
        else if (a == "--min-p") sp.min_p = float(std::atof(next()));
        else if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--dist-test") dist_test = std::atoi(next());
        else if (a == "--teacher") teacher = true;
        else if (a == "--save-counts") save_counts = next();
        else if (a == "--prefill-chunk") {   // a length, or "auto": the longest that fits the free VRAM (ForwardRef::pick_chunk)
            const std::string v = next();
            chunk = v == "auto" ? 0 : std::max(1, std::atoi(v.c_str()));
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<int32_t> seq, text;   // text: the whole ids file (--teacher forces its tokens after the prompt)
    {
        std::ifstream f(ids_path);
        long v;
        while (f >> v) text.push_back(int32_t(v));
        seq.assign(text.begin(), text.begin() + std::min<size_t>(text.size(), size_t(n_prompt)));
    }
    if (int(seq.size()) < n_prompt) { std::fprintf(stderr, "prompt file has only %zu tokens\n", seq.size()); return 1; }
    if (teacher && text.size() < size_t(n_prompt) + size_t(windows) * gen + 64) {
        std::fprintf(stderr, "--teacher: the ids file has no %d tokens after the prompt\n", windows * gen + 64);
        return 1;
    }

    const Gguf g = Gguf::open(argv[1]);
    const Spec s = parse(g);
    const WeightPlan plan = qwen4exp::plan(g, s);
    GpuWeights w;
    w.load(g, plan, q3r);
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    load_experts(g, s, arena, 12);
    {
        const auto tr = Clock::now();
        arena_register(arena);   // at load, outside the timed prefill
        std::printf("expert arena registered in %.1f s\n", std::chrono::duration<double>(Clock::now() - tr).count());
    }
    const std::vector<int> cpus = physical_cpus();
    CpuPool pool(workers, cpus);   // pins this thread to cpus[0]
    if (spec_k > 0 && mtp_path.empty()) { std::fprintf(stderr, "--spec needs --mtp\n"); return 2; }
    if (spec_k > 0) draft_k = spec_k;
    ForwardRef fwd(g, s, w, arena, pool, n_prompt + windows * gen + 16 + draft_k, 64, kv_q8 || kv_hot > 0, kv_hot);
    if (spec_k > 0) fwd.enable_windows(spec_k + 1);
    fwd.set_count_half_life(half_life);
    std::printf("KV cache: %s%s\n", kv_q8 || kv_hot > 0 ? "q8_0" : "fp16",
                kv_hot > 0 ? (", host-resident, hot set of " + std::to_string(kv_hot) + " blocks per layer").c_str() : "");

    float* logits_dev = nullptr;
    cudaMalloc(&logits_dev, size_t(s.n_vocab) * 4);
    int32_t *tok_dev = nullptr, *tok_host = nullptr;
    cudaMalloc(&tok_dev, 64 * 4);
    cudaHostAlloc(&tok_host, 64 * 4, cudaHostAllocDefault);
    // the tokens of rows [0, R) of logits, rows at positions pos0 .. (greedy: argmax)
    // (--teacher: the ids file's own tokens at those positions instead, so every run routes alike)
    auto pick = [&](const float* lg, int R, int64_t pos0) {
        sample::sample_rows(lg, R, s.n_vocab, sp, seed, pos0, tok_dev, fwd.stream());
        cudaMemcpyAsync(tok_host, tok_dev, size_t(R) * 4, cudaMemcpyDeviceToHost, fwd.stream());
        cudaStreamSynchronize(fwd.stream());
        std::vector<int32_t> y(tok_host, tok_host + R);
        if (teacher)
            for (int j = 0; j < R; ++j) y[j] = text[size_t(pos0) + j];
        return y;
    };
    auto argmax = [&]() {
        if (teacher) {
            fwd.argmax(logits_dev);   // same work as a greedy step
            return text[seq.size()];
        }
        return sp.temperature > 0 ? pick(logits_dev, 1, int64_t(seq.size()))[0] : fwd.argmax(logits_dev);
    };

    // MTP draft head: it runs over every prompt batch after the target, taking the target's
    // streams shifted by one position (h_{p-1} with x_p)
    std::unique_ptr<Gguf> g_mtp;
    std::unique_ptr<MtpHead> mtp;
    const size_t hrow = size_t(s.hc_count) * s.d_model;
    float *h_carry = nullptr, *h_buf = nullptr, *mtp_logits = nullptr;
    if (!mtp_path.empty()) {
        g_mtp = std::make_unique<Gguf>(Gguf::open(mtp_path));
        mtp = std::make_unique<MtpHead>(*g_mtp, s, w, fwd.stream(), n_prompt + windows * gen + 16 + draft_k, 64, kv_q8 || kv_hot > 0, kv_hot,
                                        mtp_bits);
        cudaMalloc(&h_carry, hrow * 4);
        cudaMemset(h_carry, 0, hrow * 4);
        cudaMalloc(&h_buf, 64 * hrow * 4);
        cudaMalloc(&mtp_logits, size_t(s.n_vocab) * 4);
        std::printf("MTP draft head: layer %d, %.0f MiB of weights in VRAM, %d drafts %s\n", mtp->layer(),
                    mtp->weight_bytes() / 1048576.0, draft_k, spec_k > 0 ? "per verify round" : "per token (probe)");
    }
    auto mtp_catchup = [&](int p0, int T) {   // in slices of 64 rows (the head's batch)
        if (!mtp) return;
        cudaStream_t st = fwd.stream();
        for (int j = 0; j < T; j += 64) {
            const int Tj = std::min(64, T - j);
            const float* h = fwd.streams() + size_t(j - 1) * hrow;   // rows j-1 .. j+Tj-2
            if (j == 0) {
                cudaMemcpyAsync(h_buf, h_carry, hrow * 4, cudaMemcpyDeviceToDevice, st);
                if (Tj > 1) cudaMemcpyAsync(h_buf + hrow, fwd.streams(), size_t(Tj - 1) * hrow * 4, cudaMemcpyDeviceToDevice, st);
                h = h_buf;
            }
            mtp->forward(h, seq.data() + p0 + j, Tj, p0 + j, Tj, nullptr);
        }
        cudaMemcpyAsync(h_carry, fwd.streams() + size_t(T - 1) * hrow, hrow * 4, cudaMemcpyDeviceToDevice, st);
    };

    // prefill
    {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::printf("VRAM before prefill: %zu MiB free\n", fr >> 20);
        if (chunk == 0) {
            chunk = fwd.pick_chunk(n_prompt, n_prompt, fr);
            std::printf("prefill chunk: %d tokens (auto; estimated %zu MiB)\n", chunk, fwd.chunk_bytes(chunk, n_prompt) >> 20);
        }
    }
    const auto tp = Clock::now();
    if (!load_state.empty()) {
        // the state holds positions 0 .. n_prompt-2; the last prompt token runs now, for its logits
        fwd.load_state(load_state);
        if (fwd.pos() != n_prompt - 1) { std::fprintf(stderr, "state holds %d positions, expected %d\n", fwd.pos(), n_prompt - 1); return 1; }
        if (mtp) {   // the head's own state, if saved with the target's
            std::FILE* mf = std::fopen((load_state + ".mtp").c_str(), "rb");
            if (mf) {
                std::fclose(mf);
                const int mp = mtp->load_state(load_state + ".mtp", h_carry);
                if (mp != n_prompt - 1) { std::fprintf(stderr, "MTP state holds %d positions, expected %d\n", mp, n_prompt - 1); return 1; }
                std::printf("state: MTP head state loaded\n");
            } else {
                std::printf("state: warning: no %s.mtp; the MTP head has no KV for the loaded positions\n", load_state.c_str());
            }
        }
        fwd.forward(seq.data(), 1, 0, logits_dev);
        mtp_catchup(n_prompt - 1, 1);
        std::printf("state: loaded %s (%d positions) in %.1f s\n", load_state.c_str(), n_prompt - 1,
                    std::chrono::duration<double>(Clock::now() - tp).count());
    } else {
        fwd.set_prefill_lookahead(seq.data(), n_prompt);
        for (int p = 0; p < n_prompt; p += chunk) {
            const int T = std::min(chunk, n_prompt - p);
            const bool last = p + T >= n_prompt;
            if (last && !save_state.empty()) {   // save before the last token, so a load can re-run it
                if (T > 1) fwd.forward(seq.data(), T - 1, T - 1, nullptr);
                if (T > 1) mtp_catchup(p, T - 1);
                fwd.save_state(save_state);
                if (mtp) mtp->save_state(save_state + ".mtp", fwd.pos(), h_carry);
                std::printf("state: saved %s%s (%d positions)\n", save_state.c_str(), mtp ? " and .mtp" : "", fwd.pos());
                fwd.forward(seq.data(), 1, 0, logits_dev);
                mtp_catchup(p + T - 1, 1);
                break;
            }
            const auto tc = Clock::now();
            fwd.forward(seq.data(), T, last ? T - 1 : T, last ? logits_dev : nullptr);
            mtp_catchup(p, T);
            if (chunk > 64 && (p / chunk) % 4 == 0)
                std::printf("  chunk at %6d: %d tokens in %.2f s (%.0f tok/s)\n", p, T, std::chrono::duration<double>(Clock::now() - tc).count(),
                            T / std::chrono::duration<double>(Clock::now() - tc).count());
        }
        const double prefill_s = std::chrono::duration<double>(Clock::now() - tp).count();
        std::printf("prefill: %d tokens in %.1f s (%.1f tok/s, %s)\n", n_prompt, prefill_s, n_prompt / prefill_s,
                    chunk > 64 ? ("chunks of " + std::to_string(chunk) + ", experts streamed to the GPU").c_str() : "reference path");
    }
    if (chunk > 64) {   // what the chunk path held at its peak (sizes the chunk for a depth)
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::printf("VRAM after prefill: %zu MiB used, %zu MiB free (chunk buffers %zu MiB)\n", (tot - fr) >> 20, fr >> 20,
                    fwd.chunk_buffer_bytes() >> 20);
    }
    fwd.release_chunk_buffers();   // the expert cache takes that VRAM
    if (!save_counts.empty()) {   // the prefill's routing counts, a cache prior for flashrt-engine --cache-prior
        std::FILE* f = std::fopen(save_counts.c_str(), "wb");
        const int64_t h[4] = {0x50435246 /* "FRCP" */, s.n_layer, s.n_expert, n_prompt};
        const bool ok = f && std::fwrite(h, sizeof(h), 1, f) == 1 && std::fwrite(fwd.counts().data(), 4, fwd.counts().size(), f) == fwd.counts().size();
        if (f) std::fclose(f);
        std::printf("routing counts: %s %s\n", ok ? "saved to" : "FAILED to save", save_counts.c_str());
    }
    if (mtp && !vocab_path.empty()) {   // the drafter's vocabulary: top ranked tokens plus the prompt's, at most 65536
        std::vector<int32_t> ids;
        std::vector<char> in(s.n_vocab, 0);
        std::ifstream f(vocab_path);
        long v;
        while (int(ids.size()) < vocab_n && f >> v)
            if (v >= 0 && v < s.n_vocab && !in[v]) { in[v] = 1; ids.push_back(int32_t(v)); }
        std::vector<int> cnt(s.n_vocab, 0);
        for (int p = 0; p < n_prompt; ++p) ++cnt[seq[p]];
        std::vector<int32_t> extra;
        for (int t = 0; t < s.n_vocab; ++t)
            if (cnt[t] && !in[t]) extra.push_back(t);
        std::stable_sort(extra.begin(), extra.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        const int n_static = int(ids.size());
        for (int32_t t : extra) {
            if (ids.size() >= 65536) break;
            ids.push_back(t);
        }
        mtp->set_vocab(ids);
        std::printf("MTP draft vocabulary: %zu tokens (%d ranked, %zu from the prompt)\n", ids.size(), n_static, ids.size() - n_static);
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
        host = alloc_moe_fast_host(s, spec_k + 1);
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
            std::printf("adaptive cache: decayed LFU, swap budget %d (set up in %.1f s)\n", swap_budget,
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
    // speculative decoding state: the MTP rows still to run (positions p - pend + 1 .. p, their h
    // in h_buf), and statistics
    int pend = 1;
    if (spec_k > 0) cudaMemcpy(h_buf, h_carry, hrow * 4, cudaMemcpyDeviceToDevice);
    float* logits_win = nullptr;
    if (spec_k > 0) cudaMalloc(&logits_win, size_t(spec_k + 1) * s.n_vocab * 4);
    std::vector<long> acc_hist(spec_k + 1, 0);
    long rounds = 0, drafted = 0;
    double draft_s = 0, verify_s = 0, commit_s = 0;
    if (dist_test > 0 && spec_k > 0) {
        // the drafts from this state (deterministic; the head's KV at these positions is rewritten
        // by the real rounds later)
        const int p = int(seq.size()) - 1;
        std::vector<int32_t> win(seq.end() - 1, seq.end());
        mtp->forward(h_buf, seq.data() + p - pend + 1, pend, p - pend + 1, pend - 1, mtp_logits);
        int32_t d = mtp->argmax(mtp_logits);
        win.push_back(d);
        for (int j = 1; j < spec_k; ++j) {
            mtp->forward(mtp->h_out() + size_t(j == 1 ? pend - 1 : 0) * hrow, &d, 1, p + j, 0, mtp_logits);
            d = mtp->argmax(mtp_logits);
            win.push_back(d);
        }
        std::vector<int32_t> seqw(seq.begin(), seq.end() - 1);
        seqw.insert(seqw.end(), win.begin(), win.end());
        const uint64_t seed0 = seed;
        long eq0 = 0, n1 = 0, eq1 = 0, acc1 = 0;
        std::map<int32_t, long> h_spec, h_plain;
        for (int i = 0; i < dist_test; ++i) {
            seed = seed0 + 1000003ull * uint64_t(i + 1);
            // speculative round: sample every row, keep while equal to the draft
            fwd.forward_window(seqw.data(), spec_k + 1, logits_win);
            const std::vector<int32_t> y = pick(logits_win, spec_k + 1, p + 1);
            fwd.commit(0);
            int a = 0;
            while (a < spec_k && y[a] == win[a + 1]) ++a;
            // plain steps: y0 from x_p; y1 from x_p, y0
            std::vector<int32_t> s2(seq.begin(), seq.end());
            fwd.forward_window(s2.data(), 1, logits_win);
            const int32_t z0 = pick(logits_win, 1, p + 1)[0];
            fwd.commit(0);
            s2.push_back(z0);
            fwd.forward_window(s2.data(), 2, logits_win);
            const int32_t z1 = pick(logits_win + s.n_vocab, 1, p + 2)[0];
            fwd.commit(0);
            eq0 += y[0] == z0;
            ++h_spec[y[0]];
            ++h_plain[z0];
            if (a >= 1) {   // the spec round emitted a second token
                ++acc1;
                if (z0 == y[0]) {
                    ++n1;
                    eq1 += y[1] == z1;
                }
            }
        }
        seed = seed0;
        double tv = 0;
        for (const auto& [t, c] : h_spec) tv += std::fabs(double(c) - double(h_plain.count(t) ? h_plain[t] : 0));
        for (const auto& [t, c] : h_plain)
            if (!h_spec.count(t)) tv += double(c);
        std::printf("distribution test: %d seeds at position %d, %zu drafts; first token equal in %ld (%.2f%%), histogram TV %.4f; "
                    "second token emitted in %ld, equal in %ld of %ld\n",
                    dist_test, p + 1, win.size() - 1, eq0, 100.0 * eq0 / dist_test, tv / (2.0 * dist_test), acc1, eq1, n1);
    }
    cudaProfilerStart();   // nsys --capture-range=cudaProfilerApi profiles the decode loop only
    for (int wi = 0; wi < windows; ++wi) {
        const long hits0 = host.hits, misses0 = host.misses, gmiss0 = host.gpu_misses;
        const int depth = int(seq.size()) - 1;
        const auto td = Clock::now();
        int emitted = 0;
        while (spec_k > 0 && emitted < gen) {
            // 1. draft: the pending MTP rows (catch-up) give d1, then K - 1 chained steps
            const auto t0 = Clock::now();
            const int p = int(seq.size()) - 1;
            std::vector<int32_t> win(seq.end() - 1, seq.end());   // x_p, d1 .. dK
            if (draft_pmin > 0) {   // step by step, checking each draft's probability
                mtp->forward(h_buf, seq.data() + p - pend + 1, pend, p - pend + 1, pend - 1, mtp_logits);
                float pd = 1.0f;
                int32_t d = mtp->argmax(mtp_logits, &pd);
                if (pd >= draft_pmin) win.push_back(d);
                for (int j = 1; j < spec_k && pd >= draft_pmin; ++j) {
                    mtp->forward(mtp->h_out() + size_t(j == 1 ? pend - 1 : 0) * hrow, &d, 1, p + j, 0, mtp_logits);
                    d = mtp->argmax(mtp_logits, &pd);
                    if (pd >= draft_pmin) win.push_back(d);
                }
            } else {   // the catch-up rows, then the chain in one sync
                mtp->forward(h_buf, seq.data() + p - pend + 1, pend, p - pend + 1, pend - 1, mtp->chain_logits());
                const std::vector<int32_t> dr = mtp->draft_chain(pend - 1, p + 1, spec_k);
                win.insert(win.end(), dr.begin(), dr.end());
            }
            const int kd = int(win.size()) - 1;   // drafts this round
            // 2. verify the window x_p, d1 .. dK
            const auto t1 = Clock::now();
            std::vector<int32_t> seqw(seq.begin(), seq.end() - 1);
            seqw.insert(seqw.end(), win.begin(), win.end());
            fwd.forward_window(seqw.data(), kd + 1, logits_win);
            const std::vector<int32_t> y = pick(logits_win, kd + 1, p + 1);   // y_j for position p + 1 + j
            int a = 0;
            while (a < kd && y[a] == win[a + 1]) ++a;
            drafted += kd;
            // 3. keep x_p, d1 .. da; emit y_0 .. y_a
            const auto t2 = Clock::now();
            fwd.commit(a + 1);
            cudaMemcpyAsync(h_buf, fwd.streams(), size_t(a + 1) * hrow * 4, cudaMemcpyDeviceToDevice, fwd.stream());
            pend = a + 1;
            for (int j = 0; j <= a; ++j) {
                seq.push_back(y[j]);
                out.push_back(y[j]);
            }
            emitted += a + 1;
            ++acc_hist[a];
            ++rounds;
            const auto t3 = Clock::now();
            draft_s += std::chrono::duration<double>(t1 - t0).count();
            verify_s += std::chrono::duration<double>(t2 - t1).count();
            commit_s += std::chrono::duration<double>(t3 - t2).count();
        }
        for (int i = 0; spec_k == 0 && i < gen; ++i) {
            if (mtp) {   // draft from (h_{p-1}, x_p): the first step is also position p's catch-up
                const auto tm = Clock::now();
                const int p = int(seq.size()) - 1;
                int32_t d = seq[p];
                draft_pos.push_back(p);
                for (int j = 0; j < draft_k; ++j) {
                    mtp->forward(j == 0 ? h_carry : mtp->h_out(), &d, 1, p + j, 0, mtp_logits);
                    d = mtp->argmax(mtp_logits);
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
        const int n_dec = spec_k > 0 ? emitted : gen;
        std::printf("decode: %d tokens at depth %d in %.2f s: %.2f tok/s (%s)", n_dec, depth, dec_s, n_dec / dec_s,
                    spec_k > 0 ? "speculative, MTP drafts" : reference ? "reference path" : doorbell ? "fast path, doorbell" : "fast path, host sync per layer");
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
    if (spec_k > 0 && rounds > 0) {
        long toks = 0;
        for (int a = 0; a <= spec_k; ++a) toks += acc_hist[a] * (a + 1);
        std::printf("speculative: %ld rounds, %.3f tokens per round, %.2f drafts verified per round; accepted drafts:", rounds,
                    double(toks) / rounds, double(drafted) / rounds);
        for (int a = 0; a <= spec_k; ++a) std::printf(" %d:%.1f%%", a, 100.0 * acc_hist[a] / rounds);
        std::printf("\n  per round: draft %.2f ms, verify %.2f ms, commit %.2f ms\n", 1e3 * draft_s / rounds, 1e3 * verify_s / rounds,
                    1e3 * commit_s / rounds);
    }
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
        if (logits_win) cudaFree(logits_win);
    }
    arena_free(arena);
    return 0;
}
