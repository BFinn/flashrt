// SPDX-License-Identifier: Apache-2.0
#include "engine/session.hpp"

#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/mtp.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/cpu_pool.hpp"
#include "core/expert_arena.hpp"
#include "core/gguf.hpp"
#include "core/platform.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace flashrt {

using namespace qwen4exp;
using Clock = std::chrono::steady_clock;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

struct Session::Impl {
    SessionOptions o;
    Gguf g;
    Spec s;
    WeightPlan plan;
    GpuWeights w;
    ExpertArena arena;
    std::vector<int> cpus;
    std::unique_ptr<CpuPool> pool;
    std::unique_ptr<ForwardRef> fwd;
    std::unique_ptr<Gguf> g_mtp;
    std::unique_ptr<MtpHead> mtp;
    ExpertCache cache;
    MoeFastHost host;
    CacheManager* mgr = nullptr;
    bool cache_filled = false;
    std::vector<int32_t> ranked;   // the head's static vocabulary, best first
    std::vector<float> prior;      // routing counts of a calibration prefill, scaled to 4,096 tokens

    std::vector<int32_t> seq;      // the tokens the target has processed: positions 0 .. pos() - 1
    size_t hrow = 0;
    float *logits = nullptr, *logits_win = nullptr, *h_carry = nullptr, *h_buf = nullptr;
    int32_t *tok_dev = nullptr, *tok_host = nullptr;

    explicit Impl(const SessionOptions& opt) : o(opt), g(Gguf::open(opt.model)), s(parse(g)), plan(qwen4exp::plan(g, s)) {
        w.load(g, plan, true);
        arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
        if (!arena.buf.ptr) throw std::runtime_error("expert arena allocation failed");
        load_experts(g, s, arena, 12);
        cpus = physical_cpus();
        pool = std::make_unique<CpuPool>(o.workers, cpus);   // pins this thread to cpus[0] for now
        const bool spec = !o.mtp.empty() && o.spec_k > 0;
        const int W = spec ? o.spec_k + 1 : 1;
        fwd = std::make_unique<ForwardRef>(g, s, w, arena, *pool, o.max_ctx + 16, o.prefill_batch, o.kv_q8 || o.kv_hot > 0, o.kv_hot);
        fwd->set_count_half_life(4096);
        hrow = size_t(s.hc_count) * s.d_model;
        if (spec) {
            fwd->enable_windows(W);
            g_mtp = std::make_unique<Gguf>(Gguf::open(o.mtp));
            mtp = std::make_unique<MtpHead>(*g_mtp, s, w, fwd->stream(), o.max_ctx + 16, o.prefill_batch, o.kv_q8 || o.kv_hot > 0, o.kv_hot,
                                            o.mtp_bits);
            if (!o.draft_vocab.empty()) {
                std::ifstream f(o.draft_vocab);
                if (!f) throw std::runtime_error("cannot read " + o.draft_vocab);
                long v;
                std::vector<char> in(s.n_vocab, 0);
                while (int(ranked.size()) < o.draft_vocab_n && f >> v)
                    if (v >= 0 && v < s.n_vocab && !in[v]) {
                        in[v] = 1;
                        ranked.push_back(int32_t(v));
                    }
                mtp->reserve_vocab(65536);
            }
        }
        ck(cudaMalloc(&logits, size_t(s.n_vocab) * 4), "cudaMalloc logits");
        ck(cudaMalloc(&logits_win, size_t(W) * s.n_vocab * 4), "cudaMalloc logits");
        ck(cudaMalloc(&h_carry, hrow * 4), "cudaMalloc h");
        ck(cudaMemset(h_carry, 0, hrow * 4), "memset h");
        ck(cudaMalloc(&h_buf, size_t(o.prefill_batch) * hrow * 4), "cudaMalloc h");
        ck(cudaMalloc(&tok_dev, 64 * 4), "cudaMalloc tokens");
        ck(cudaHostAlloc(&tok_host, 64 * 4, cudaHostAllocDefault), "cudaHostAlloc tokens");
        // the expert cache takes the VRAM that is left; it is filled after the first prefill
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t eb = q2_0::expert_bytes({s.d_model, s.d_ff_expert}), keep = size_t(o.reserve_mib) << 20;
        if (free_b < keep + 256 * eb) throw std::runtime_error("not enough VRAM left for the expert cache");
        cache = alloc_expert_cache(s, int((free_b - keep) / eb));
        host = alloc_moe_fast_host(s, W);
        host.arena = &arena;
        host.pool = pool.get();
        start_doorbell(s, host, cpus[0]);                             // the miss server takes cpus[0]
        pin_current_thread(cpus[size_t(o.workers) % cpus.size()]);    // this (the enqueue) thread moves off it
        pool->set_spin_us(2000);
        fwd->set_graphs(true);
        fwd->set_fast_moe(&cache, &host);
        if (!o.cache_prior.empty()) {   // fill the cache now, so the first requests do not start cold
            std::FILE* f = std::fopen(o.cache_prior.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open " + o.cache_prior);
            int64_t h[4] = {0, 0, 0, 0};
            std::vector<uint32_t> c(size_t(s.n_layer) * s.n_expert);
            const bool ok = std::fread(h, sizeof(h), 1, f) == 1 && h[0] == 0x50435246 && h[1] == s.n_layer && h[2] == s.n_expert &&
                            std::fread(c.data(), 4, c.size(), f) == c.size();
            std::fclose(f);
            if (!ok || h[3] <= 0) throw std::runtime_error(o.cache_prior + " is not a routing prior for this model");
            prior.resize(c.size());
            for (size_t i = 0; i < c.size(); ++i) prior[i] = float(c[i]) * 4096.0f / float(h[3]);
            refill_cache();
            std::fprintf(stderr, "flashrt: expert cache filled from %s (%lld tokens)\n", o.cache_prior.c_str(), (long long)h[3]);
        }
        std::fprintf(stderr, "flashrt: %s, %d layers, expert cache %d slots (%.1f%%), %s\n", s.arch.c_str(), s.n_layer, cache.n_slots,
                     100.0 * cache.n_slots / (s.n_layer * s.n_expert),
                     spec ? ("MTP drafts " + std::to_string(o.spec_k) + " per round").c_str() : "no speculation");
    }

    ~Impl() {
        if (fwd) fwd->set_cache_manager(nullptr);
        destroy_cache_manager(mgr);
        free_moe_fast_host(host);
        free_expert_cache(cache);
        mtp.reset();
        fwd.reset();
        for (void* p : {static_cast<void*>(logits), static_cast<void*>(logits_win), static_cast<void*>(h_carry), static_cast<void*>(h_buf),
                        static_cast<void*>(tok_dev)})
            if (p) cudaFree(p);
        if (tok_host) cudaFreeHost(tok_host);
        pool.reset();
        arena_free(arena);
    }

    // The draft head catches up on target rows p0 .. p0 + T - 1 (their streams are in
    // fwd->streams()): its input at q is (h_{q-1}, x_q).
    void mtp_catchup(const int32_t* toks, int p0, int T) {
        if (!mtp) return;
        cudaStream_t st = fwd->stream();
        ck(cudaMemcpyAsync(h_buf, h_carry, hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
        if (T > 1) ck(cudaMemcpyAsync(h_buf + hrow, fwd->streams(), size_t(T - 1) * hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
        ck(cudaMemcpyAsync(h_carry, fwd->streams() + size_t(T - 1) * hrow, hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
        mtp->forward(h_buf, toks + p0, T, p0, T, nullptr);
    }

    std::vector<int32_t> pick(const float* lg, int R, int64_t pos0, const GenerateRequest& r) {
        sample::sample_rows(lg, R, s.n_vocab, r.sampling, r.seed, pos0, tok_dev, fwd->stream());
        ck(cudaMemcpyAsync(tok_host, tok_dev, size_t(R) * 4, cudaMemcpyDeviceToHost, fwd->stream()), "tokens");
        ck(cudaStreamSynchronize(fwd->stream()), "sample");
        return std::vector<int32_t>(tok_host, tok_host + R);
    }

    // Refills the whole cache from the prefill's routing counts (plus the prior, worth 4,096
    // tokens) and restarts the adaptive policy.
    void refill_cache() {
        fwd->set_cache_manager(nullptr);
        destroy_cache_manager(mgr);
        mgr = nullptr;
        std::fill(cache.table.begin(), cache.table.end(), -1);
        std::fill(cache.owner.begin(), cache.owner.end(), -1);
        std::vector<uint32_t> cnt = fwd->counts();
        for (size_t i = 0; i < cnt.size() && i < prior.size(); ++i) cnt[i] += uint32_t(prior[i] + 0.5f);
        std::vector<int> idx(cnt.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        std::vector<std::pair<int, int>> order;
        for (int i : idx) order.push_back({i / s.n_expert, i % s.n_expert});
        expert_cache_fill(s, cache, arena, order, fwd->stream());
        CachePolicyConfig cfg;
        cfg.budget = o.swap_budget;
        mgr = create_cache_manager(s, cache, arena, cfg, cnt);
        fwd->set_cache_manager(mgr);
        cache_filled = true;
    }

    void set_draft_vocab(const std::vector<int32_t>& prompt) {
        if (!mtp || ranked.empty()) return;
        std::vector<int32_t> ids = ranked;
        std::vector<char> in(s.n_vocab, 0);
        for (int32_t t : ids) in[t] = 1;
        std::vector<int> cnt(s.n_vocab, 0);
        for (int32_t t : prompt)
            if (t >= 0 && t < s.n_vocab) ++cnt[t];
        std::vector<int32_t> extra;
        for (int t = 0; t < s.n_vocab; ++t)
            if (cnt[t] && !in[t]) extra.push_back(t);
        std::stable_sort(extra.begin(), extra.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        for (int32_t t : extra) {
            if (ids.size() >= 65536) break;
            ids.push_back(t);
        }
        mtp->set_vocab(ids);
    }
};

Session::Session(const SessionOptions& o) : m_(std::make_unique<Impl>(o)) {}
Session::~Session() = default;
int Session::max_context() const { return m_->o.max_ctx; }
int Session::n_vocab() const { return m_->s.n_vocab; }
std::string Session::arch() const { return m_->s.arch; }
bool Session::speculative() const { return m_->mtp != nullptr; }

GenerateResult Session::generate(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                                 const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel) {
    Impl& m = *m_;
    GenerateResult res;
    const auto t0 = Clock::now();
    const std::vector<int32_t>& P = r.prompt;
    const int n = int(P.size());
    res.prompt_tokens = n;
    if (n < 1) throw std::runtime_error("empty prompt");
    for (int32_t t : P)
        if (t < 0 || t >= m.s.n_vocab) throw std::runtime_error("token id out of range: " + std::to_string(t));
    const int K = m.mtp ? m.o.spec_k : 0;
    if (n + K + 2 > m.o.max_ctx) throw std::runtime_error("prompt longer than the context");

    // 1. reuse: the whole previous sequence, or the last prompt's checkpoint, or nothing
    size_t L = 0;
    while (L < m.seq.size() && L < P.size() && m.seq[L] == P[L]) ++L;
    const size_t usable = std::min(L, size_t(n - 1));   // the last prompt token always runs, for its logits
    if (!m.seq.empty() && usable == m.seq.size()) {
        // continue from the current state
    } else if (m.fwd->checkpoint_pos() > 0 && size_t(m.fwd->checkpoint_pos()) <= usable) {
        m.fwd->restore_checkpoint();
        if (m.mtp) m.mtp->restore_checkpoint(m.h_carry);
        m.seq.resize(size_t(m.fwd->checkpoint_pos()));
    } else {
        m.fwd->reset();
        if (m.mtp) m.mtp->reset();
        ck(cudaMemset(m.h_carry, 0, m.hrow * 4), "memset h");
        m.seq.clear();
    }
    res.reused = int(m.seq.size());

    // 2. prefill every new prompt token but the last in batches (reference path), the head catching
    // up on each batch; checkpoint there, so the same prompt again or one extending it reuses all
    // of it; then the last token alone, for its logits
    const int from = int(m.seq.size());
    for (int p = from; p < n - 1; p += m.o.prefill_batch) {
        const int T = std::min(m.o.prefill_batch, n - 1 - p);
        m.fwd->forward(P.data(), T, T, nullptr);
        m.mtp_catchup(P.data(), p, T);
        if (on_progress) on_progress(p + T, n);
        if (cancel.load()) {
            m.seq.assign(P.begin(), P.begin() + p + T);
            res.finish = "cancelled";
            res.prompt_ms = ms_since(t0);
            return res;
        }
    }
    m.seq.assign(P.begin(), P.end() - 1);
    m.fwd->save_checkpoint();
    if (m.mtp) m.mtp->save_checkpoint(m.h_carry);
    if (!m.cache_filled || n - from >= 4096) m.refill_cache();
    m.fwd->forward(P.data(), 1, 0, m.logits);
    m.mtp_catchup(P.data(), n - 1, 1);
    m.seq = P;
    if (on_progress) on_progress(n, n);
    if (m.mtp) m.set_draft_vocab(P);
    res.prompt_ms = ms_since(t0);

    // 3. decode
    const auto td = Clock::now();
    auto is_stop = [&](int32_t t) { return std::find(r.stop_ids.begin(), r.stop_ids.end(), t) != r.stop_ids.end(); };
    int32_t y = m.pick(m.logits, 1, n, r)[0];   // the pending token: sampled, not yet run
    bool done = false;
    auto emit = [&](int32_t t) {
        if (is_stop(t)) {
            res.finish = "stop";
            done = true;
            return;
        }
        on_token(t);
        if (++res.generated >= r.max_new) {
            res.finish = "length";
            done = true;
        }
    };
    emit(y);
    int pend = 1;   // speculative: the head's rows still to run, positions seq.size() - pend + 1 .. seq.size()
    if (K > 0) ck(cudaMemcpy(m.h_buf, m.h_carry, m.hrow * 4, cudaMemcpyDeviceToDevice), "h");
    std::vector<int32_t> seqw;
    while (!done) {
        if (cancel.load()) {
            res.finish = "cancelled";
            break;
        }
        const int p = int(m.seq.size());   // y's position
        if (p + K + 2 > m.o.max_ctx) {
            res.finish = "length";
            break;
        }
        seqw = m.seq;
        seqw.push_back(y);
        if (K == 0) {
            m.fwd->forward(seqw.data(), 1, 0, m.logits);
            m.seq.push_back(y);
            y = m.pick(m.logits, 1, p + 1, r)[0];
            emit(y);
            continue;
        }
        // draft: the pending head rows, then the chain
        m.mtp->forward(m.h_buf, seqw.data() + p - pend + 1, pend, p - pend + 1, pend - 1, m.mtp->chain_logits());
        const std::vector<int32_t> d = m.mtp->draft_chain(pend - 1, p + 1, K);
        seqw.insert(seqw.end(), d.begin(), d.end());
        // verify, sample every row, keep drafts while they match
        m.fwd->forward_window(seqw.data(), K + 1, m.logits_win);
        const std::vector<int32_t> ys = m.pick(m.logits_win, K + 1, p + 1, r);
        int a = 0;
        while (a < K && ys[a] == d[a]) ++a;
        m.fwd->commit(a + 1);
        ck(cudaMemcpyAsync(m.h_buf, m.fwd->streams(), size_t(a + 1) * m.hrow * 4, cudaMemcpyDeviceToDevice, m.fwd->stream()), "h");
        m.seq.insert(m.seq.end(), seqw.begin() + p, seqw.begin() + p + a + 1);
        pend = a + 1;
        res.drafts_proposed += K;
        res.drafts_accepted += a;
        for (int j = 0; j <= a && !done; ++j) {
            y = ys[j];
            emit(y);
        }
    }
    // leave the head caught up on everything the target processed (for a continuing request)
    if (K > 0 && pend > 1) {
        const int p = int(m.seq.size()) - (pend - 1);
        m.mtp->forward(m.h_buf, m.seq.data() + p, pend - 1, p, pend - 1, nullptr);
    }
    if (K > 0) ck(cudaMemcpyAsync(m.h_carry, m.h_buf + size_t(pend - 1) * m.hrow, m.hrow * 4, cudaMemcpyDeviceToDevice, m.fwd->stream()), "h");
    ck(cudaStreamSynchronize(m.fwd->stream()), "generate");
    res.decode_ms = ms_since(td);
    return res;
}

}  // namespace flashrt
