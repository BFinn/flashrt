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
#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

// Everything generate() can check before it changes any state.
void validate(const GenerateRequest& r, int n_vocab, int max_ctx, int K) {
    const int n = int(r.prompt.size());
    if (n < 1) throw std::runtime_error("empty prompt");
    for (int32_t t : r.prompt)
        if (t < 0 || t >= n_vocab) throw std::runtime_error("token id out of range: " + std::to_string(t));
    for (int32_t t : r.stop_ids)
        if (t < 0 || t >= n_vocab) throw std::runtime_error("stop id out of range: " + std::to_string(t));
    if (n + K + 2 > max_ctx) throw std::runtime_error("prompt longer than the context");
    if (r.max_new < 1) throw std::runtime_error("max_new must be at least 1");
    const sample::Params& p = r.sampling;
    if (!std::isfinite(p.temperature) || p.temperature < 0.0f) throw std::runtime_error("temperature must be >= 0");
    if (p.top_k < 1 || p.top_k > sample::kMaxTopK)
        throw std::runtime_error("top_k must be 1.." + std::to_string(sample::kMaxTopK));
    if (!(p.top_p > 0.0f && p.top_p <= 1.0f)) throw std::runtime_error("top_p must be in (0, 1]");
    if (!(p.min_p >= 0.0f && p.min_p < 1.0f)) throw std::runtime_error("min_p must be in [0, 1)");
}

// True when the device is usable: waits for the queued work, clears a one-off error, and fails
// on a sticky one (it reports again).
bool cuda_usable() {
    const cudaError_t e = cudaDeviceSynchronize();
    (void)cudaGetLastError();
    return e == cudaSuccess || cudaDeviceSynchronize() == cudaSuccess;
}
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
    // Host checkpoints, taken during prefills (SessionOptions::ckpts). Each is valid for seq's
    // first `pos` tokens: run() drops those past the prefix a new prompt shares with seq.
    struct HostCkpt {
        int pos = -1;
        char* buf = nullptr;   // the target's state, then the head's
    };
    std::vector<HostCkpt> ring;
    char* ring_mem = nullptr;
    size_t ckpt_fwd = 0, ckpt_mtp = 0;
    bool healthy = true;
    size_t hrow = 0;
    float *logits = nullptr, *logits_win = nullptr, *h_carry = nullptr, *h_buf = nullptr;
    int32_t *tok_dev = nullptr, *tok_host = nullptr;

    explicit Impl(const SessionOptions& opt) : o(opt), g(Gguf::open(opt.model)), s(parse(g)), plan(qwen4exp::plan(g, s)) {
        auto t = Clock::now();
        auto stage = [&](const char* what) {   // the load's stages, timed, in the log
            std::fprintf(stderr, "flashrt: %s in %.1f s\n", what, ms_since(t) / 1000);
            t = Clock::now();
        };
        w.load(g, plan, true);
        stage("GPU weights loaded");
        arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
        if (!arena.buf.ptr) throw std::runtime_error("expert arena allocation failed");
        load_experts(g, s, arena, 12);
        arena_register(arena);   // now, not in the first long prompt's prefill
        stage("experts read into the host arena and registered");
        cpus = physical_cpus();
        pool = std::make_unique<CpuPool>(o.workers, cpus);   // pins this thread to cpus[0] for now
        const bool spec = !o.mtp.empty() && o.spec_k > 0;
        const int W = spec ? o.spec_k + 1 : 1;
        fwd = std::make_unique<ForwardRef>(g, s, w, arena, *pool, o.max_ctx + 16, o.prefill_batch, o.kv_q8 || o.kv_hot > 0, o.kv_hot);
        fwd->set_count_half_life(4096);
        hrow = size_t(s.hc_count) * s.d_model;
        stage("forward pass set up");
        if (spec) {
            fwd->enable_windows(W);
            g_mtp = std::make_unique<Gguf>(Gguf::open(o.mtp));
            mtp = std::make_unique<MtpHead>(*g_mtp, s, w, fwd->stream(), o.max_ctx + 16, o.prefill_batch, o.kv_q8 || o.kv_hot > 0, o.kv_hot,
                                            o.mtp_bits);
            stage("MTP head loaded");
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
        // what requests allocate on first use goes first: the recurrent-state checkpoints (sw86: a
        // server with the reserve at 256 MiB failed its first request allocating them)
        fwd->reserve_checkpoint();
        if (mtp) mtp->reserve_checkpoint();
        if (o.ckpts > 0) {
            ckpt_fwd = fwd->checkpoint_bytes();
            ckpt_mtp = mtp ? mtp->checkpoint_bytes() : 0;
            const size_t each = (ckpt_fwd + ckpt_mtp + 4095) & ~size_t(4095);
            ck(cudaHostAlloc(&ring_mem, each * size_t(o.ckpts), cudaHostAllocDefault), "cudaHostAlloc checkpoints");
            for (int i = 0; i < o.ckpts; ++i) ring.push_back({-1, ring_mem + each * size_t(i)});
            std::fprintf(stderr, "flashrt: %d prefill checkpoints of %.1f MiB in host RAM\n", o.ckpts, double(each) / (1 << 20));
        }
        stage("buffers and checkpoints allocated");
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
        stage("expert cache and miss server set up");
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
        if (ring_mem) cudaFreeHost(ring_mem);
        pool.reset();
        arena_free(arena);
    }

    // The draft head catches up on target rows p0 .. p0 + T - 1 (their streams are in
    // fwd->streams()): its input at q is (h_{q-1}, x_q). In slices of its batch.
    void mtp_catchup(const int32_t* toks, int p0, int T) {
        if (!mtp) return;
        cudaStream_t st = fwd->stream();
        const int B = o.prefill_batch;
        for (int j = 0; j < T; j += B) {
            const int Tj = std::min(B, T - j);
            const float* h = fwd->streams() + size_t(j - 1) * hrow;   // rows j-1 .. j+Tj-2
            if (j == 0) {
                ck(cudaMemcpyAsync(h_buf, h_carry, hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
                if (Tj > 1) ck(cudaMemcpyAsync(h_buf + hrow, fwd->streams(), size_t(Tj - 1) * hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
                h = h_buf;
            }
            mtp->forward(h, toks + p0 + j, Tj, p0 + j, Tj, nullptr);
        }
        ck(cudaMemcpyAsync(h_carry, fwd->streams() + size_t(T - 1) * hrow, hrow * 4, cudaMemcpyDeviceToDevice, st), "h");
    }

    // The expert cache's VRAM goes to prefill chunks, and comes back afterwards.
    void cache_release() {
        fwd->set_cache_manager(nullptr);
        destroy_cache_manager(mgr);
        mgr = nullptr;
        ck(cudaFree(cache.slots), "free expert cache");
        cache.slots = nullptr;
        std::fill(cache.table.begin(), cache.table.end(), -1);
        std::fill(cache.owner.begin(), cache.owner.end(), -1);
        cache_filled = false;
    }
    void cache_restore() {
        fwd->release_chunk_buffers();
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t eb = cache.slot_bytes, keep = size_t(o.reserve_mib) << 20;
        const int slots = int(std::min<size_t>(size_t(cache.owner.size()), (free_b - std::min(free_b, keep)) / eb));
        ck(cudaMalloc(&cache.slots, size_t(slots) * eb + gemv::kWeightTailPad), "cudaMalloc expert cache");
        ck(cudaMemset(cache.slots, 0, size_t(slots) * eb + gemv::kWeightTailPad), "memset expert cache");
        cache.n_slots = slots;
        cache.owner.assign(slots, -1);
        refill_cache();
    }

    // After a request threw: an empty sequence, the expert cache back in place, no open window or
    // half-done chunk. Leaves healthy false when that is impossible.
    void recover() {
        if (!cuda_usable() || doorbell_failed(host)) {
            healthy = false;
            return;
        }
        try {
            if (!cache.slots) cache_restore();   // the failure came during a chunked prefill
            fwd->reset();
            if (mtp) mtp->reset();
            ck(cudaMemset(h_carry, 0, hrow * 4), "memset h");
            seq.clear();
            drop_checkpoints_after(0);
            ck(cudaDeviceSynchronize(), "recover");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "flashrt: recovery failed: %s\n", e.what());
            healthy = false;
        }
    }

    // Checkpoints hold the state after seq's first pos tokens; those past L (the prefix a new
    // prompt shares with seq) describe other tokens.
    void drop_checkpoints_after(size_t L) {
        if (fwd->checkpoint_pos() > 0 && size_t(fwd->checkpoint_pos()) > L) fwd->drop_checkpoint();
        for (HostCkpt& c : ring)
            if (c.pos > 0 && size_t(c.pos) > L) c.pos = -1;
    }

    // The latest checkpoint at or before `usable` restored, target and head; its position, or 0.
    int restore_best(size_t usable) {
        const HostCkpt* best = nullptr;
        for (const HostCkpt& c : ring)
            if (c.pos > 0 && size_t(c.pos) <= usable && (!best || c.pos > best->pos)) best = &c;
        const int dev = fwd->checkpoint_pos() > 0 && size_t(fwd->checkpoint_pos()) <= usable ? fwd->checkpoint_pos() : 0;
        if (dev > 0 && (!best || dev >= best->pos)) {
            fwd->restore_checkpoint();
            if (mtp) mtp->restore_checkpoint(h_carry);
            return dev;
        }
        if (!best) return 0;
        fwd->restore_checkpoint_from(best->buf, best->pos);
        if (mtp) mtp->restore_checkpoint_from(best->buf + ckpt_fwd, h_carry);
        return best->pos;
    }

    // A host checkpoint at the current position. When the ring is full, the entry whose removal
    // leaves the smallest gap goes (the older one on a tie), so the kept positions thin out
    // evenly over the prefix while the newest stays.
    void save_host_checkpoint() {
        if (ring.empty()) return;
        const int pos = fwd->pos();
        HostCkpt* slot = nullptr;
        for (HostCkpt& c : ring)
            if (c.pos <= 0 || c.pos == pos) {
                slot = &c;
                break;
            }
        if (!slot) {
            std::vector<HostCkpt*> v;
            for (HostCkpt& c : ring) v.push_back(&c);
            std::sort(v.begin(), v.end(), [](const HostCkpt* a, const HostCkpt* b) { return a->pos < b->pos; });
            long best_gap = -1;
            for (size_t i = 0; i < v.size(); ++i) {   // the new one (at pos) follows v.back()
                const long gap = long(i + 1 < v.size() ? v[i + 1]->pos : pos) - long(i > 0 ? v[i - 1]->pos : 0);
                if (best_gap < 0 || gap < best_gap) {
                    best_gap = gap;
                    slot = v[i];
                }
            }
        }
        fwd->save_checkpoint_to(slot->buf);
        if (mtp) mtp->save_checkpoint_to(slot->buf + ckpt_fwd, h_carry);
        slot->pos = pos;
    }

    // Sampled drafts (temperature > 0; FLASHRT_ARGMAX_DRAFTS=1: argmax drafts): the verify rows
    // through speculative sampling with the head's q; a = the drafts kept (sw85)
    std::vector<int32_t> spec_pick(const float* lg, int R, int64_t pos0, const GenerateRequest& r, int& a) {
        sample::spec_verify(lg, R, s.n_vocab, r.sampling, r.seed, pos0, mtp->drafts_dev(), mtp->q_ids(), mtp->q_p(), mtp->q_n(), tok_dev,
                            fwd->stream());
        ck(cudaMemcpyAsync(tok_host, tok_dev, size_t(2 * R) * 4, cudaMemcpyDeviceToHost, fwd->stream()), "tokens");
        ck(cudaStreamSynchronize(fwd->stream()), "sample");
        a = 0;
        while (a < R - 1 && tok_host[R + a]) ++a;
        return std::vector<int32_t>(tok_host, tok_host + a + 1);
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
bool Session::healthy() const { return m_->healthy; }

GenerateResult Session::generate(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                                 const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel) {
    Impl& m = *m_;
    if (!m.healthy) throw std::runtime_error("the engine failed earlier and must be restarted");
    const int K = m.mtp ? m.o.spec_k : 0;
    validate(r, m.s.n_vocab, m.o.max_ctx, K);
    try {
        return run(r, on_token, on_progress, cancel);
    } catch (...) {
        m.recover();
        throw;
    }
}

GenerateResult Session::run(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                            const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel) {
    Impl& m = *m_;
    GenerateResult res;
    const auto t0 = Clock::now();
    const std::vector<int32_t>& P = r.prompt;
    const int n = int(P.size());
    res.prompt_tokens = n;
    const int K = m.mtp ? m.o.spec_k : 0;

    // 1. reuse: the whole previous sequence, or the latest checkpoint inside the shared prefix, or
    // nothing
    size_t L = 0;
    while (L < m.seq.size() && L < P.size() && m.seq[L] == P[L]) ++L;
    const size_t usable = std::min(L, size_t(n - 1));   // the last prompt token always runs, for its logits
    m.drop_checkpoints_after(L);
    if (!m.seq.empty() && usable == m.seq.size()) {
        // continue from the current state
    } else if (const int c = m.restore_best(usable); c > 0) {
        m.seq.resize(size_t(c));
    } else {
        m.fwd->reset();
        if (m.mtp) m.mtp->reset();
        ck(cudaMemset(m.h_carry, 0, m.hrow * 4), "memset h");
        m.seq.clear();
    }
    res.reused = int(m.seq.size());

    // 2. prefill every new prompt token but the last in batches (reference path) or chunks, the head
    // catching up on each; host checkpoints at chunk ends and before the prompt's tail; a device
    // checkpoint at the end, so the same prompt again or one extending it reuses all of it; then
    // the last token alone, for its logits
    const int from = int(m.seq.size()), end = n - 1;
    const bool chunked = end - from >= m.o.chunk_min;
    // the tail runs in batches after the chunks, so a checkpoint can sit just before it
    const int tail = chunked && !m.ring.empty() && m.o.ckpt_tail > 0 && end - from - m.o.ckpt_tail >= m.o.chunk_min ? m.o.ckpt_tail : 0;
    int step = m.o.prefill_batch;
    if (chunked) {   // experts stream to the GPU; the expert cache's memory is lent to the chunks
        m.cache_release();
        m.fwd->set_prefill_lookahead(P.data(), end);
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        step = m.o.prefill_chunk > 0 ? m.o.prefill_chunk
                                     : m.fwd->pick_chunk(end - tail - from, end - tail, free_b, m.o.prefill_chunk_max);
        std::fprintf(stderr, "flashrt: prefill of %d tokens in chunks of %d (%zu MiB free)\n", end - from, step, free_b >> 20);
    }
    int last_ckpt = from;
    for (int p = from; p < end;) {
        const int T = p < end - tail ? std::min(step, end - tail - p) : std::min(m.o.prefill_batch, end - p);
        m.fwd->forward(P.data(), T, T, nullptr);
        m.mtp_catchup(P.data(), p, T);
        p += T;
        if (r.fail_at == 1) throw std::runtime_error("injected fault after a prefill step");
        if (p < end && ((tail > 0 && p == end - tail) || (chunked && p - last_ckpt >= m.o.ckpt_interval))) {
            m.save_host_checkpoint();
            last_ckpt = p;
        }
        if (on_progress) on_progress(p, n);
        if (cancel.load()) {
            m.seq.assign(P.begin(), P.begin() + p);
            if (chunked) m.cache_restore();
            res.finish = "cancelled";
            res.prompt_ms = ms_since(t0);
            return res;
        }
    }
    m.seq.assign(P.begin(), P.end() - 1);
    m.fwd->save_checkpoint();   // the device checkpoint: this prompt's end
    if (m.mtp) m.mtp->save_checkpoint(m.h_carry);
    if (chunked) m.cache_restore();   // refilled from the prefill's routing counts
    else if (!m.cache_filled || n - from >= 4096) m.refill_cache();
    m.fwd->forward(P.data(), 1, 0, m.logits);
    if (r.first_top) {   // tests: the state after the prompt, independent of the sampling path
        std::vector<float> lg(m.s.n_vocab);
        ck(cudaMemcpy(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost), "logits");
        std::vector<int32_t> idx(lg.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
        for (int i = 0; i < 8; ++i) res.first_top.push_back({idx[i], lg[idx[i]]});
    }
    m.mtp_catchup(P.data(), n - 1, 1);
    m.seq = P;
    if (on_progress) on_progress(n, n);
    if (m.mtp) m.set_draft_vocab(P);
    res.prompt_ms = ms_since(t0);

    // 3. decode
    const auto td = Clock::now();
    const long hits0 = m.host.hits, misses0 = m.host.misses + m.host.gpu_misses;
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
    static const bool argmax_drafts = [] {
        const char* e = std::getenv("FLASHRT_ARGMAX_DRAFTS");
        return e && e[0] == '1';
    }();
    const bool sampled = K > 0 && r.sampling.temperature > 0.0f && !argmax_drafts;
    if (K > 0) m.mtp->set_draft_sampling(sampled ? r.sampling : sample::Params{0.0f}, r.seed);
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
            if (r.fail_at == 2) throw std::runtime_error("injected fault in a decode step");
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
        if (r.fail_at == 2) throw std::runtime_error("injected fault in a verify window");
        int a = 0;
        std::vector<int32_t> ys;
        if (sampled) ys = m.spec_pick(m.logits_win, K + 1, p + 1, r, a);
        else {
            ys = m.pick(m.logits_win, K + 1, p + 1, r);
            while (a < K && ys[a] == d[a]) ++a;
        }
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
    res.cache_hits = m.host.hits - hits0;
    res.cache_misses = m.host.misses + m.host.gpu_misses - misses0;
    return res;
}

}  // namespace flashrt
