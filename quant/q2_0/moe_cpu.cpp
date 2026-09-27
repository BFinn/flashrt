// SPDX-License-Identifier: Apache-2.0
#include "quant/q2_0/moe_cpu.hpp"

#include <atomic>
#include <cmath>
#include <cstring>

namespace flashrt::q2_0 {

namespace {

constexpr size_t kAlign = 64;
size_t up64(size_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }
constexpr int kChunkA = 32;   // hidden rows per phase-A item: one Q8 block
constexpr int kChunkB = 64;   // output rows per phase-B item

struct Ctx {
    ExpertShape s;
    const Miss* miss;
    int n_miss;
    const Q8Act* x;
    int n_tok_window;
    float* out;
    int ldo;
    Q8Act* h;           // [n_miss][4] hidden activations
    uint8_t* worker_mem;
    size_t worker_bytes;
    decltype(&matvec_avx512) mv;
    CpuPool* pool;
    alignas(64) std::atomic<int> next_a{0};
    alignas(64) std::atomic<int> next_b{0};
};

size_t worker_bytes(ExpertShape s) { return 2 * up64(size_t(4) * s.d_ff * 4) + up64(size_t(4) * s.d_model * 4); }

void run_worker(void* p, int w, int) {
    Ctx& c = *static_cast<Ctx*>(p);
    const ExpertShape s = c.s;
    uint8_t* mem = c.worker_mem + size_t(w) * c.worker_bytes;
    float* g = reinterpret_cast<float*>(mem);                                   // [4][d_ff]
    float* u = reinterpret_cast<float*>(mem + up64(size_t(4) * s.d_ff * 4));    // [4][d_ff]
    float* t = reinterpret_cast<float*>(mem + 2 * up64(size_t(4) * s.d_ff * 4));  // [4][d_model]

    // phase A: gate, up, SwiGLU and Q8 of one 32-row block of one miss
    const int na = s.d_ff / kChunkA;
    for (int k; (k = c.next_a.fetch_add(1, std::memory_order_relaxed)) < c.n_miss * na;) {
        const int i = k / na, r0 = (k % na) * kChunkA;
        const Miss& m = c.miss[i];
        const Expert e = expert_view(m.blob, s);
        Q8Act xa[4];
        for (int j = 0; j < m.n_tok; ++j) xa[j] = c.x[m.tok[j]];
        c.mv(e.gate, xa, m.n_tok, r0, r0 + kChunkA, g, s.d_ff);
        c.mv(e.up, xa, m.n_tok, r0, r0 + kChunkA, u, s.d_ff);
        for (int j = 0; j < m.n_tok; ++j) {
            float h32[kChunkA];
            for (int r = 0; r < kChunkA; ++r) {
                const float gv = g[j * s.d_ff + r0 + r];
                h32[r] = gv / (1.0f + std::exp(-gv)) * u[j * s.d_ff + r0 + r];
            }
            quantize_q8_block(h32, c.h[i * 4 + j], r0 / kChunkA);
        }
    }
    c.pool->barrier();

    // phase B: 64 output rows, summed over every miss
    const int nb = s.d_model / kChunkB;
    for (int k; (k = c.next_b.fetch_add(1, std::memory_order_relaxed)) < nb;) {
        const int r0 = k * kChunkB;
        for (int tt = 0; tt < c.n_tok_window; ++tt) std::memset(c.out + size_t(tt) * c.ldo + r0, 0, kChunkB * 4);
        for (int i = 0; i < c.n_miss; ++i) {
            const Miss& m = c.miss[i];
            const Expert e = expert_view(m.blob, s);
            c.mv(e.down, c.h + i * 4, m.n_tok, r0, r0 + kChunkB, t, s.d_model);
            for (int j = 0; j < m.n_tok; ++j) {
                float* o = c.out + size_t(m.tok[j]) * c.ldo;
                const float wj = m.w[j];
                for (int r = r0; r < r0 + kChunkB; ++r) o[r] += wj * t[j * s.d_model + r];
            }
        }
    }
}

}  // namespace

size_t moe_cpu_scratch_bytes(ExpertShape s, int max_miss, int n_workers) {
    return kAlign + up64(sizeof(Ctx)) + up64(size_t(max_miss) * 4 * sizeof(Q8Act)) +
           size_t(max_miss) * 4 * q8_bytes(s.d_ff) + size_t(n_workers) * worker_bytes(s);
}

void moe_cpu(CpuPool& pool, ExpertShape s, const Miss* miss, int n_miss, const Q8Act* x, int n_tok_window, float* out,
             int ldo, void* scratch) {
    auto p = (reinterpret_cast<uintptr_t>(scratch) + kAlign - 1) & ~uintptr_t(kAlign - 1);
    Ctx* c = new (reinterpret_cast<void*>(p)) Ctx();
    p += up64(sizeof(Ctx));
    c->h = reinterpret_cast<Q8Act*>(p);
    p += up64(size_t(n_miss) * 4 * sizeof(Q8Act));
    for (int i = 0; i < n_miss * 4; ++i) {
        c->h[i] = q8_view(reinterpret_cast<void*>(p), s.d_ff);
        p += q8_bytes(s.d_ff);
    }
    c->worker_mem = reinterpret_cast<uint8_t*>(p);
    c->worker_bytes = worker_bytes(s);
    c->s = s;
    c->miss = miss;
    c->n_miss = n_miss;
    c->x = x;
    c->n_tok_window = n_tok_window;
    c->out = out;
    c->ldo = ldo;
    c->mv = have_avx512() ? matvec_avx512 : matvec_ref;
    c->pool = &pool;
    pool.run(run_worker, c);
    c->~Ctx();
}

}  // namespace flashrt::q2_0
