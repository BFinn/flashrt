// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/moe_fast.hpp"

#include "kernels/cuda/ggml_gemv.h"
#include "quant/q2_0/moe_cpu.hpp"

#include "core/platform.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace flashrt::qwen4exp {

namespace {

constexpr int kMaxK = 16;   // top-k upper bound for the fixed launch shapes

// mailbox layout (one per layer, doorbell mode)
constexpr size_t kMbRouted = 0;     // uint32, GPU -> host: routing and x are in place for token seq
constexpr size_t kMbDone = 64;      // uint32, host -> GPU: out is in place for token seq
constexpr size_t kMbRoute = 128;    // int32 [2 + 2K], as route_dev
constexpr size_t kMbX = 512;        // float [d_model]
size_t mb_out_off(int n) { return (kMbX + size_t(n) * 4 + 63) & ~size_t(63); }   // float [d_model]

__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// planar repacked matrix (codes [rows][nb][16], scales [rows][nb]) -> ggml Q2_0 blocks
// {fp16 d, qs[16]}: qs[k] holds elements 4k..4k+3; planar element j is in byte j%16 at bit 2*(j/16)
__global__ void k_unrepack(const uint8_t* src, uint8_t* dst, size_t nblk) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint8_t* p = src + i * 16;
    const uint16_t d = reinterpret_cast<const uint16_t*>(src + nblk * 16)[i];
    uint8_t out[18];
    out[0] = uint8_t(d & 0xff);
    out[1] = uint8_t(d >> 8);
    for (int k = 0; k < 16; ++k) {
        uint8_t v = 0;
        for (int e = 0; e < 4; ++e) {
            const int j = 4 * k + e;
            v |= uint8_t(((p[j % 16] >> (2 * (j / 16))) & 3) << (2 * e));
        }
        out[2 + k] = v;
    }
    for (int b = 0; b < 18; ++b) dst[i * 18 + b] = out[b];
}

// Exclusive prefix count of flag over the block (blockDim.x a multiple of 32); *total = all.
__device__ int block_scan_flags(bool flag, int* total) {
    __shared__ int warp_tot[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned bal = __ballot_sync(0xffffffff, flag);
    const int in_warp = __popc(bal & ((1u << lane) - 1));
    if (lane == 0) warp_tot[warp] = __popc(bal);
    __syncthreads();
    if (warp == 0) {
        int v = lane < nw ? warp_tot[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffff, v, o);
            if (lane >= o) v += n;
        }
        if (lane < nw) warp_tot[lane] = v;
    }
    __syncthreads();
    const int before = warp > 0 ? warp_tot[warp - 1] : 0;
    *total = warp_tot[nw - 1];
    __syncthreads();
    return before + in_warp;
}

// One block of E threads (E <= 1024): softmax over the router logits, top-k by probability,
// weights renormalised (sum clamped at 6.1e-5, as llama.cpp). Hits get their slot; the hit list
// is padded to k with a real slot (or slot 0) at weight 0 so the grouped launches have a fixed
// shape. route_host gets [n_hits, n_miss, miss experts[k], miss weights[k] as float bits].
// Doorbell mode (mb != nullptr): the record goes to the mailbox instead, followed by x [n], and
// then the routed flag is raised to seq.
__global__ void k_route(const float* logits, const int32_t* table, int E, int K, int32_t* hit_slot, float* hit_w,
                        int32_t* route_dev, const float* x, int n, uint8_t* mb, uint32_t seq) {
    if (mb) route_dev = reinterpret_cast<int32_t*>(mb + kMbRoute);
    __shared__ float p[1024];
    __shared__ float red_v[32];
    __shared__ int sel[kMaxK];
    __shared__ float selp[kMaxK];
    __shared__ int sel_slot[kMaxK];
    __shared__ int cand_i[1024];
    __shared__ float cand_p[1024];
    const int e = threadIdx.x;
    const float lg = e < E ? logits[e] : -INFINITY;
    const int my_slot = e < E ? table[e] : -1;   // loaded early, used if e is selected
    // max
    float m = lg;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((e & 31) == 0) red_v[e >> 5] = m;
    __syncthreads();
    if (e < 32) {
        float v = e < (blockDim.x + 31) / 32 ? red_v[e] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (e == 0) red_v[0] = v;
    }
    __syncthreads();
    m = red_v[0];
    __syncthreads();
    const float ex = e < E ? __expf(lg - m) : 0.0f;
    float sum = ex;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((e & 31) == 0) red_v[e >> 5] = sum;
    __syncthreads();
    if (e < 32) {
        float v = e < (blockDim.x + 31) / 32 ? red_v[e] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (e == 0) red_v[0] = v;
    }
    __syncthreads();
    const float total = red_v[0];
    __syncthreads();
    p[e] = e < E ? ex / total : -1.0f;
    __syncthreads();
    // top-k by rank: expert e's rank is the number of experts ahead of it (higher probability,
    // or equal and a lower index), so the order is that of k rounds of argmax, lowest index first.
    // Only candidates can be in the top k: p >= the k-th largest warp maximum (k warps have an
    // element at least that large). They are compacted, and ranked among themselves.
    const float pe = p[e];
    {
        float wm = pe;
        for (int o = 16; o > 0; o >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffff, wm, o));
        if ((e & 31) == 0) red_v[e >> 5] = wm;
    }
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float thr = -1.0f;
    if (K <= nw) {   // the K-th largest of the warp maxima
        for (int w = 0; w < nw; ++w) {
            const float v = red_v[w];
            int ahead = 0;
            for (int u = 0; u < nw; ++u) ahead += (red_v[u] > v) | ((red_v[u] == v) & (u < w));
            if (ahead == K - 1) thr = v;
        }
    }
    const bool cand = e < E && pe >= thr;
    int n_cand;
    const int ci = block_scan_flags(cand, &n_cand);
    if (cand) {
        cand_i[ci] = e;
        cand_p[ci] = pe;
    }
    __syncthreads();
    if (cand) {
        int rank = 0;
        for (int j = 0; j < n_cand; ++j) {
            const float pj = cand_p[j];
            rank += (pj > pe) | ((pj == pe) & (cand_i[j] < e));
        }
        if (rank < K) {
            sel[rank] = e;
            selp[rank] = pe;
            sel_slot[rank] = my_slot;
        }
    }
    __syncthreads();
    if (e == 0) {
        float ws = 0.0f;
        for (int k = 0; k < K; ++k) ws += selp[k];
        ws = fmaxf(ws, 6.103515625e-5f);
        int nh = 0, nm = 0;
        int pad = 0;
        for (int k = 0; k < K; ++k) {
            const int slot = sel_slot[k];
            const float w = selp[k] / ws;
            if (slot >= 0) {
                hit_slot[nh] = slot;
                hit_w[nh] = w;
                pad = slot;
                ++nh;
            } else {
                route_dev[2 + nm] = sel[k];
                reinterpret_cast<float*>(route_dev)[2 + K + nm] = w;
                ++nm;
            }
        }
        for (int k = nh; k < K; ++k) {
            hit_slot[k] = pad;
            hit_w[k] = 0.0f;
        }
        route_dev[0] = nh;
        route_dev[1] = nm;
    }
    if (mb) {
        float* xh = reinterpret_cast<float*>(mb + kMbX);
        for (int i = e; i < n; i += blockDim.x) xh[i] = x[i];
        __threadfence_system();
        __syncthreads();
        if (e == 0) *reinterpret_cast<volatile uint32_t*>(mb + kMbRouted) = seq;
    }
}

// out = sum_k hit_w[k] * yh[k] + cpu + shexp * sigmoid(gate)
__global__ void k_moe_combine(float* out, const float* yh, const float* hit_w, int K, const float* cpu, const float* shexp,
                              const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = cpu[i];
    for (int k = 0; k < K; ++k) acc += hit_w[k] * yh[size_t(k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[0]));
    out[i] = acc + shexp[i] * g;
}

// Doorbell mode: as k_moe_combine, with the CPU part read from the mailbox once the host has
// raised its done flag to seq (traps after 10 s, so a dead miss server fails loudly).
__global__ void k_moe_combine_db(float* out, const float* yh, const float* hit_w, int K, const uint8_t* mb, size_t out_off,
                                 uint32_t seq, const float* shexp, const float* gate, int n) {
    if (threadIdx.x == 0) {
        const volatile uint32_t* done = reinterpret_cast<const volatile uint32_t*>(mb + kMbDone);
        const uint64_t t0 = global_ns();
        while (*done != seq) {
            __nanosleep(128);
            if (global_ns() - t0 > 10000000000ull) __trap();
        }
        __threadfence_system();
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = reinterpret_cast<const volatile float*>(mb + out_off)[i];
    for (int k = 0; k < K; ++k) acc += hit_w[k] * yh[size_t(k) * n + i];
    const float g = 1.0f / (1.0f + __expf(-gate[0]));
    out[i] = acc + shexp[i] * g;
}

__global__ void k_swiglu_1(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

}  // namespace

ExpertCache alloc_expert_cache(const Spec& s, int n_slots) {
    ExpertCache c;
    c.n_slots = n_slots;
    c.slot_bytes = q2_0::expert_bytes({s.d_model, s.d_ff_expert});
    ck(cudaMalloc(&c.slots, size_t(n_slots) * c.slot_bytes + gemv::kWeightTailPad), "cudaMalloc expert cache");
    ck(cudaMemset(c.slots, 0, size_t(n_slots) * c.slot_bytes + gemv::kWeightTailPad), "memset expert cache");
    const size_t tn = size_t(s.n_layer) * s.n_expert;
    ck(cudaMalloc(&c.table_dev, tn * 4), "cudaMalloc slot table");
    c.table.assign(tn, -1);
    ck(cudaMemcpy(c.table_dev, c.table.data(), tn * 4, cudaMemcpyHostToDevice), "slot table");
    c.owner.assign(n_slots, -1);
    ck(cudaMalloc(&c.staging, c.slot_bytes), "cudaMalloc staging");
    return c;
}

void free_expert_cache(ExpertCache& c) {
    if (c.slots) cudaFree(c.slots);
    if (c.table_dev) cudaFree(c.table_dev);
    if (c.staging) cudaFree(c.staging);
    c = ExpertCache{};
}

void expert_cache_fill(const Spec& s, ExpertCache& cache, const ExpertArena& arena,
                       const std::vector<std::pair<int, int>>& order, cudaStream_t stream) {
    const q2_0::ExpertShape es{s.d_model, s.d_ff_expert};
    const size_t mats[3] = {q2_0::mat_bytes(es.d_ff, es.d_model), q2_0::mat_bytes(es.d_ff, es.d_model),
                            q2_0::mat_bytes(es.d_model, es.d_ff)};
    int slot = 0;
    for (const auto& [l, e] : order) {
        while (slot < cache.n_slots && cache.owner[slot] >= 0) ++slot;
        if (slot >= cache.n_slots) break;
        if (cache.table[size_t(l) * s.n_expert + e] >= 0) continue;
        ck(cudaMemcpyAsync(cache.staging, arena.blob(l, e), cache.slot_bytes, cudaMemcpyHostToDevice, stream), "upload expert");
        size_t off = 0;
        for (size_t m : mats) {
            const size_t nblk = m / 18;
            k_unrepack<<<unsigned((nblk + 255) / 256), 256, 0, stream>>>(cache.staging + off,
                                                                      cache.slots + size_t(slot) * cache.slot_bytes + off, nblk);
            off += m;
        }
        cache.owner[slot] = l * s.n_expert + e;
        cache.table[size_t(l) * s.n_expert + e] = slot;
        ++slot;
    }
    ck(cudaMemcpyAsync(cache.table_dev, cache.table.data(), cache.table.size() * 4, cudaMemcpyHostToDevice, stream), "slot table");
    ck(cudaStreamSynchronize(stream), "expert_cache_fill");
}

MoeFastHost alloc_moe_fast_host(const Spec& s) {
    MoeFastHost h;
    ck(cudaHostAlloc(&h.route_host, size_t(2 + 2 * kMaxK) * 4, cudaHostAllocDefault), "cudaHostAlloc route");
    ck(cudaHostAlloc(&h.x_host, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc x");
    ck(cudaHostAlloc(&h.cpu_out, size_t(s.d_model) * 4, cudaHostAllocDefault), "cudaHostAlloc cpu out");
    ck(cudaEventCreateWithFlags(&h.routed, cudaEventDisableTiming), "cudaEventCreate");
    return h;
}

// ---- doorbell mode: the miss server

struct MissServer {
    const Spec* s = nullptr;
    MoeFastHost* h = nullptr;
    int cpu = -1;
    std::thread th;
    alignas(64) std::atomic<uint32_t> post{0};   // latest token to serve
    std::atomic<bool> quit{false};
    std::atomic<bool> failed{false};
    std::string error;

    void run() {
        if (cpu >= 0) pin_current_thread(cpu);
        const Spec& sp = *s;
        const int n = sp.d_model, K = sp.top_k;
        const q2_0::ExpertShape es{n, sp.d_ff_expert};
        std::vector<uint8_t> act_mem(q2_0::q8_bytes(n) + 64);
        q2_0::Q8Act act =
            q2_0::q8_view(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(act_mem.data()) + 63) & ~uintptr_t(63)), n);
        std::vector<uint8_t> scratch(q2_0::moe_cpu_scratch_bytes(es, kMaxK, h->pool->size()));
        q2_0::Miss miss[kMaxK];
        const size_t out_off = mb_out_off(n);
        uint32_t seen = post.load();
        for (;;) {
            uint32_t seq;
            while ((seq = post.load(std::memory_order_acquire)) == seen) {
                if (quit.load()) return;
                post.wait(seen, std::memory_order_acquire);
            }
            seen = seq;
            for (int il = 0; il < sp.n_layer; ++il) {
                uint8_t* mb = h->mbox + size_t(il) * h->mbox_stride;
                const auto t0 = std::chrono::steady_clock::now();
                int k = 0;
                while (__atomic_load_n(reinterpret_cast<uint32_t*>(mb + kMbRouted), __ATOMIC_ACQUIRE) != seq) {
#if defined(__x86_64__)
                    _mm_pause();
#endif
                    if (++k == 4096) {
                        k = 0;
                        if (quit.load()) return;
                        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
                            error = "miss server: no routing from the GPU for layer " + std::to_string(il) + " in 10 s";
                            failed.store(true);
                            return;
                        }
                    }
                }
                const auto t1 = std::chrono::steady_clock::now();
                const int32_t* route = reinterpret_cast<const int32_t*>(mb + kMbRoute);
                const int nh = route[0], nm = route[1];
                float* out = reinterpret_cast<float*>(mb + out_off);
                if (nm > 0) {
                    q2_0::quantize_q8(reinterpret_cast<const float*>(mb + kMbX), act);
                    const float* mw = reinterpret_cast<const float*>(route + 2 + K);
                    for (int i = 0; i < nm; ++i) miss[i] = q2_0::Miss{h->arena->blob(il, route[2 + i]), 1, {0}, {mw[i]}};
                    q2_0::moe_cpu(*h->pool, es, miss, nm, &act, 1, out, n, scratch.data());
                } else {
                    std::memset(out, 0, size_t(n) * 4);
                }
                const auto t2 = std::chrono::steady_clock::now();
                h->hits += nh;
                h->misses += nm;
                h->wait_s += std::chrono::duration<double>(t1 - t0).count();
                const double dt = std::chrono::duration<double>(t2 - t1).count();
                h->cpu_s += dt;
                h->cpu_by_nm[std::min(nm, 16)] += dt;
                ++h->layers_by_nm[std::min(nm, 16)];
                __atomic_store_n(reinterpret_cast<uint32_t*>(mb + kMbDone), seq, __ATOMIC_RELEASE);
            }
        }
    }
};

void start_doorbell(const Spec& s, MoeFastHost& h, int cpu) {
    if (s.top_k > kMaxK) throw std::runtime_error("start_doorbell: top-k too large");
    h.mbox_stride = (mb_out_off(s.d_model) + size_t(s.d_model) * 4 + 4095) & ~size_t(4095);
    const size_t bytes = size_t(s.n_layer) * h.mbox_stride;
    ck(cudaHostAlloc(&h.mbox, bytes, cudaHostAllocMapped), "cudaHostAlloc mailboxes");
    std::memset(h.mbox, 0, bytes);
    ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h.mbox_dev), h.mbox, 0), "mailbox device pointer");
    h.seq = 0;
    h.server = new MissServer;
    h.server->s = &s;
    h.server->h = &h;
    h.server->cpu = cpu;
    h.server->th = std::thread([srv = h.server] { srv->run(); });
    h.doorbell = true;
}

void doorbell_begin_token(MoeFastHost& h) {
    ++h.seq;
    h.server->post.store(h.seq, std::memory_order_release);
    h.server->post.notify_one();
}

void doorbell_end_token(MoeFastHost& h) {
    if (h.server->failed.load()) throw std::runtime_error(h.server->error);
}

void free_moe_fast_host(MoeFastHost& h) {
    if (h.server) {
        h.server->quit.store(true);
        h.server->post.fetch_add(0x80000000u);
        h.server->post.notify_one();
        h.server->th.join();
        delete h.server;
    }
    if (h.mbox) cudaFreeHost(h.mbox);
    if (h.route_host) cudaFreeHost(h.route_host);
    if (h.x_host) cudaFreeHost(h.x_host);
    if (h.cpu_out) cudaFreeHost(h.cpu_out);
    if (h.routed) cudaEventDestroy(h.routed);
    h = MoeFastHost{};
}

void moe_block_fast(const BlockCtx& c, int il, const float* x, const ExpertCache& cache, MoeFastHost& h, float* out) {
    const Spec& s = c.s;
    const int n = s.d_model, E = s.n_expert, K = s.top_k, ff = s.d_ff_expert, ffs = s.d_ff_shared;
    if (K > kMaxK || E > 1024) throw std::runtime_error("moe_block_fast: unsupported routing shape");
    // device scratch layout
    float* logits = c.scratch.f32;                 // [E]
    float* sg = logits + E;                        // [ffs]
    float* su = sg + ffs;                          // [ffs]
    float* sh = su + ffs;                          // [n]
    float* gate = sh + n;                          // [1] (padded to 4)
    float* hbuf = gate + 4;                        // [K][ff]
    float* yh = hbuf + size_t(K) * ff;             // [K][n]
    float* cpu_dev = yh + size_t(K) * n;           // [n]
    float* hit_w = cpu_dev + n;                    // [K]
    int32_t* hit_slot = reinterpret_cast<int32_t*>(hit_w + kMaxK);     // [K]
    int32_t* route_dev = hit_slot + kMaxK;                              // [2 + 2K]
    uint8_t* xq = static_cast<uint8_t*>(c.scratch.q8);                  // Q8_1 of x
    uint8_t* hq = xq + gemv::q8_1_bytes(n, 1);                          // Q8_1 of the K hidden rows
    if (size_t(reinterpret_cast<float*>(route_dev + 2 + 2 * kMaxK) - c.scratch.f32) > c.scratch.f32_elems ||
        gemv::q8_1_bytes(n, 1) + gemv::q8_1_bytes(ff, K) > c.scratch.q8_bytes)
        throw std::runtime_error("moe_block_fast: scratch too small");

    // 1. routing, and the input for the CPU
    linear(c, c.w.layer(il, "ffn_gate_inp.weight"), x, logits, 1);
    uint8_t* mb = h.doorbell ? h.mbox_dev + size_t(il) * h.mbox_stride : nullptr;
    k_route<<<1, ((E + 31) / 32) * 32, 0, c.stream>>>(logits, cache.table_dev + size_t(il) * E, E, K, hit_slot, hit_w, route_dev,
                                                     x, n, mb, h.seq);
    if (!h.doorbell) {
        ck(cudaMemcpyAsync(h.route_host, route_dev, size_t(2 + 2 * K) * 4, cudaMemcpyDeviceToHost, c.stream), "route to host");
        ck(cudaMemcpyAsync(h.x_host, x, size_t(n) * 4, cudaMemcpyDeviceToHost, c.stream), "x to host");
        ck(cudaEventRecord(h.routed, c.stream), "event");
    }

    // 2. GPU: cache hits (fused gate+up, then down) and the shared expert
    const uint8_t* slots = cache.slots;
    const size_t gu = q2_0::mat_bytes(ff, n);
    gemv::quantize_q8_1(x, n, 1, xq, c.stream);
    gemv::moe_q2_0(slots + gu, slots, xq, hit_slot, hbuf, K, n, ff, int64_t(cache.slot_bytes), false, c.stream);
    gemv::quantize_q8_1(hbuf, ff, K, hq, c.stream);
    gemv::moe_q2_0(slots + 2 * gu, nullptr, hq, hit_slot, yh, K, ff, n, int64_t(cache.slot_bytes), true, c.stream);
    linear(c, c.w.layer(il, "ffn_gate_shexp.weight"), x, sg, 1);
    linear(c, c.w.layer(il, "ffn_up_shexp.weight"), x, su, 1);
    k_swiglu_1<<<(ffs + 255) / 256, 256, 0, c.stream>>>(sg, su, ffs);
    linear(c, c.w.layer(il, "ffn_down_shexp.weight"), sg, sh, 1);
    linear(c, c.w.layer(il, "ffn_gate_inp_shexp.weight"), x, gate, 1);

    if (h.doorbell) {   // 3'. the miss server fills the mailbox; the combine waits for it on the GPU
        k_moe_combine_db<<<(n + 255) / 256, 256, 0, c.stream>>>(out, yh, hit_w, K, mb, mb_out_off(n), h.seq, sh, gate, n);
        ck(cudaGetLastError(), "moe_block_fast");
        return;
    }

    // 3. CPU: the misses, while the GPU works
    const auto t0 = std::chrono::steady_clock::now();
    ck(cudaEventSynchronize(h.routed), "wait routing");
    const auto t1 = std::chrono::steady_clock::now();
    const int nh = h.route_host[0], nm = h.route_host[1];
    h.hits += nh;
    h.misses += nm;
    if (nm > 0) {
        const q2_0::ExpertShape es{n, ff};
        const size_t qb = q2_0::q8_bytes(n);
        h.act_mem.resize(qb + 64);
        q2_0::Q8Act act = q2_0::q8_view(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(h.act_mem.data()) + 63) & ~uintptr_t(63)), n);
        q2_0::quantize_q8(h.x_host, act);
        std::vector<q2_0::Miss> miss(nm);
        const float* mw = reinterpret_cast<const float*>(h.route_host + 2 + K);
        for (int i = 0; i < nm; ++i) miss[i] = q2_0::Miss{h.arena->blob(il, h.route_host[2 + i]), 1, {0}, {mw[i]}};
        h.scratch.resize(q2_0::moe_cpu_scratch_bytes(es, nm, h.pool->size()));
        q2_0::moe_cpu(*h.pool, es, miss.data(), nm, &act, 1, h.cpu_out, n, h.scratch.data());
    } else {
        std::memset(h.cpu_out, 0, size_t(n) * 4);
    }
    h.wait_s += std::chrono::duration<double>(t1 - t0).count();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    h.cpu_s += dt;
    h.cpu_by_nm[std::min(nm, 16)] += dt;
    ++h.layers_by_nm[std::min(nm, 16)];
    ck(cudaMemcpyAsync(cpu_dev, h.cpu_out, size_t(n) * 4, cudaMemcpyHostToDevice, c.stream), "cpu result to device");
    k_moe_combine<<<(n + 255) / 256, 256, 0, c.stream>>>(out, yh, hit_w, K, cpu_dev, sh, gate, n);
    ck(cudaGetLastError(), "moe_block_fast");
}

}  // namespace flashrt::qwen4exp
