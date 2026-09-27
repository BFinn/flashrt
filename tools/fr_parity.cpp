// SPDX-License-Identifier: Apache-2.0
// fr_parity: run flashrt's GPU blocks on llama.cpp's recorded inputs (tools/ref_dump) and
// compare the outputs, block by block.
//
//   fr_parity MODEL.gguf REF.frd [test ...]      tests: hc (default), gdn, qsa, qsa_proj, moe
//
// Every step of the dump is used (-1 = the prompt batch, 0.. = decode steps). The metric is
// relative L2, ||ours - ref|| / ||ref||, per layer; the tool fails if any exceeds the tolerance.
#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "tools/frd.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

namespace {

struct Dev {
    float* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t elems) : n(elems) { cudaMalloc(&p, std::max<size_t>(elems, 1) * 4); }
    Dev(const std::vector<float>& v) : Dev(v.size()) { cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); }
    ~Dev() { cudaFree(p); }
    std::vector<float> host() const {
        std::vector<float> v(n);
        cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost);
        return v;
    }
};

double rel_l2(const std::vector<float>& a, const std::vector<float>& ref) {
    if (a.size() != ref.size()) return INFINITY;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - ref[i];
        num += d * d;
        den += double(ref[i]) * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

struct Checker {
    double tol;
    int fails = 0, checks = 0;
    double worst = 0;
    std::string worst_what;
    void check(const std::string& what, const std::vector<float>& ours, const std::vector<float>& ref, bool verbose) {
        const double e = rel_l2(ours, ref);
        ++checks;
        if (!(e <= tol)) ++fails;
        if (!(e <= worst)) { worst = e; worst_what = what; }
        if (verbose || !(e <= tol)) std::printf("  %-44s rel L2 %.3e%s\n", what.c_str(), e, e <= tol ? "" : "  FAIL");
    }
};

// [T][n] -> [T][hc][n], every stream a copy
std::vector<float> repeat_streams(const std::vector<float>& x, int T, int n, int hc) {
    std::vector<float> y(size_t(T) * hc * n);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < hc; ++s) std::memcpy(&y[(size_t(t) * hc + s) * n], &x[size_t(t) * n], size_t(n) * 4);
    return y;
}

void test_hc(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    const int n = s.d_model, hc = s.hc_count;
    std::set<int> ple(s.ple_layers.begin(), s.ple_layers.end());
    for (int step : steps) {
        const int T = int(ref.get("model.input_embed", step).ne[1]);
        std::printf("hc: step %d (%d tokens)\n", step, T);
        for (int il = 0; il < s.n_layer; ++il) {
            const std::string L = "-" + std::to_string(il);
            const bool last_prompt = step < 0 && il == s.n_layer - 1;   // llama.cpp keeps only the output rows there
            // --- attention-side mix: input is the previous layer output (or the embedding)
            if (!ple.count(il)) {
                std::vector<float> xin = il == 0 ? repeat_streams(ref.floats(ref.get("model.input_embed", step)), T, n, hc)
                                                 : ref.floats(ref.get("l_last-" + std::to_string(il - 1), step));
                Dev x(xin), mixed(size_t(T) * n), inject(size_t(T) * hc), xn(size_t(T) * hc * n);
                hc_mix(c, il, 0, x.p, T, mixed.p, inject.p, xn.p);
                cudaStreamSynchronize(c.stream);
                ck.check("hc_norm" + L + " (attn)", xn.host(), ref.floats(ref.get("hc_norm" + L, step, 0)), false);
                ck.check("hc_mixed" + L + " (attn)", mixed.host(), ref.floats(ref.get("hc_mixed" + L, step, 0)), il < 4);
                ck.check("hc_inject" + L + " (attn)", inject.host(), ref.floats(ref.get("hc_inject" + L, step, 0)), false);
                // attention-side combine with llama.cpp's own mixer output
                if (!last_prompt) {
                    const Frd::Rec* mo = ref.find(s.mixer[il] == Mixer::QSA ? "attn_output" + L : "linear_attn_out" + L, step);
                    Dev out(ref.floats(*mo)), inj(ref.floats(ref.get("hc_inject" + L, step, 0)));
                    hc_combine(c, x.p, out.p, inj.p, T);
                    cudaStreamSynchronize(c.stream);
                    ck.check("hc_combine" + L, x.host(), ref.floats(ref.get("hc_combine" + L, step)), false);
                }
            }
            // --- FFN-side mix: input is the attention-side combine
            {
                const std::vector<float> xin = ref.floats(ref.get("hc_combine" + L, step));
                const int Tl = int(xin.size() / (size_t(hc) * n));
                Dev x(xin), mixed(size_t(Tl) * n), inject(size_t(Tl) * hc), xn(size_t(Tl) * hc * n);
                hc_mix(c, il, 1, x.p, Tl, mixed.p, inject.p, xn.p);
                cudaStreamSynchronize(c.stream);
                ck.check("hc_mixed" + L + " (ffn)", mixed.host(), ref.floats(ref.get("hc_mixed" + L, step, 1)), il < 4);
                ck.check("hc_inject" + L + " (ffn)", inject.host(), ref.floats(ref.get("hc_inject" + L, step, 1)), false);
                Dev out(ref.floats(ref.get("ffn_out" + L, step)));
                hc_combine(c, x.p, out.p, inject.p, Tl);
                cudaStreamSynchronize(c.stream);
                ck.check("l_last" + L, x.host(), ref.floats(ref.get("l_last" + L, step)), false);
            }
        }
        // --- head mix
        const std::vector<float> xin = ref.floats(ref.get("l_last-" + std::to_string(s.n_layer - 1), step));
        const int Tl = int(xin.size() / (size_t(hc) * n));
        Dev x(xin), mixed(size_t(Tl) * n);
        hc_mix(c, -1, 2, x.p, Tl, mixed.p, nullptr);
        cudaStreamSynchronize(c.stream);
        ck.check("result_norm (head mix)", mixed.host(), ref.floats(ref.get("result_norm", step)), true);
    }
}

// GDN mixers: states carried across the steps in order (prompt batch, then decode steps).
void test_gdn(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    std::vector<GdnState> st(s.n_layer);
    for (int il : s.gdn_layers) st[il] = alloc_gdn_state(s);
    for (int step : steps) {
        const int T = int(ref.get("model.input_embed", step).ne[1]);
        std::printf("gdn: step %d (%d tokens)\n", step, T);
        for (int il : s.gdn_layers) {
            const std::string L = "-" + std::to_string(il);
            Dev x(ref.floats(ref.get("hc_mixed" + L, step, 0)));
            Dev out(size_t(T) * s.d_model), o(size_t(T) * s.ssm_heads * s.ssm_state);
            gdn_mixer(c, il, x.p, T, st[il], out.p, o.p);
            cudaStreamSynchronize(c.stream);
            ck.check("gdn attn_output" + L, o.host(), ref.floats(ref.get("attn_output" + L, step)), il < 3);
            ck.check("linear_attn_out" + L, out.host(), ref.floats(ref.get("linear_attn_out" + L, step)), il < 3);
        }
    }
    for (int il : s.gdn_layers) free_gdn_state(st[il]);
}

// QSA mixers: KV caches and positions carried across the steps in order.
void test_qsa(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    std::vector<QsaCache> kv(s.n_layer);
    for (int il : s.qsa_layers) kv[il] = alloc_qsa_cache(s, 4096);
    int pos = 0;
    for (int step : steps) {
        const int T = int(ref.get("model.input_embed", step).ne[1]);
        std::printf("qsa: step %d (%d tokens at %d)\n", step, T, pos);
        for (int il : s.qsa_layers) {
            const std::string L = "-" + std::to_string(il);
            Dev x(ref.floats(ref.get("hc_mixed" + L, step, 0)));
            Dev out(size_t(T) * s.d_model);
            qsa_mixer(c, il, x.p, T, pos, kv[il], out.p);
            cudaStreamSynchronize(c.stream);
            std::vector<float> want = ref.floats(ref.get("attn_output" + L, step));
            std::vector<float> got = out.host();
            if (want.size() != got.size()) {   // last layer of a prompt batch keeps the output row only
                got.erase(got.begin(), got.end() - want.size());
            }
            ck.check("qsa attn_output" + L, got, want, il < 12);
        }
        pos += T;
    }
    for (int il : s.qsa_layers) free_qsa_cache(kv[il]);
}

// Each QSA projection alone, on llama.cpp's exact inputs (needs a dump with the attention
// intermediates: Qcur_full, Kcur, Vcur, attn_gated).
void test_qsa_proj(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    for (int step : steps) {
        for (int il : s.qsa_layers) {
            const std::string L = "-" + std::to_string(il);
            if (!ref.find("Qcur_full" + L, step)) continue;
            const Frd::Rec& xr = ref.get("hc_mixed" + L, step, 0);
            const int T = int(xr.ne[1]);
            Dev x(ref.floats(xr)), q(size_t(T) * s.n_head * 2 * s.head_dim_k), k(size_t(T) * s.n_head_kv * s.head_dim_k),
                v(size_t(T) * s.n_head_kv * s.head_dim_k);
            linear(c, c.w.layer(il, "attn_q.weight"), x.p, q.p, T);
            linear(c, c.w.layer(il, "attn_k.weight"), x.p, k.p, T);
            linear(c, c.w.layer(il, "attn_v.weight"), x.p, v.p, T);
            const Frd::Rec& ag = ref.get("attn_gated" + L, step);
            Dev g(ref.floats(ag)), o(size_t(T) * s.d_model);
            linear(c, c.w.layer(il, "attn_output.weight"), g.p, o.p, T);
            cudaStreamSynchronize(c.stream);
            ck.check("Qcur_full" + L, q.host(), ref.floats(ref.get("Qcur_full" + L, step)), true);
            ck.check("Kcur (raw)" + L, k.host(), ref.floats(ref.get("Kcur" + L, step, 0)), true);
            ck.check("Vcur" + L, v.host(), ref.floats(ref.get("Vcur" + L, step, 0)), true);
            ck.check("attn_output from attn_gated" + L, o.host(), ref.floats(ref.get("attn_output" + L, step)), true);
        }
    }
}

// MoE blocks: routing (exact top-k ids and probabilities) and outputs, with every routed
// expert computed by the CPU miss path from the host arena (loaded here: 32 GB of RAM).
void test_moe(const BlockCtx& c, const Gguf& g, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) throw std::runtime_error("arena allocation failed");
    const LoadStats ls = load_experts(g, s, arena, 12);
    std::printf("moe: expert arena loaded in %.1f s\n", ls.seconds);
    CpuPool pool(8, physical_cpus());
    MoeHost h;
    h.arena = &arena;
    h.pool = &pool;
    long id_total = 0, id_same = 0;
    for (int step : steps) {
        std::printf("moe: step %d\n", step);
        for (int il = 0; il < s.n_layer; ++il) {
            const std::string L = "-" + std::to_string(il);
            const Frd::Rec& xr = ref.get("hc_mixed" + L, step, 1);
            const int T = int(xr.ne[1]);
            Dev x(ref.floats(xr)), out(size_t(T) * s.d_model);
            MoeTrace tr;
            moe_block(c, il, x.p, T, h, out.p, &tr);
            const std::vector<int32_t> want_ids = ref.ints(ref.get("ffn_moe_topk" + L, step));
            // compare the selected sets per token (order within the top-k may differ on near ties)
            for (int t = 0; t < T; ++t) {
                std::vector<int32_t> a(tr.topk.begin() + size_t(t) * s.top_k, tr.topk.begin() + size_t(t + 1) * s.top_k);
                std::vector<int32_t> b(want_ids.begin() + size_t(t) * s.top_k, want_ids.begin() + size_t(t + 1) * s.top_k);
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());
                id_total += s.top_k;
                for (int k = 0; k < s.top_k; ++k) id_same += std::binary_search(b.begin(), b.end(), a[k]);
            }
            ck.check("ffn_moe_weights (probs)" + L, tr.probs, ref.floats(ref.get("ffn_moe_weights" + L, step)), false);
            ck.check("ffn_out (moe + shexp)" + L, out.host(), ref.floats(ref.get("ffn_out" + L, step)), il < 3);
        }
    }
    std::printf("moe: %ld of %ld routed experts identical (%.4f%%)\n", id_same, id_total, 100.0 * id_same / std::max(id_total, 1L));
    arena_free(arena);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: fr_parity MODEL.gguf REF.frd [hc]\n");
        return 2;
    }
    std::vector<std::string> tests;
    for (int i = 3; i < argc; ++i) tests.push_back(argv[i]);
    if (tests.empty()) tests.push_back("hc");

    const Gguf g = Gguf::open(argv[1]);
    const Spec s = parse(g);
    const WeightPlan plan = qwen4exp::plan(g, s);
    GpuWeights w;
    w.load(g, plan);
    const Frd ref(argv[2]);
    std::vector<int> steps{-1};
    for (int st = 0; ref.find("model.input_embed", st); ++st) steps.push_back(st);

    const int max_t = int(ref.get("model.input_embed", -1).ne[1]);
    BlockScratch scratch = alloc_block_scratch(s, max_t);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const BlockCtx c{s, w, scratch, stream};

    Checker ck{2e-3};
    for (const std::string& t : tests) {
        if (t == "hc") test_hc(c, ref, steps, ck);
        else if (t == "gdn") test_gdn(c, ref, steps, ck);
        else if (t == "qsa") test_qsa(c, ref, steps, ck);
        else if (t == "qsa_proj") test_qsa_proj(c, ref, steps, ck);
        else if (t == "moe") test_moe(c, g, ref, steps, ck);
        else { std::fprintf(stderr, "unknown test %s\n", t.c_str()); return 2; }
    }
    std::printf("fr_parity: %d checks, %d over tolerance %.0e; worst %.3e (%s)\n", ck.checks, ck.fails, ck.tol, ck.worst,
                ck.worst_what.c_str());
    free_block_scratch(scratch);
    return ck.fails ? 1 : 0;
}
