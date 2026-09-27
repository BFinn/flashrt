// SPDX-License-Identifier: Apache-2.0
// fr_parity: run flashrt's GPU blocks on llama.cpp's recorded inputs (tools/ref_dump) and
// compare the outputs, block by block.
//
//   fr_parity MODEL.gguf REF.frd [test ...]      tests: hc (default), gdn, qsa, qsa_proj, moe, ple, head, full
//
// Every step of the dump is used (-1 = the prompt batch, 0.. = decode steps). The metric is
// relative L2, ||ours - ref|| / ||ref||, per layer; the tool fails if any exceeds the tolerance.
#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/experts.hpp"
#include "quant/q2_0/q2_0.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "tools/frd.hpp"
#include "arch/qwen4exp/ple.hpp"
#include "core/row_reader.hpp"

#include <fstream>
#include <iterator>
#include <sstream>

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

// QSA mixers: KV caches, indexer state and positions carried across the steps in order. Where
// llama.cpp selected cells (indexer_top_k; long contexts), the selections are compared too.
void test_qsa(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    const Spec& s = c.s;
    std::vector<QsaCache> kv(s.n_layer);
    int total_tokens = 0;
    for (int step : steps) total_tokens += int(ref.get("hc_mixed-" + std::to_string(s.qsa_layers[0]), step, 0).ne[1]);
    for (int il : s.qsa_layers) kv[il] = alloc_qsa_cache(s, total_tokens + 16);
    int pos = 0;
    long sel_steps = 0, sel_cells = 0, sel_same = 0;
    double sel_worst = 1.0;
    for (int step : steps) {
        const int T = int(ref.get("hc_mixed-" + std::to_string(s.qsa_layers[0]), step, 0).ne[1]);
        const bool verbose = steps.size() < 100 || step % 256 == 0 || step >= int(steps.size()) - 4;
        if (verbose) std::printf("qsa: step %d (%d tokens at %d)\n", step, T, pos);
        for (int il : s.qsa_layers) {
            const std::string L = "-" + std::to_string(il);
            Dev x(ref.floats(ref.get("hc_mixed" + L, step, 0)));
            Dev out(size_t(T) * s.d_model), gated(size_t(T) * s.n_head * s.head_dim_k);
            std::vector<std::vector<int32_t>> sel;
            qsa_mixer(c, il, x.p, T, pos, kv[il], out.p, gated.p, &sel);
            cudaStreamSynchronize(c.stream);
            if (const Frd::Rec* ag = ref.find("attn_gated" + L, step))
                ck.check("qsa attn_gated" + L, gated.host(), ref.floats(*ag), verbose && il < 12);
            std::vector<float> want = ref.floats(ref.get("attn_output" + L, step));
            std::vector<float> got = out.host();
            if (want.size() != got.size()) {   // last layer of a prompt batch keeps the output row only
                got.erase(got.begin(), got.end() - want.size());
            }
            ck.check("qsa attn_output" + L + " step " + std::to_string(step), got, want, verbose && il < 8);
            // selection: llama.cpp's unique selected cells up to the query position
            const Frd::Rec* tk = ref.find("indexer_top_k" + L, step);
            if (tk && T == 1 && !sel[0].empty()) {
                std::vector<int32_t> theirs = ref.ints(*tk);
                std::sort(theirs.begin(), theirs.end());
                theirs.erase(std::unique(theirs.begin(), theirs.end()), theirs.end());
                theirs.erase(std::remove_if(theirs.begin(), theirs.end(), [&](int32_t v) { return v > pos || v < 0; }), theirs.end());
                std::vector<int32_t> ours = sel[0];
                std::sort(ours.begin(), ours.end());
                std::vector<int32_t> common;
                std::set_intersection(ours.begin(), ours.end(), theirs.begin(), theirs.end(), std::back_inserter(common));
                const double frac = theirs.empty() ? 1.0 : double(common.size()) / double(std::max(ours.size(), theirs.size()));
                ++sel_steps;
                sel_cells += long(std::max(ours.size(), theirs.size()));
                sel_same += long(common.size());
                sel_worst = std::min(sel_worst, frac);
            }
        }
        pos += T;
    }
    if (sel_steps)
        std::printf("qsa: indexer selection over %ld (layer, step) pairs: %.4f%% of cells identical, worst pair %.4f%%\n", sel_steps,
                    100.0 * sel_same / sel_cells, 100.0 * sel_worst);
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

// token ids of the dump: the prompt, then the token fed at each decode step
std::vector<int32_t> dump_tokens(const std::string& frd_path, std::vector<int>* step_len) {
    std::ifstream m(frd_path + ".meta");
    std::vector<int32_t> seq;
    std::string line;
    while (std::getline(m, line)) {
        std::istringstream in(line);
        std::string tag;
        in >> tag;
        long v;
        int n = 0;
        while (in >> v) { seq.push_back(int32_t(v)); ++n; }
        if (tag == "prompt") step_len->push_back(n);          // the prompt was one batch (step -1)
        else for (int i = 0; i < n; ++i) step_len->push_back(1);   // prompt_tbt / generated: one token per step
    }
    return seq;
}

// PLE: the n-gram embedding rows, and layer 1's attention-side mix after the PLE update
void test_ple(const BlockCtx& c, const Gguf& g, const std::string& frd_path, const Frd& ref, const std::vector<int>& steps,
              Checker& ck) {
    const Spec& s = c.s;
    const Ple p = parse_ple(g);
    RowReader rr(g.shards[p.table_shard], p.table_offset, p.row_bytes, 16);
    PleHost h;
    h.ple = &p;
    h.reader = &rr;
    std::vector<int> lens;
    const std::vector<int32_t> seq = dump_tokens(frd_path, &lens);
    std::vector<PleState> st(s.n_layer);
    for (int il : s.ple_layers) st[il] = alloc_ple_state(s, p);
    int64_t pos = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
        const int step = steps[k], T = lens.at(k);
        Dev emb(size_t(T) * s.d_model);
        ple_embed(c, h, seq.data(), pos, T, emb.p);
        ck.check("ple_embd step " + std::to_string(step), emb.host(), ref.floats(ref.get("ple_embd", step)), true);
        for (int il : s.ple_layers) {
            const std::string L = "-" + std::to_string(il);
            Dev x(ref.floats(ref.get("l_last-" + std::to_string(il - 1), step)));
            ple_block(c, il, p, emb.p, x.p, T, st[il]);
            Dev mixed(size_t(T) * s.d_model), inject(size_t(T) * s.hc_count);
            hc_mix(c, il, 0, x.p, T, mixed.p, inject.p);
            cudaStreamSynchronize(c.stream);
            ck.check("hc_mixed" + L + " (attn, after PLE)", mixed.host(), ref.floats(ref.get("hc_mixed" + L, step, 0)), true);
        }
        pos += T;
    }
    for (int il : s.ple_layers) free_ple_state(st[il]);
}

// head: logits from llama.cpp's own result_norm
void test_head(const BlockCtx& c, const Frd& ref, const std::vector<int>& steps, Checker& ck) {
    for (int step : steps) {
        const Frd::Rec& nr = ref.get("result_norm", step);
        const int T = int(nr.ne[1]);
        Dev x(ref.floats(nr)), lg(size_t(T) * c.s.n_vocab);
        head_logits(c, x.p, T, lg.p);
        cudaStreamSynchronize(c.stream);
        ck.check("result_output step " + std::to_string(step), lg.host(), ref.floats(ref.get("result_output", step)), true);
    }
}

// End to end: token ids -> logits with flashrt's own caches and states, compared per layer
// and at the logits; also counts identical greedy tokens.
void test_full(const BlockCtx& c, const Gguf& g, const std::string& frd_path, const Frd& ref, const std::vector<int>& steps,
               Checker& ck) {
    const Spec& s = c.s;
    const int n = s.d_model, hc = s.hc_count;
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({n, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) throw std::runtime_error("arena allocation failed");
    load_experts(g, s, arena, 12);
    CpuPool pool(8, physical_cpus());
    MoeHost mh;
    mh.arena = &arena;
    mh.pool = &pool;
    const Ple p = parse_ple(g);
    RowReader rr(g.shards[p.table_shard], p.table_offset, p.row_bytes, 16);
    PleHost ph;
    ph.ple = &p;
    ph.reader = &rr;
    std::vector<GdnState> gdn(s.n_layer);
    std::vector<QsaCache> kv(s.n_layer);
    std::vector<PleState> pst(s.n_layer);
    for (int il : s.gdn_layers) gdn[il] = alloc_gdn_state(s);
    for (int il : s.qsa_layers) kv[il] = alloc_qsa_cache(s, 2048);
    for (int il : s.ple_layers) pst[il] = alloc_ple_state(s, p);
    std::vector<int> lens;
    const std::vector<int32_t> seq = dump_tokens(frd_path, &lens);
    std::set<int> ple_set(s.ple_layers.begin(), s.ple_layers.end());
    int64_t pos = 0;
    int agree = 0, total = 0;
    double kld_sum = 0, kld_max = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
        const int step = steps[k], T = lens.at(k);
        std::vector<int32_t> toks(seq.begin() + pos, seq.begin() + pos + T);
        Dev emb(size_t(T) * n), x(size_t(T) * hc * n), mixed(size_t(T) * n), inject(size_t(T) * hc), blk(size_t(T) * n),
            pemb(size_t(T) * n);
        embed(c, toks.data(), T, emb.p);
        if (k < 2) {
            cudaStreamSynchronize(c.stream);
            std::printf("  step %d model.input_embed rel L2 %.3e\n", step, rel_l2(emb.host(), ref.floats(ref.get("model.input_embed", step))));
        }
        for (int t = 0; t < T; ++t)
            for (int st = 0; st < hc; ++st)
                cudaMemcpyAsync(x.p + (size_t(t) * hc + st) * n, emb.p + size_t(t) * n, size_t(n) * 4, cudaMemcpyDeviceToDevice, c.stream);
        if (!ple_set.empty()) ple_embed(c, ph, seq.data(), pos, T, pemb.p);
        double worst_layer = 0;
        for (int il = 0; il < s.n_layer; ++il) {
            if (ple_set.count(il)) ple_block(c, il, p, pemb.p, x.p, T, pst[il]);
            hc_mix(c, il, 0, x.p, T, mixed.p, inject.p);
            if (s.mixer[il] == Mixer::QSA) qsa_mixer(c, il, mixed.p, T, int(pos), kv[il], blk.p);
            else gdn_mixer(c, il, mixed.p, T, gdn[il], blk.p);
            hc_combine(c, x.p, blk.p, inject.p, T);
            hc_mix(c, il, 1, x.p, T, mixed.p, inject.p);
            MoeTrace tr;
            moe_block(c, il, mixed.p, T, mh, blk.p, k < 2 ? &tr : nullptr);
            hc_combine(c, x.p, blk.p, inject.p, T);
            cudaStreamSynchronize(c.stream);
            if (T == 1 || il < s.n_layer - 1) {
                const std::vector<float> want = ref.floats(ref.get("l_last-" + std::to_string(il), step));
                const double e = rel_l2(x.host(), want);
                worst_layer = std::max(worst_layer, e);
                if (k < 2) {   // the first steps in detail: layer error and routing agreement
                    const std::vector<int32_t> ids = ref.ints(ref.get("ffn_moe_topk-" + std::to_string(il), step));
                    int same = 0;
                    for (int q = 0; q < s.top_k; ++q)
                        same += std::find(ids.end() - s.top_k, ids.end(), tr.topk[size_t(T - 1) * s.top_k + q]) != ids.end();
                    std::printf("    layer %2d (%s): l_last rel L2 %.3e, routing %d/%d\n", il,
                                s.mixer[il] == Mixer::QSA ? "QSA" : "GDN", e, same, s.top_k);
                }
            }
        }
        // head on the last token
        Dev last(size_t(hc) * n), norm(n), lg(size_t(s.n_vocab));
        cudaMemcpyAsync(last.p, x.p + size_t(T - 1) * hc * n, size_t(hc) * n * 4, cudaMemcpyDeviceToDevice, c.stream);
        hc_mix(c, -1, 2, last.p, 1, norm.p, nullptr);
        head_logits(c, norm.p, 1, lg.p);
        cudaStreamSynchronize(c.stream);
        const std::vector<float> ours = lg.host();
        const std::vector<float> want = ref.floats(ref.get("result_output", step));
        const std::vector<float> want_last(want.end() - s.n_vocab, want.end());
        ck.check("logits step " + std::to_string(step), ours, want_last, false);
        // KL(ref || ours) over the full vocabulary
        double kld = 0;
        {
            const float mr = *std::max_element(want_last.begin(), want_last.end());
            const float mo = *std::max_element(ours.begin(), ours.end());
            double zr = 0, zo = 0;
            for (int v = 0; v < s.n_vocab; ++v) { zr += std::exp(double(want_last[v]) - mr); zo += std::exp(double(ours[v]) - mo); }
            const double lzr = std::log(zr) + mr, lzo = std::log(zo) + mo;
            for (int v = 0; v < s.n_vocab; ++v) {
                const double lr = want_last[v] - lzr, lo = ours[v] - lzo;
                kld += std::exp(lr) * (lr - lo);
            }
        }
        kld_sum += kld;
        kld_max = std::max(kld_max, kld);
        const int a = int(std::max_element(ours.begin(), ours.end()) - ours.begin());
        const int b = int(std::max_element(want_last.begin(), want_last.end()) - want_last.begin());
        agree += a == b;
        ++total;
        std::printf("full: step %3d (%2d tok at %3lld): worst layer rel L2 %.2e, logits rel L2 %.2e, KLD %.5f, argmax %d vs %d%s\n",
                    step, T, (long long) pos, worst_layer, rel_l2(ours, want_last), kld, a, b, a == b ? "" : "  DIFFERENT");
        pos += T;
    }
    std::printf("full: greedy token agreement %d / %d; KLD(ref || flashrt) mean %.5f, max %.5f over %d steps\n", agree,
                total, kld_sum / std::max(total, 1), kld_max, total);
    for (int il : s.gdn_layers) free_gdn_state(gdn[il]);
    for (int il : s.qsa_layers) free_qsa_cache(kv[il]);
    for (int il : s.ple_layers) free_ple_state(pst[il]);
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
    w.load(g, plan, false);   // ggml Q3_K, so projections stay bit-comparable with llama.cpp's
    const Frd ref(argv[2]);
    const std::string probe = ref.find("model.input_embed", -1) ? "model.input_embed" : "hc_mixed-" + std::to_string(s.qsa_layers[0]);
    std::vector<int> steps{-1};
    for (int st = 0; ref.find(probe, st); ++st) steps.push_back(st);

    const int max_t = int(ref.get(probe, -1).ne[1]);
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
        else if (t == "ple") test_ple(c, g, argv[2], ref, steps, ck);
        else if (t == "head") test_head(c, ref, steps, ck);
        else if (t == "full") test_full(c, g, argv[2], ref, steps, ck);
        else { std::fprintf(stderr, "unknown test %s\n", t.c_str()); return 2; }
    }
    std::printf("fr_parity: %d checks, %d over tolerance %.0e; worst %.3e (%s)\n", ck.checks, ck.fails, ck.tol, ck.worst,
                ck.worst_what.c_str());
    free_block_scratch(scratch);
    return ck.fails ? 1 : 0;
}
