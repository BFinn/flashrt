// SPDX-License-Identifier: Apache-2.0
// route_trace: dump MoE routing decisions from a llama.cpp run, for tools/cache_sim.py and
// the routing studies (cache-conditional routing, draft-window expert unions).
//
//   route_trace --model M.gguf --ids prompt.txt [--n-prompt N] [--gen G] [--out PREFIX]
//               [--temp T --top-k K --top-p P --seed S] [--probs] [--prefill-trace] [--qsa]
//               [--ctx C] [--threads T] [--ubatch U]
//
// Links against a llama.cpp build (MIT) and observes the graph through the scheduler's
// eval callback: `ffn_moe_topk-<layer>` (I32 [k, n_tokens]) and, with --probs,
// `ffn_moe_probs-<layer>` (F32 [n_expert, n_tokens]); with --qsa, the sparse-attention
// selection `indexer_top_k-<layer>` (I32 [width, n_tokens], KV cell ids). Experts stay on the CPU
// (`ffn_.*_exps` override), as in the deployed configuration.
//
// Outputs (.npy, C order):
//   PREFIX.decode_topk.npy   int32 [G, n_layer, k]        routed experts per decoded token
//   PREFIX.decode_probs.npy  float32 [G, n_layer, n_exp]  router probabilities (--probs)
//   PREFIX.prefill_topk.npy  int32 [N, n_layer, k]        prompt tokens (--prefill-trace);
//                                                         rows the graph skipped are -1
//   PREFIX.decode_qsa.npy    int32 [G, n_attn_layer, width] selected KV cells per decoded
//                            token (--qsa); layer ids are in meta.json, padding is -1
//   PREFIX.tokens.txt        the generated token ids
//   PREFIX.meta.json         run settings and timings
//
// The prompt file holds token ids separated by whitespace or commas. Decoding samples with
// end-of-generation tokens masked, so every run produces exactly G tokens.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

// ---- .npy writer (format 1.0)
void write_npy(const std::string& path, const char* descr, const std::vector<size_t>& shape,
               const void* data, size_t bytes) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) dims += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const size_t total = 10 + hdr.size() + 1;
    hdr.append((64 - total % 64) % 64, ' ');
    hdr += '\n';
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const uint16_t len = uint16_t(hdr.size());
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&len, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(data, 1, bytes, f);
    std::fclose(f);
}

// Parses "<prefix>-<layer>"; returns the layer or -1.
int layer_of(const char* name, const char* prefix) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    char* end = nullptr;
    const long il = std::strtol(name + n + 1, &end, 10);
    return (end && *end == '\0') ? int(il) : -1;
}

struct Tracer {
    int n_layer = 0, k = 0, n_exp = 0;
    bool want_probs = false, want_qsa = false;
    bool recording = false;
    int n_tokens = 0;                 // rows in the current ubatch
    std::vector<int32_t> topk;        // [n_tokens, n_layer, k] for the current ubatch
    std::vector<float> probs;         // [n_tokens, n_layer, n_exp]
    std::vector<std::vector<int32_t>> qsa;   // per layer, the current decode token's selection
    std::vector<char> scratch;

    void begin(int n) {
        n_tokens = n;
        topk.assign(size_t(n) * n_layer * k, -1);
        if (want_probs) probs.assign(size_t(n) * n_layer * n_exp, 0.0f);
        if (want_qsa) qsa.assign(n_layer, {});
    }
};

// Copies rows of a (possibly strided) 2-D tensor to the host in one backend read.
void read_rows(ggml_tensor* t, std::vector<char>& scratch, int64_t rows, size_t row_bytes) {
    const size_t span = size_t(rows - 1) * t->nb[1] + row_bytes;
    scratch.resize(span);
    ggml_backend_tensor_get(t, scratch.data(), 0, span);
}

bool eval_cb(ggml_tensor* t, bool ask, void* ud) {
    auto* tr = static_cast<Tracer*>(ud);
    const int il_topk = layer_of(t->name, "ffn_moe_topk");
    const int il_prob = tr->want_probs ? layer_of(t->name, "ffn_moe_probs") : -1;
    const int il_qsa = tr->want_qsa && tr->n_tokens == 1 ? layer_of(t->name, "indexer_top_k") : -1;
    if (ask) return tr->recording && (il_topk >= 0 || il_prob >= 0 || il_qsa >= 0);
    if (!tr->recording) return true;

    if (il_topk >= 0 && il_topk < tr->n_layer && t->type == GGML_TYPE_I32 && t->ne[0] == tr->k) {
        const int64_t rows = t->ne[1];
        read_rows(t, tr->scratch, rows, size_t(tr->k) * 4);
        // The last layer may run on the output rows only; those are the trailing tokens.
        const int64_t first = tr->n_tokens - rows;
        for (int64_t r = 0; r < rows && first + r >= 0; ++r)
            std::memcpy(&tr->topk[(size_t(first + r) * tr->n_layer + il_topk) * tr->k],
                        tr->scratch.data() + size_t(r) * t->nb[1], size_t(tr->k) * 4);
    } else if (il_prob >= 0 && il_prob < tr->n_layer && t->type == GGML_TYPE_F32 && t->ne[0] == tr->n_exp) {
        const int64_t rows = t->ne[1];
        read_rows(t, tr->scratch, rows, size_t(tr->n_exp) * 4);
        const int64_t first = tr->n_tokens - rows;
        for (int64_t r = 0; r < rows && first + r >= 0; ++r)
            std::memcpy(&tr->probs[(size_t(first + r) * tr->n_layer + il_prob) * tr->n_exp],
                        tr->scratch.data() + size_t(r) * t->nb[1], size_t(tr->n_exp) * 4);
    } else if (il_qsa >= 0 && il_qsa < tr->n_layer && t->type == GGML_TYPE_I32) {
        auto& v = tr->qsa[il_qsa];
        v.resize(size_t(t->ne[0]));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);   // decode: one contiguous row
    }
    return true;
}

std::vector<llama_token> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::vector<llama_token> ids;
    long v;
    while (in >> v) ids.push_back(llama_token(v));
    return ids;
}

llama_token sample(const float* logits, int n_vocab, const llama_vocab* vocab, float temp, int top_k,
                   float top_p, std::mt19937_64& rng, std::vector<int>& idx) {
    idx.resize(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    auto masked = [&](int i) { return llama_vocab_is_eog(vocab, i); };
    const int kk = std::min(temp <= 0.0f ? 1 : std::max(top_k, 1), n_vocab);
    // candidates: top-kk non-EOG tokens by logit
    std::vector<int> cand;
    cand.reserve(kk);
    std::partial_sort(idx.begin(), idx.begin() + std::min(n_vocab, kk + 64), idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    for (int i = 0; i < n_vocab && int(cand.size()) < kk; ++i)
        if (!masked(idx[i])) cand.push_back(idx[i]);
    if (temp <= 0.0f || cand.size() == 1) return cand[0];
    std::vector<double> p(cand.size());
    const double mx = logits[cand[0]];
    double sum = 0;
    for (size_t i = 0; i < cand.size(); ++i) sum += p[i] = std::exp((logits[cand[i]] - mx) / temp);
    double acc = 0;
    size_t keep = cand.size();
    for (size_t i = 0; i < cand.size(); ++i) {
        acc += p[i] / sum;
        if (acc >= top_p) { keep = i + 1; break; }
    }
    std::discrete_distribution<size_t> d(p.begin(), p.begin() + keep);
    return cand[d(rng)];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out = "trace";
    int n_prompt = 0, gen = 1024, ctx = 0, threads = 12, ubatch = 2048, top_k = 20;
    float temp = 1.0f, top_p = 0.95f;
    uint64_t seed = 42;
    bool want_probs = false, prefill_trace = false, want_qsa = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model_path = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--n-prompt") n_prompt = std::atoi(next());
        else if (a == "--gen") gen = std::atoi(next());
        else if (a == "--out") out = next();
        else if (a == "--ctx") ctx = std::atoi(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--ubatch") ubatch = std::atoi(next());
        else if (a == "--temp") temp = float(std::atof(next()));
        else if (a == "--top-k") top_k = std::atoi(next());
        else if (a == "--top-p") top_p = float(std::atof(next()));
        else if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--probs") want_probs = true;
        else if (a == "--prefill-trace") prefill_trace = true;
        else if (a == "--qsa") want_qsa = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || ids_path.empty()) {
        std::fprintf(stderr, "usage: route_trace --model M.gguf --ids prompt.txt [--n-prompt N] [--gen G] [--out PREFIX]\n");
        return 2;
    }

    std::vector<llama_token> prompt = read_ids(ids_path);
    if (n_prompt > 0 && size_t(n_prompt) < prompt.size()) prompt.resize(n_prompt);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 1; }
    if (ctx <= 0) ctx = int(prompt.size()) + gen + 256;

    ggml_backend_load_all();
    llama_backend_init();

    // experts on the CPU, everything else on the GPU: the deployed layout
    llama_model_tensor_buft_override ov[2] = {{"ffn_.*_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = ov;

    const auto t_load0 = Clock::now();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const double load_s = std::chrono::duration<double>(Clock::now() - t_load0).count();

    Tracer tr;
    tr.want_probs = want_probs;
    tr.want_qsa = want_qsa;
    tr.n_layer = llama_model_n_layer(model);
    {
        char buf[64];
        const std::string arch = llama_model_meta_val_str(model, "general.architecture", buf, sizeof buf) > 0 ? buf : "";
        auto meta_int = [&](const std::string& key) {
            char v[64];
            return llama_model_meta_val_str(model, (arch + "." + key).c_str(), v, sizeof v) > 0 ? std::atoi(v) : 0;
        };
        tr.k = meta_int("expert_used_count");
        tr.n_exp = meta_int("expert_count");
        std::fprintf(stderr, "route_trace: arch=%s n_layer=%d n_expert=%d top_k=%d prompt=%zu gen=%d ctx=%d\n",
                     arch.c_str(), tr.n_layer, tr.n_exp, tr.k, prompt.size(), gen, ctx);
    }
    if (tr.k <= 0 || tr.n_exp <= 0) { std::fprintf(stderr, "not a MoE model?\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(ctx);
    cp.n_batch = uint32_t(ubatch);
    cp.n_ubatch = uint32_t(ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &tr;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) { std::fprintf(stderr, "context creation failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- prefill, in ubatch-sized pieces so the tracer sees one ubatch per decode call
    std::vector<int32_t> prefill_topk;
    if (prefill_trace) prefill_topk.assign(prompt.size() * tr.n_layer * tr.k, -1);
    tr.recording = prefill_trace;
    const auto t_pf0 = Clock::now();
    for (size_t pos = 0; pos < prompt.size(); pos += size_t(ubatch)) {
        const int n = int(std::min(prompt.size() - pos, size_t(ubatch)));
        tr.begin(n);
        if (llama_decode(lctx, llama_batch_get_one(prompt.data() + pos, n)) != 0) {
            std::fprintf(stderr, "prefill decode failed at %zu\n", pos);
            return 1;
        }
        if (prefill_trace)
            std::memcpy(&prefill_topk[pos * tr.n_layer * tr.k], tr.topk.data(), tr.topk.size() * 4);
        std::fprintf(stderr, "\rprefill %zu/%zu", pos + n, prompt.size());
    }
    const double prefill_s = std::chrono::duration<double>(Clock::now() - t_pf0).count();
    std::fprintf(stderr, "\nprefill %.1f s (%.1f tok/s, tracing %s)\n", prefill_s, prompt.size() / prefill_s,
                 prefill_trace ? "on" : "off");

    // ---- decode, one token per call
    std::vector<int32_t> dec_topk(size_t(gen) * tr.n_layer * tr.k, -1);
    std::vector<float> dec_probs;
    if (want_probs) dec_probs.assign(size_t(gen) * tr.n_layer * tr.n_exp, 0.0f);
    std::vector<std::vector<std::vector<int32_t>>> dec_qsa;   // [G][n_layer][width]
    std::vector<llama_token> out_ids;
    std::mt19937_64 rng(seed);
    std::vector<int> idx;
    llama_token tok = sample(llama_get_logits_ith(lctx, -1), n_vocab, vocab, temp, top_k, top_p, rng, idx);
    tr.recording = true;
    const auto t_dec0 = Clock::now();
    for (int g = 0; g < gen; ++g) {
        out_ids.push_back(tok);
        tr.begin(1);
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1)) != 0) {
            std::fprintf(stderr, "decode failed at token %d\n", g);
            return 1;
        }
        std::memcpy(&dec_topk[size_t(g) * tr.n_layer * tr.k], tr.topk.data(), tr.topk.size() * 4);
        if (want_probs)
            std::memcpy(&dec_probs[size_t(g) * tr.n_layer * tr.n_exp], tr.probs.data(), tr.probs.size() * 4);
        if (want_qsa) dec_qsa.push_back(std::move(tr.qsa));
        tok = sample(llama_get_logits_ith(lctx, -1), n_vocab, vocab, temp, top_k, top_p, rng, idx);
        if ((g + 1) % 64 == 0) std::fprintf(stderr, "\rdecode %d/%d", g + 1, gen);
    }
    const double decode_s = std::chrono::duration<double>(Clock::now() - t_dec0).count();
    std::fprintf(stderr, "\ndecode %.1f s (%.2f tok/s with tracing syncs)\n", decode_s, gen / decode_s);

    const size_t missing = size_t(std::count(dec_topk.begin(), dec_topk.end(), -1));
    if (missing) std::fprintf(stderr, "warning: %zu decode entries were not observed\n", missing);

    // ---- outputs
    const size_t L = size_t(tr.n_layer), K = size_t(tr.k), E = size_t(tr.n_exp);
    write_npy(out + ".decode_topk.npy", "<i4", {size_t(gen), L, K}, dec_topk.data(), dec_topk.size() * 4);
    if (want_probs)
        write_npy(out + ".decode_probs.npy", "<f4", {size_t(gen), L, E}, dec_probs.data(), dec_probs.size() * 4);
    if (prefill_trace)
        write_npy(out + ".prefill_topk.npy", "<i4", {prompt.size(), L, K}, prefill_topk.data(), prefill_topk.size() * 4);
    std::vector<int> qsa_layers;
    if (want_qsa && !dec_qsa.empty()) {
        size_t W = 0;
        for (int il = 0; il < tr.n_layer; ++il)
            if (!dec_qsa[0][il].empty()) qsa_layers.push_back(il);
        for (auto& tok_rows : dec_qsa)
            for (int il : qsa_layers) W = std::max(W, tok_rows[il].size());
        std::vector<int32_t> flat(dec_qsa.size() * qsa_layers.size() * W, -1);
        for (size_t g = 0; g < dec_qsa.size(); ++g)
            for (size_t j = 0; j < qsa_layers.size(); ++j) {
                const auto& v = dec_qsa[g][qsa_layers[j]];
                std::copy(v.begin(), v.end(), flat.begin() + (g * qsa_layers.size() + j) * W);
            }
        write_npy(out + ".decode_qsa.npy", "<i4", {dec_qsa.size(), qsa_layers.size(), W}, flat.data(), flat.size() * 4);
    }
    {
        std::ofstream f(out + ".tokens.txt");
        for (auto t : out_ids) f << t << '\n';
    }
    {
        std::ofstream f(out + ".meta.json");
        f << "{\"model\":\"" << model_path << "\",\"n_prompt\":" << prompt.size() << ",\"gen\":" << gen
          << ",\"n_layer\":" << L << ",\"n_expert\":" << E << ",\"top_k\":" << K << ",\"temp\":" << temp
          << ",\"top_p\":" << top_p << ",\"sample_top_k\":" << top_k << ",\"seed\":" << seed
          << ",\"ubatch\":" << ubatch << ",\"threads\":" << threads << ",\"load_s\":" << load_s
          << ",\"prefill_s\":" << prefill_s << ",\"decode_s\":" << decode_s << ",\"unobserved\":" << missing
          << ",\"qsa_layers\":[";
        for (size_t j = 0; j < qsa_layers.size(); ++j) f << (j ? "," : "") << qsa_layers[j];
        f << "]}\n";
    }
    std::fprintf(stderr, "wrote %s.*\n", out.c_str());

    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
