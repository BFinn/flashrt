// SPDX-License-Identifier: Apache-2.0
// ref_dump: record llama.cpp's intermediate tensors for a prompt, as the parity reference for
// flashrt's forward pass.
//
//   ref_dump --model M.gguf --ids prompt.txt [--n-prompt N] [--gen G] [--out FILE.frd]
//            [--capture prefix,prefix,...] [--ctx C] [--all-logits 1]
//            [--token-by-token 1] [--layers 3,7,...] [--from-step N]
//
// Runs the prompt as one batch, then G greedy decode steps, with experts on the CPU as in the
// deployed layout, and saves every graph tensor whose name (before "-<layer>") is in the
// capture list. Records are written as they are computed:
//   u32 name_len, name, i32 step (-1 = prompt batch, g = decode step g), i32 ggml type,
//   i64 ne[4], u64 nbytes, data (contiguous, ne[0] fastest)
// Only F32 and I32 tensors are recorded; strided views are gathered element by element.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Dumper {
    FILE* f = nullptr;
    std::set<std::string> want;
    int step = -1;
    long records = 0;
    std::vector<char> span, dense;

    std::set<int> layers;      // empty = every layer
    int from_step = -1;        // record steps >= this only

    bool wanted(const char* name) const {
        if (step < from_step) return false;
        std::string n(name);
        const size_t dash = n.rfind('-');
        if (dash != std::string::npos && dash + 1 < n.size() &&
            std::all_of(n.begin() + dash + 1, n.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            if (!layers.empty() && !layers.count(std::atoi(n.c_str() + dash + 1))) return false;
            n.resize(dash);
        }
        return want.count(n) > 0;
    }

    void write(const ggml_tensor* t) {
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return;
        const size_t es = 4;
        size_t max_off = 0;
        for (int d = 0; d < 4; ++d) max_off += size_t(t->ne[d] - 1) * t->nb[d];
        span.resize(max_off + es);
        ggml_backend_tensor_get(t, span.data(), 0, span.size());
        const int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        dense.resize(size_t(n) * es);
        size_t k = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, k += es)
                        std::memcpy(dense.data() + k,
                                    span.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], es);
        const uint32_t nl = uint32_t(std::strlen(t->name));
        const int32_t type = int32_t(t->type);
        const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        const uint64_t nb = dense.size();
        std::fwrite(&nl, 4, 1, f);
        std::fwrite(t->name, 1, nl, f);
        std::fwrite(&step, 4, 1, f);
        std::fwrite(&type, 4, 1, f);
        std::fwrite(ne, 8, 4, f);
        std::fwrite(&nb, 8, 1, f);
        std::fwrite(dense.data(), 1, dense.size(), f);
        ++records;
    }
};

bool eval_cb(ggml_tensor* t, bool ask, void* ud) {
    auto* d = static_cast<Dumper*>(ud);
    if (ask) return d->wanted(t->name);
    if (d->wanted(t->name)) d->write(t);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out = "ref.frd";
    std::string capture =
        "model.input_embed,ple_embd,hc_norm,hc_gate,hc_mixed,hc_inject,hc_combine,attn_output,linear_attn_out,"
        "ffn_moe_topk,ffn_moe_weights,ffn_moe_out,ffn_shexp_gated,ffn_out,l_last,result_norm,result_output";
    int n_prompt = 64, gen = 4, ctx = 0;
    bool all_logits = false, token_by_token = false;
    std::string layers_arg;
    int from_step = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") model_path = argv[i + 1];
        else if (a == "--ids") ids_path = argv[i + 1];
        else if (a == "--n-prompt") n_prompt = std::atoi(argv[i + 1]);
        else if (a == "--gen") gen = std::atoi(argv[i + 1]);
        else if (a == "--out") out = argv[i + 1];
        else if (a == "--capture") capture = argv[i + 1];
        else if (a == "--ctx") ctx = std::atoi(argv[i + 1]);
        else if (a == "--all-logits") { all_logits = std::atoi(argv[i + 1]) != 0; }
        else if (a == "--token-by-token") token_by_token = std::atoi(argv[i + 1]) != 0;
        else if (a == "--layers") layers_arg = argv[i + 1];
        else if (a == "--from-step") from_step = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || ids_path.empty()) {
        std::fprintf(stderr, "usage: ref_dump --model M.gguf --ids prompt.txt [--n-prompt N] [--gen G] [--out F]\n");
        return 2;
    }
    std::vector<llama_token> prompt;
    {
        std::ifstream f(ids_path);
        long v;
        while (f >> v) prompt.push_back(llama_token(v));
    }
    if (n_prompt > 0 && size_t(n_prompt) < prompt.size()) prompt.resize(n_prompt);
    if (ctx <= 0) ctx = int(prompt.size()) + gen + 256;

    Dumper d;
    std::stringstream ss(capture);
    for (std::string x; std::getline(ss, x, ',');) d.want.insert(x);
    {
        std::stringstream ls(layers_arg);
        for (std::string x; std::getline(ls, x, ',');) d.layers.insert(std::atoi(x.c_str()));
    }
    d.from_step = from_step;
    d.f = std::fopen(out.c_str(), "wb");
    if (!d.f) { std::perror(out.c_str()); return 1; }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_tensor_buft_override ov[2] = {{"ffn_.*_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = ov;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(ctx);
    cp.n_batch = cp.n_ubatch = uint32_t(std::max<size_t>(prompt.size(), 512));
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 12;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;   // unquantized KV: the reference flashrt compares against
    cp.type_v = GGML_TYPE_F16;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &d;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) { std::fprintf(stderr, "context creation failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    d.step = -1;
    if (token_by_token) {
        // the prompt one token per decode call (decode kernels throughout): prompt token i is
        // step i - 1, so step -1 is the first token, as in a 1-token prompt followed by decoding
        for (size_t i = 0; i < prompt.size(); ++i) {
            d.step = int(i) - 1;
            if (llama_decode(lctx, llama_batch_get_one(&prompt[i], 1)) != 0) {
                std::fprintf(stderr, "token-by-token decode failed at %zu\n", i);
                return 1;
            }
            if (i % 256 == 0) std::fprintf(stderr, "\rtoken %zu/%zu", i, prompt.size());
        }
        std::fprintf(stderr, "\n");
    } else {
        // the prompt as one batch; with --all-logits 1 every position produces logits (and the
        // last layer keeps every row), otherwise only the last one
        llama_batch b = llama_batch_init(int(prompt.size()), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            b.token[i] = prompt[i];
            b.pos[i] = llama_pos(i);
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = all_logits || i + 1 == prompt.size();
        }
        b.n_tokens = int(prompt.size());
        const int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            std::fprintf(stderr, "prompt decode failed\n");
            return 1;
        }
    }
    const int step0 = token_by_token ? int(prompt.size()) - 1 : 0;   // numbering of the decode steps
    std::vector<llama_token> out_ids;
    for (int g = 0; g < gen; ++g) {
        const float* lg = llama_get_logits_ith(lctx, -1);
        llama_token tok = int32_t(std::max_element(lg, lg + n_vocab) - lg);
        out_ids.push_back(tok);
        d.step = step0 + g;
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1)) != 0) {
            std::fprintf(stderr, "decode failed at step %d\n", g);
            return 1;
        }
    }
    std::fclose(d.f);
    {
        std::ofstream m(out + ".meta");
        m << (token_by_token ? "prompt_tbt" : "prompt");
        for (auto t : prompt) m << ' ' << t;
        m << "\ngenerated";
        for (auto t : out_ids) m << ' ' << t;
        m << "\n";
    }
    std::fprintf(stderr, "ref_dump: %ld records for %zu prompt tokens and %d decode steps -> %s\n", d.records,
                 prompt.size(), gen, out.c_str());
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
