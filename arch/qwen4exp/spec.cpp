// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/spec.hpp"

#include "core/formats.hpp"
#include "core/gguf.hpp"

#include <regex>
#include <stdexcept>

namespace flashrt::qwen4exp {

namespace {

int64_t need_int(const Gguf& g, const std::string& key) {
    const GgufValue* v = g.get(key);
    if (!v || !v->as_int()) throw std::runtime_error("qwen4exp: missing integer key " + key);
    return *v->as_int();
}

double need_float(const Gguf& g, const std::string& key) {
    const GgufValue* v = g.get(key);
    if (!v || !v->as_float()) throw std::runtime_error("qwen4exp: missing float key " + key);
    return *v->as_float();
}

void expect_dims(const GgufTensor& t, std::initializer_list<int64_t> dims) {
    if (t.dims != std::vector<int64_t>(dims))
        throw std::runtime_error("qwen4exp: unexpected shape for " + t.name);
}

}  // namespace

Spec parse(const Gguf& g) {
    Spec s;
    s.arch = g.get_string("general.architecture");
    if (s.arch != "qwen4exp") throw std::runtime_error("not a qwen4exp model: " + s.arch);
    const std::string a = "qwen4exp.";

    s.n_layer = int(need_int(g, a + "block_count"));
    s.d_model = int(need_int(g, a + "embedding_length"));
    s.max_context = int(need_int(g, a + "context_length"));
    s.rms_eps = need_float(g, a + "attention.layer_norm_rms_epsilon");

    s.n_head = int(need_int(g, a + "attention.head_count"));
    s.n_head_kv = int(need_int(g, a + "attention.head_count_kv"));
    s.head_dim_k = int(need_int(g, a + "attention.key_length"));
    s.head_dim_v = int(need_int(g, a + "attention.value_length"));
    s.idx_heads = int(need_int(g, a + "attention.indexer.head_count"));
    s.idx_dim = int(need_int(g, a + "attention.indexer.key_length"));
    s.idx_top_k = int(need_int(g, a + "attention.indexer.top_k"));
    s.rope_dims = int(need_int(g, a + "rope.dimension_count"));
    s.rope_base = need_float(g, a + "rope.freq_base");

    s.ssm_conv = int(need_int(g, a + "ssm.conv_kernel"));
    s.ssm_groups = int(need_int(g, a + "ssm.group_count"));
    s.ssm_inner = int(need_int(g, a + "ssm.inner_size"));
    s.ssm_state = int(need_int(g, a + "ssm.state_size"));
    s.ssm_heads = int(need_int(g, a + "ssm.time_step_rank"));

    s.n_expert = int(need_int(g, a + "expert_count"));
    s.top_k = int(need_int(g, a + "expert_used_count"));
    s.d_ff_expert = int(need_int(g, a + "expert_feed_forward_length"));
    s.d_ff_shared = int(need_int(g, a + "expert_shared_feed_forward_length"));

    s.hc_count = int(need_int(g, a + "hyper_connection.count"));
    s.hc_rank = int(need_int(g, a + "hyper_connection.low_rank"));
    for (int64_t l : g.get_int_array(a + "ple.layers")) s.ple_layers.push_back(int(l));
    s.ple_ngram = int(need_int(g, a + "ple.ngram_size"));
    s.ple_heads = int(g.get_int_array(a + "ple.head_offsets").size());
    s.ple_dim = int(need_int(g, a + "embedding_length_per_layer_input"));

    const int interval = int(need_int(g, a + "full_attention_interval"));
    const std::vector<int64_t> ratios = g.get_int_array(a + "attention.compress_ratios");
    for (int i = 0; i < s.n_layer; ++i) {
        const bool qsa = (i + 1) % interval == 0;
        s.mixer.push_back(qsa ? Mixer::QSA : Mixer::GDN);
        (qsa ? s.qsa_layers : s.gdn_layers).push_back(i);
        if (qsa && size_t(i) < ratios.size()) {
            if (s.qsa_block && s.qsa_block != ratios[i]) throw std::runtime_error("qwen4exp: mixed compress ratios");
            s.qsa_block = int(ratios[i]);
        }
    }

    const GgufTensor* emb = g.tensor("token_embd.weight");
    if (!emb) throw std::runtime_error("qwen4exp: token_embd.weight missing");
    s.n_vocab = int(emb->dims.at(1));
    const GgufTensor* ple = g.tensor("per_layer_token_embd.weight");
    if (!ple) throw std::runtime_error("qwen4exp: per_layer_token_embd.weight missing");
    s.ple_rows = ple->dims.at(1);
    return s;
}

const char* tier_name(Tier t) {
    switch (t) {
        case Tier::VramDense: return "vram-dense";
        case Tier::HostExperts: return "host-experts";
        default: return "ssd-ple";
    }
}

WeightPlan plan(const Gguf& g, const Spec& s) {
    WeightPlan p;
    static const std::regex blk(R"(^blk\.(\d+)\.(.+)$)");
    static const std::regex exps(R"(^ffn_(gate|up|down)_exps\.weight$)");
    int n_exps = 0;
    for (const GgufTensor& t : g.tensors) {
        Placement pl{&t, Tier::VramDense, -1};
        std::smatch m;
        if (std::regex_match(t.name, m, blk)) {
            pl.layer = std::stoi(m[1]);
            const std::string rest = m[2];
            std::smatch e;
            if (std::regex_match(rest, e, exps)) {
                pl.tier = Tier::HostExperts;
                if (t.type != ggml_type::kQ2_0) throw std::runtime_error("qwen4exp: expert tensor is not Q2_0: " + t.name);
                if (e[1] == "down") expect_dims(t, {s.d_ff_expert, s.d_model, s.n_expert});
                else expect_dims(t, {s.d_model, s.d_ff_expert, s.n_expert});
                ++n_exps;
            }
        } else if (t.name == "per_layer_token_embd.weight") {
            pl.tier = Tier::SsdPle;
        }
        p.bytes[int(pl.tier)] += t.bytes;
        p.tensors.push_back(pl);
    }
    if (n_exps != 3 * s.n_layer) throw std::runtime_error("qwen4exp: expected 3 expert tensors per layer");
    p.n_experts_total = s.n_layer * s.n_expert;
    p.expert_bytes = p.bytes[int(Tier::HostExperts)] / uint64_t(p.n_experts_total);
    return p;
}

uint64_t qsa_kv_bytes_per_cell(const Spec& s) {
    const uint64_t values = uint64_t(s.n_head_kv) * (s.head_dim_k + s.head_dim_v);   // K and V
    return uint64_t(s.qsa_layers.size()) * (values + values / 32 * 2);
}

uint64_t gdn_state_bytes(const Spec& s) {
    // per GDN layer: one state_size x state_size matrix per value head (fp32), plus the
    // convolution window over the qkv channels
    const uint64_t rec = uint64_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4;
    const uint64_t conv = uint64_t(s.ssm_conv - 1) * (s.ssm_inner + 2 * s.ssm_groups * s.ssm_state) * 4;
    return uint64_t(s.gdn_layers.size()) * (rec + conv);
}

}  // namespace flashrt::qwen4exp
