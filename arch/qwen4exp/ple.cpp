// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/ple.hpp"

#include "core/gguf.hpp"

#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {
std::vector<uint64_t> u64_array(const Gguf& g, const std::string& key) {
    const GgufValue* v = g.get(key);
    const GgufArray* a = v ? v->as_array() : nullptr;
    if (!a) throw std::runtime_error("qwen4exp PLE: missing array " + key);
    std::vector<uint64_t> out;
    for (const GgufValue& e : *a) {
        if (auto u = std::get_if<uint64_t>(&e.v)) out.push_back(*u);
        else if (auto i = e.as_int()) out.push_back(uint64_t(*i));
        else throw std::runtime_error("qwen4exp PLE: non-integer in " + key);
    }
    return out;
}
}  // namespace

Ple parse_ple(const Gguf& g) {
    Ple p;
    p.ngram = int(g.get_int("qwen4exp.ple.ngram_size", 0));
    p.heads_per_ngram = int(g.get_int("qwen4exp.ple.heads_per_ngram", 0));
    p.eos = int32_t(g.get_int("qwen4exp.ple.eos_token_id", -1));
    p.n_heads = (p.ngram - 1) * p.heads_per_ngram;
    p.mult = u64_array(g, "qwen4exp.ple.layer_multipliers");
    p.offset = u64_array(g, "qwen4exp.ple.head_offsets");
    p.vocab = u64_array(g, "qwen4exp.ple.head_vocab_sizes");
    if (p.ngram < 2 || p.n_heads <= 0 || int(p.mult.size()) < p.ngram || int(p.offset.size()) != p.n_heads ||
        int(p.vocab.size()) != p.n_heads || p.eos < 0)
        throw std::runtime_error("qwen4exp PLE: inconsistent metadata");
    const GgufTensor* t = g.tensor("per_layer_token_embd.weight");
    if (!t) throw std::runtime_error("qwen4exp PLE: per_layer_token_embd.weight missing");
    constexpr uint32_t kIQ4_NL = 20;   // ggml type id: blocks of 32 values in 18 bytes
    if (t->type != kIQ4_NL || t->dims.at(0) % 32) throw std::runtime_error("qwen4exp PLE: expected an IQ4_NL table");
    p.row_bytes = uint64_t(t->dims.at(0)) / 32 * 18;
    if (p.row_bytes * uint64_t(t->dims.at(1)) > t->bytes) throw std::runtime_error("qwen4exp PLE: table is truncated");
    p.table_offset = t->file_offset;
    p.table_shard = t->shard;
    return p;
}

void ple_rows(const Ple& p, const int32_t* seq, int64_t pos, uint32_t* out) {
    int64_t ctx[8];
    ctx[0] = seq[pos];
    bool cut = false;
    for (int s = 1; s < p.ngram; ++s) {
        const int64_t t = cut || pos - s < 0 ? -1 : seq[pos - s];
        cut = cut || t < 0 || t == p.eos;
        ctx[s] = cut ? p.eos : t;
    }
    for (int n = 2; n <= p.ngram; ++n) {
        uint64_t mixed = uint64_t(ctx[0]) * p.mult[0];
        for (int j = 1; j < n; ++j) mixed ^= uint64_t(ctx[j]) * p.mult[j];
        const int base = (n - 2) * p.heads_per_ngram;
        for (int h = 0; h < p.heads_per_ngram; ++h)
            out[base + h] = uint32_t(mixed % p.vocab[base + h] + p.offset[base + h]);
    }
}

}  // namespace flashrt::qwen4exp
