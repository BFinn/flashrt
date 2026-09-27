// SPDX-License-Identifier: Apache-2.0
// gguf_dump: print a GGUF's metadata and tensor directory, with bytes per tensor type and
// per tensor group (the input for sizing tiers: dense in VRAM, experts on the host).
//
//   gguf_dump FILE.gguf [--tensors] [--arrays]
//
// For split models, pass shard 1; the siblings are opened too.
#include "core/gguf.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <regex>
#include <string>

using namespace flashrt;

namespace {

std::string show(const GgufValue& v, bool full_arrays) {
    if (auto s = v.as_string()) {
        std::string e;
        for (char c : s->substr(0, 120)) e += c == '\n' ? "\\n" : std::string(1, c);
        return "\"" + e + (s->size() > 120 ? "...\"" : "\"");
    }
    if (auto a = v.as_array()) {
        std::string out = "[" + std::to_string(a->size()) + "]";
        const size_t n = full_arrays ? a->size() : std::min<size_t>(a->size(), 8);
        if (n) out += " ";
        for (size_t i = 0; i < n; ++i) out += (i ? "," : "") + show((*a)[i], false);
        if (n < a->size()) out += ",...";
        return out;
    }
    if (auto b = std::get_if<bool>(&v.v)) return *b ? "true" : "false";
    if (auto d = std::get_if<double>(&v.v)) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%g", *d);
        return buf;
    }
    if (auto i = v.as_int()) return std::to_string(*i);
    return "?";
}

// "blk.12.ffn_up_exps.weight" -> "blk.*.ffn_up_exps"
std::string group_of(const std::string& name) {
    static const std::regex blk(R"(^blk\.\d+\.)");
    std::string g = std::regex_replace(name, blk, "blk.*.");
    if (g.size() > 7 && g.compare(g.size() - 7, 7, ".weight") == 0) g.resize(g.size() - 7);
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gguf_dump FILE.gguf [--tensors] [--arrays]\n");
        return 2;
    }
    bool tensors = false, arrays = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tensors")) tensors = true;
        else if (!std::strcmp(argv[i], "--arrays")) arrays = true;
    }
    const Gguf g = Gguf::open(argv[1]);
    std::printf("# %zu shard(s), GGUF v%u, %zu tensors, %zu metadata keys\n", g.shards.size(), g.version,
                g.tensors.size(), g.meta.size());
    for (const auto& [k, v] : g.meta) {
        if (k.rfind("tokenizer.ggml.", 0) == 0 && v.as_array() && !arrays)
            std::printf("%s = [%zu]\n", k.c_str(), v.as_array()->size());
        else
            std::printf("%s = %s\n", k.c_str(), show(v, arrays).c_str());
    }

    std::map<std::string, std::pair<size_t, uint64_t>> by_type, by_group;
    uint64_t total = 0;
    for (const auto& t : g.tensors) {
        auto& a = by_type[ggml_type_name(t.type)];
        a.first++, a.second += t.bytes;
        auto& b = by_group[group_of(t.name) + " " + ggml_type_name(t.type)];
        b.first++, b.second += t.bytes;
        total += t.bytes;
        if (tensors) {
            std::printf("T %-48s %-6s [", t.name.c_str(), ggml_type_name(t.type));
            for (size_t i = 0; i < t.dims.size(); ++i) std::printf("%s%lld", i ? "," : "", (long long) t.dims[i]);
            std::printf("] %.2f MiB shard %d\n", t.bytes / 1048576.0, t.shard);
        }
    }
    std::printf("\n# bytes by type\n");
    for (const auto& [k, v] : by_type) std::printf("%-8s %6zu tensors %10.1f MiB\n", k.c_str(), v.first, v.second / 1048576.0);
    std::printf("\n# bytes by tensor group\n");
    for (const auto& [k, v] : by_group) std::printf("%-44s %5zu %10.1f MiB\n", k.c_str(), v.first, v.second / 1048576.0);
    std::printf("\n# total %.2f GiB\n", total / 1073741824.0);
    return 0;
}
