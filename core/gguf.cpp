// SPDX-License-Identifier: Apache-2.0
#include "core/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace flashrt {

namespace {

constexpr uint32_t kMagic = 0x46554747;   // "GGUF" little-endian

enum : uint32_t {
    T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7,
    T_STRING = 8, T_ARRAY = 9, T_U64 = 10, T_I64 = 11, T_F64 = 12,
};

class Reader {
public:
    explicit Reader(const std::string& path) : path_(path) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("cannot open " + path);
        std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
    }
    ~Reader() { if (f_) std::fclose(f_); }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    template <class T> T pod() {
        T v;
        read(&v, sizeof v);
        return v;
    }
    std::string str() {
        const uint64_t n = pod<uint64_t>();
        if (n > (1ull << 30)) fail("string length");
        std::string s(n, '\0');
        read(s.data(), n);
        return s;
    }
    uint64_t tell() const { return uint64_t(std::ftell(f_)); }
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error("malformed GGUF (" + std::string(what) + "): " + path_);
    }

private:
    void read(void* dst, size_t n) {
        if (n && std::fread(dst, 1, n, f_) != n) fail("unexpected end of file");
    }
    std::string path_;
    FILE* f_ = nullptr;
};

GgufValue read_value(Reader& r, uint32_t type, int depth = 0) {
    GgufValue out;
    out.type = type;
    switch (type) {
        case T_U8:   out.v = uint64_t(r.pod<uint8_t>()); break;
        case T_I8:   out.v = int64_t(r.pod<int8_t>()); break;
        case T_U16:  out.v = uint64_t(r.pod<uint16_t>()); break;
        case T_I16:  out.v = int64_t(r.pod<int16_t>()); break;
        case T_U32:  out.v = uint64_t(r.pod<uint32_t>()); break;
        case T_I32:  out.v = int64_t(r.pod<int32_t>()); break;
        case T_U64:  out.v = r.pod<uint64_t>(); break;
        case T_I64:  out.v = r.pod<int64_t>(); break;
        case T_F32:  out.v = double(r.pod<float>()); break;
        case T_F64:  out.v = r.pod<double>(); break;
        case T_BOOL: out.v = r.pod<uint8_t>() != 0; break;
        case T_STRING: out.v = r.str(); break;
        case T_ARRAY: {
            if (depth > 4) r.fail("array nesting");
            const uint32_t et = r.pod<uint32_t>();
            const uint64_t n = r.pod<uint64_t>();
            if (n > (1ull << 32)) r.fail("array length");
            auto arr = std::make_shared<GgufArray>();
            arr->reserve(std::min<uint64_t>(n, 1u << 20));   // a malformed count fails on the read, not here
            for (uint64_t i = 0; i < n; ++i) arr->push_back(read_value(r, et, depth + 1));
            out.v = arr;
            break;
        }
        default: r.fail("value type");
    }
    return out;
}

// Reads one shard's header, metadata and tensor directory; appends tensors with shard index.
void read_shard(Gguf& g, const std::string& path, int shard) {
    Reader r(path);
    if (r.pod<uint32_t>() != kMagic) r.fail("magic");
    const uint32_t version = r.pod<uint32_t>();
    if (version < 2 || version > 3) r.fail("version");
    if (shard == 0) g.version = version;
    const uint64_t n_tensors = r.pod<uint64_t>();
    const uint64_t n_kv = r.pod<uint64_t>();

    uint64_t alignment = 32;   // each file's own (the merged metadata holds shard 1's)
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str();
        const uint32_t t = r.pod<uint32_t>();
        GgufValue v = read_value(r, t);
        if (key == "general.alignment")
            if (auto a = v.as_int()) alignment = uint64_t(*a);
        g.meta.emplace(std::move(key), std::move(v));   // emplace: the first shard's value wins
    }
    if (alignment == 0 || (alignment & (alignment - 1)) || alignment > (1u << 20)) r.fail("alignment");

    const size_t first = g.tensors.size();
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.str();
        const uint32_t nd = r.pod<uint32_t>();
        if (nd > 8) r.fail("tensor rank");
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(int64_t(r.pod<uint64_t>()));
        t.type = r.pod<uint32_t>();
        t.file_offset = r.pod<uint64_t>();   // relative to the data section for now
        t.shard = shard;
        g.tensors.push_back(std::move(t));
    }

    const uint64_t data_start = (r.tell() + alignment - 1) / alignment * alignment;
    const uint64_t file_size = std::filesystem::file_size(path);
    for (size_t i = first; i < g.tensors.size(); ++i)
        if (data_start > file_size || g.tensors[i].file_offset > file_size - data_start) r.fail("tensor offset past the end");

    // sizes from the gaps between consecutive offsets (the data section is dense up to padding)
    std::vector<size_t> order;
    for (size_t i = first; i < g.tensors.size(); ++i) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return g.tensors[a].file_offset < g.tensors[b].file_offset; });
    for (size_t k = 0; k < order.size(); ++k) {
        GgufTensor& t = g.tensors[order[k]];
        const uint64_t next = k + 1 < order.size() ? g.tensors[order[k + 1]].file_offset : file_size - data_start;
        t.bytes = next - t.file_offset;
        t.file_offset += data_start;
    }
}

}  // namespace

std::optional<int64_t> GgufValue::as_int() const {
    if (auto p = std::get_if<int64_t>(&v)) return *p;
    if (auto p = std::get_if<uint64_t>(&v)) return int64_t(*p);
    if (auto p = std::get_if<bool>(&v)) return int64_t(*p);
    return std::nullopt;
}
std::optional<double> GgufValue::as_float() const {
    if (auto p = std::get_if<double>(&v)) return *p;
    if (auto i = as_int()) return double(*i);
    return std::nullopt;
}
const std::string* GgufValue::as_string() const { return std::get_if<std::string>(&v); }
const GgufArray* GgufValue::as_array() const {
    auto p = std::get_if<std::shared_ptr<GgufArray>>(&v);
    return p ? p->get() : nullptr;
}

int64_t GgufTensor::n_elements() const {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    return n;
}

Gguf Gguf::open(const std::string& path) {
    Gguf g;
    g.shards.push_back(path);
    read_shard(g, path, 0);

    const int64_t count = g.get_int("split.count", 1);
    const std::string tag = "-00001-of-";
    const size_t at = path.rfind(tag);
    if (count > 1) {
        if (at == std::string::npos) throw std::runtime_error("split.count > 1 but the name has no -00001-of-: " + path);
        for (int64_t s = 2; s <= count; ++s) {
            char num[8];
            std::snprintf(num, sizeof num, "%05d", int(s));
            std::string p = path;
            p.replace(at + 1, 5, num);
            g.shards.push_back(p);
            read_shard(g, p, int(s - 1));
        }
    }
    for (size_t i = 0; i < g.tensors.size(); ++i) g.by_name[g.tensors[i].name] = i;
    return g;
}

const GgufValue* Gguf::get(const std::string& key) const {
    auto it = meta.find(key);
    return it == meta.end() ? nullptr : &it->second;
}
int64_t Gguf::get_int(const std::string& key, int64_t fallback) const {
    const GgufValue* v = get(key);
    auto i = v ? v->as_int() : std::nullopt;
    return i ? *i : fallback;
}
double Gguf::get_float(const std::string& key, double fallback) const {
    const GgufValue* v = get(key);
    auto f = v ? v->as_float() : std::nullopt;
    return f ? *f : fallback;
}
std::string Gguf::get_string(const std::string& key, const std::string& fallback) const {
    const GgufValue* v = get(key);
    const std::string* s = v ? v->as_string() : nullptr;
    return s ? *s : fallback;
}
std::vector<int64_t> Gguf::get_int_array(const std::string& key) const {
    std::vector<int64_t> out;
    const GgufValue* v = get(key);
    if (!v) return out;
    if (const GgufArray* a = v->as_array()) {
        for (const GgufValue& e : *a)
            if (auto i = e.as_int()) out.push_back(*i);
    } else if (auto i = v->as_int()) {
        out.push_back(*i);   // scalar stored where an array may appear
    }
    return out;
}
const GgufTensor* Gguf::tensor(const std::string& name) const {
    auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : &tensors[it->second];
}

const char* ggml_type_name(uint32_t type) {
    // ids as assigned by ggml (the Q1_0/Q2_0 ids follow the GSQ-RCO-capable forks)
    static const char* names[] = {
        "F32", "F16", "Q4_0", "Q4_1", nullptr, nullptr, "Q5_0", "Q5_1", "Q8_0", "Q8_1",
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
        "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
        "BF16", nullptr, nullptr, nullptr, "TQ1_0", "TQ2_0", nullptr, nullptr, nullptr, "MXFP4",
        "NVFP4", "Q1_0", "Q2_0",
    };
    static thread_local char buf[16];
    if (type < sizeof(names) / sizeof(names[0]) && names[type]) return names[type];
    std::snprintf(buf, sizeof buf, "type%u", type);
    return buf;
}

}  // namespace flashrt
