// SPDX-License-Identifier: Apache-2.0
// core/gguf on synthetic files: a two-shard split with every metadata type, nested arrays and
// tensors (shard index, absolute offset and size from the gaps, alignment), then malformed files,
// which must throw std::runtime_error (not crash, not bad_alloc): truncation, a bad magic, a zero
// alignment, a tensor offset past the end, an array claiming 2^32 elements.
#include "core/gguf.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

using namespace flashrt;

namespace {
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

// A little-endian GGUF v3 writer: metadata, a tensor directory, then the data section.
struct Writer {
    std::vector<uint8_t> b;
    template <class T> Writer& pod(T v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof v);
        return *this;
    }
    Writer& str(const std::string& s) {
        pod<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
        return *this;
    }
    Writer& key(const std::string& k, uint32_t type) { return str(k).pod<uint32_t>(type); }
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;   // in the data section
};

// header + kv (written by `kv`, n_kv of them) + tensors + padding to `align` + `data` bytes
std::vector<uint8_t> file(uint32_t magic, uint64_t n_kv, const std::function<void(Writer&)>& kv, const std::vector<Tensor>& ts,
                          uint64_t align, size_t data) {
    Writer w;
    w.pod<uint32_t>(magic).pod<uint32_t>(3).pod<uint64_t>(ts.size()).pod<uint64_t>(n_kv);
    kv(w);
    for (const Tensor& t : ts) {
        w.str(t.name).pod<uint32_t>(uint32_t(t.dims.size()));
        for (uint64_t d : t.dims) w.pod<uint64_t>(d);
        w.pod<uint32_t>(t.type).pod<uint64_t>(t.offset);
    }
    if (align) w.b.resize((w.b.size() + align - 1) / align * align, 0);
    w.b.resize(w.b.size() + data, 0xAB);
    return w.b;
}

void save(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

bool rejects(const std::string& path) {
    try {
        Gguf::open(path);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;   // bad_alloc and the like: a crash-like failure
    }
    return false;
}
}  // namespace

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("flashrt-test-gguf-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const std::string s1 = (dir / "m-00001-of-00002.gguf").string(), s2 = (dir / "m-00002-of-00002.gguf").string();

    // shard 1: every scalar type, a string, arrays (ints, strings, nested), alignment 64; two tensors
    save(s1, file(0x46554747, 16,
                  [](Writer& w) {
                      w.key("general.architecture", 8).str("qwen4exp");
                      w.key("general.alignment", 4).pod<uint32_t>(64);
                      w.key("split.count", 2).pod<uint16_t>(2);
                      w.key("u8", 0).pod<uint8_t>(250);
                      w.key("i8", 1).pod<int8_t>(-5);
                      w.key("i16", 3).pod<int16_t>(-300);
                      w.key("u32", 4).pod<uint32_t>(4000000000u);
                      w.key("i32", 5).pod<int32_t>(-7);
                      w.key("f32", 6).pod<float>(0.5f);
                      w.key("bool", 7).pod<uint8_t>(1);
                      w.key("u64", 10).pod<uint64_t>(1ull << 40);
                      w.key("i64", 11).pod<int64_t>(-(1ll << 40));
                      w.key("f64", 12).pod<double>(2.25);
                      w.key("layers", 9).pod<uint32_t>(5).pod<uint64_t>(3).pod<int32_t>(1).pod<int32_t>(5).pod<int32_t>(9);
                      w.key("words", 9).pod<uint32_t>(8).pod<uint64_t>(2).str("a").str("\xe4\xb8\xad");
                      w.key("nested", 9).pod<uint32_t>(9).pod<uint64_t>(1).pod<uint32_t>(4).pod<uint64_t>(2).pod<uint32_t>(1).pod<uint32_t>(2);
                  },
                  {{"tok_embd.weight", {64, 10}, 30, 0}, {"blk.0.attn_q.weight", {64, 4}, 8, 1280}}, 64, 1280 + 272));
    // shard 2: its own split metadata (shard 1's values win) and one tensor
    save(s2, file(0x46554747, 2,
                  [](Writer& w) {
                      w.key("general.architecture", 8).str("other");
                      w.key("split.count", 2).pod<uint16_t>(2);
                  },
                  {{"output.weight", {64, 3}, 0, 0}}, 32, 768));

    try {
        const Gguf g = Gguf::open(s1);
        expect(g.shards.size() == 2 && g.version == 3, "two shards, version 3");
        expect(g.get_string("general.architecture") == "qwen4exp", "shard 1's value wins");
        expect(g.get_int("u8", 0) == 250 && g.get_int("i8", 0) == -5 && g.get_int("i16", 0) == -300, "small ints");
        expect(g.get_int("u32", 0) == 4000000000ll && g.get_int("i32", 0) == -7, "32-bit ints");
        expect(g.get_int("u64", 0) == (1ll << 40) && g.get_int("i64", 0) == -(1ll << 40), "64-bit ints");
        expect(g.get_float("f32", 0) == 0.5 && g.get_float("f64", 0) == 2.25 && g.get_float("i32", 0) == -7.0, "floats");
        expect(g.get_int("bool", 0) == 1 && g.get_int("missing", 42) == 42 && g.get_string("missing", "x") == "x", "bool, fallbacks");
        expect(g.get_int_array("layers") == std::vector<int64_t>{1, 5, 9} && g.get_int_array("i32") == std::vector<int64_t>{-7},
               "int arrays, a scalar as an array");
        const GgufArray* words = g.get("words") ? g.get("words")->as_array() : nullptr;
        expect(words && words->size() == 2 && *(*words)[1].as_string() == "\xe4\xb8\xad", "string array");
        const GgufArray* nested = g.get("nested") ? g.get("nested")->as_array() : nullptr;
        expect(nested && nested->size() == 1 && (*nested)[0].as_array() && (*nested)[0].as_array()->size() == 2, "nested array");

        const GgufTensor* e = g.tensor("tok_embd.weight");
        const GgufTensor* q = g.tensor("blk.0.attn_q.weight");
        const GgufTensor* o = g.tensor("output.weight");
        expect(e && q && o && !g.tensor("nope"), "tensors by name");
        if (e && q && o) {
            const uint64_t size1 = std::filesystem::file_size(s1), size2 = std::filesystem::file_size(s2);
            expect(e->shard == 0 && q->shard == 0 && o->shard == 1, "shard indexes");
            expect(e->file_offset % 64 == 0 && q->file_offset == e->file_offset + 1280, "absolute offsets, alignment 64");
            expect(e->bytes == 1280 && q->bytes == size1 - q->file_offset && q->bytes == 272, "sizes from the gaps");
            expect(o->file_offset % 32 == 0 && o->file_offset + o->bytes == size2 && o->bytes == 768, "shard 2's tensor");
            expect(e->dims == std::vector<int64_t>{64, 10} && e->n_elements() == 640 && std::string(ggml_type_name(e->type)) == "BF16",
                   "dims and type");
            FILE* f = std::fopen(s1.c_str(), "rb");   // the data is where the offset says
            unsigned char c = 0;
            std::fseek(f, long(q->file_offset), SEEK_SET);
            expect(std::fread(&c, 1, 1, f) == 1 && c == 0xAB, "data at the offset");
            std::fclose(f);
        }
    } catch (const std::exception& ex) {
        expect(false, std::string("open threw: ") + ex.what());
    }

    // a missing second shard
    std::filesystem::remove(s2);
    expect(rejects(s1), "rejects a split with a shard missing");

    // malformed single files
    const std::string bad = (dir / "bad.gguf").string();
    auto one_kv = [](Writer& w) { w.key("general.architecture", 8).str("x"); };
    const std::vector<uint8_t> good = file(0x46554747, 1, one_kv, {{"t", {32}, 0, 0}}, 32, 128);
    save(bad, good);
    expect(!rejects(bad), "the control file opens");
    for (size_t cut : {size_t(3), size_t(20), size_t(60), size_t(90)}) {   // header, metadata, tensor directory (it ends at 98)
        save(bad, std::vector<uint8_t>(good.begin(), good.begin() + long(cut)));
        expect(rejects(bad), "rejects a file cut at " + std::to_string(cut));
    }
    save(bad, file(0x46554748, 1, one_kv, {}, 32, 0));
    expect(rejects(bad), "rejects a bad magic");
    save(bad, file(0x46554747, 1, [](Writer& w) { w.key("general.alignment", 4).pod<uint32_t>(0); }, {{"t", {32}, 0, 0}}, 32, 128));
    expect(rejects(bad), "rejects alignment 0");
    save(bad, file(0x46554747, 1, [](Writer& w) { w.key("general.alignment", 4).pod<uint32_t>(48); }, {{"t", {32}, 0, 0}}, 48, 128));
    expect(rejects(bad), "rejects an alignment that is not a power of two");
    save(bad, file(0x46554747, 1, one_kv, {{"t", {32}, 0, 4096}}, 32, 128));
    expect(rejects(bad), "rejects a tensor offset past the end");
    save(bad, file(0x46554747, 1, [](Writer& w) { w.key("a", 9).pod<uint32_t>(4).pod<uint64_t>(1ull << 32).pod<uint32_t>(1); }, {}, 0, 0));
    expect(rejects(bad), "rejects an array claiming 2^32 elements (runtime_error, not bad_alloc)");
    save(bad, file(0x46554747, 1, [](Writer& w) { w.key("a", 99).pod<uint32_t>(1); }, {}, 0, 0));
    expect(rejects(bad), "rejects an unknown value type");
    save(bad, file(0x46554747, 1, one_kv, {{"t", {32, 1, 1, 1, 1, 1, 1, 1, 1}, 0, 0}}, 32, 128));
    expect(rejects(bad), "rejects a tensor of rank 9");

    std::filesystem::remove_all(dir);
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
