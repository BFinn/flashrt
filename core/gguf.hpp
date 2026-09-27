// SPDX-License-Identifier: Apache-2.0
// GGUF reader (format v3), written from the published format description. Reads metadata
// and tensor directories of single or split files ("-00001-of-0000N.gguf"); tensor data
// stays on disk, addressed by (shard, absolute file offset, bytes).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace flashrt {

struct GgufValue;
using GgufArray = std::vector<GgufValue>;
struct GgufValue {
    // integers are widened; the original GGUF type id is kept in `type`
    std::variant<int64_t, uint64_t, double, bool, std::string, std::shared_ptr<GgufArray>> v;
    uint32_t type = 0;

    std::optional<int64_t> as_int() const;
    std::optional<double> as_float() const;
    const std::string* as_string() const;
    const GgufArray* as_array() const;
};

struct GgufTensor {
    std::string name;
    std::vector<int64_t> dims;      // ne0 (fastest) first, as ggml stores them
    uint32_t type = 0;              // ggml type id
    int shard = 0;                  // index into Gguf::shards
    uint64_t file_offset = 0;       // absolute offset of the data in its shard
    uint64_t bytes = 0;             // from the offset gaps in the shard

    int64_t n_elements() const;
};

struct Gguf {
    std::vector<std::string> shards;             // file paths
    uint32_t version = 0;
    std::map<std::string, GgufValue> meta;       // merged over shards (shard 1 wins)
    std::vector<GgufTensor> tensors;             // all shards
    std::map<std::string, size_t> by_name;

    // Opens `path` and, if it is shard 1 of a split, every sibling shard. Throws
    // std::runtime_error on malformed input.
    static Gguf open(const std::string& path);

    const GgufValue* get(const std::string& key) const;
    int64_t get_int(const std::string& key, int64_t fallback) const;
    double get_float(const std::string& key, double fallback) const;
    std::string get_string(const std::string& key, const std::string& fallback = "") const;
    std::vector<int64_t> get_int_array(const std::string& key) const;
    const GgufTensor* tensor(const std::string& name) const;
};

const char* ggml_type_name(uint32_t type);   // "Q2_0", "BF16", ... or "type<N>"

}  // namespace flashrt
