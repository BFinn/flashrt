// SPDX-License-Identifier: Apache-2.0
// Reader for ref_dump files (.frd): llama.cpp intermediate tensors for parity tests.
// Indexes record offsets on open; data is read on demand.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace flashrt {

class Frd {
public:
    struct Rec {
        std::string name;
        int step = 0, type = 0, occurrence = 0;   // occurrence: n-th record of this (name, step)
        int64_t ne[4] = {1, 1, 1, 1};
        uint64_t offset = 0, bytes = 0;
        int64_t elems() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    };

    explicit Frd(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
        if (!f_) throw std::runtime_error("cannot open " + path);
        std::map<std::pair<std::string, int>, int> count;
        for (;;) {
            uint32_t nl;
            if (std::fread(&nl, 4, 1, f_) != 1) break;
            Rec r;
            r.name.resize(nl);
            if (std::fread(r.name.data(), 1, nl, f_) != nl) throw std::runtime_error("truncated frd");
            int32_t hdr[2];
            int64_t ne[4];
            uint64_t nb;
            if (std::fread(hdr, 4, 2, f_) != 2 || std::fread(ne, 8, 4, f_) != 4 || std::fread(&nb, 8, 1, f_) != 1)
                throw std::runtime_error("truncated frd");
            r.step = hdr[0];
            r.type = hdr[1];
            for (int d = 0; d < 4; ++d) r.ne[d] = ne[d];
            r.bytes = nb;
            r.offset = uint64_t(std::ftell(f_));
            r.occurrence = count[{r.name, r.step}]++;
            std::fseek(f_, long(nb), SEEK_CUR);
            index_[std::make_tuple(r.name, r.step, r.occurrence)] = recs_.size();
            recs_.push_back(r);
        }
    }
    ~Frd() {
        if (f_) std::fclose(f_);
    }

    const Rec* find(const std::string& name, int step, int occurrence = 0) const {
        auto it = index_.find(std::make_tuple(name, step, occurrence));
        return it == index_.end() ? nullptr : &recs_[it->second];
    }
    const Rec& get(const std::string& name, int step, int occurrence = 0) const {
        const Rec* r = find(name, step, occurrence);
        if (!r) throw std::runtime_error("frd: no record " + name + " step " + std::to_string(step));
        return *r;
    }
    std::vector<float> floats(const Rec& r) const {
        std::vector<float> v(size_t(r.bytes / 4));
        std::fseek(f_, long(r.offset), SEEK_SET);
        if (std::fread(v.data(), 1, r.bytes, f_) != r.bytes) throw std::runtime_error("frd read");
        return v;
    }
    std::vector<int32_t> ints(const Rec& r) const {
        std::vector<int32_t> v(size_t(r.bytes / 4));
        std::fseek(f_, long(r.offset), SEEK_SET);
        if (std::fread(v.data(), 1, r.bytes, f_) != r.bytes) throw std::runtime_error("frd read");
        return v;
    }

private:
    FILE* f_;
    std::vector<Rec> recs_;
    std::map<std::tuple<std::string, int, int>, size_t> index_;
};

}  // namespace flashrt
