// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/experts.hpp"

#include "core/gguf.hpp"
#include "core/scope.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef O_DIRECT
#define O_DIRECT 0   // macOS builds (tests only): fall back to buffered reads
#endif

namespace flashrt::qwen4exp {

namespace {

constexpr size_t kAlign = 4096;
constexpr int kExpertsPerItem = 32;   // one work item: 32 experts of one layer (~44 MB read)

struct AlignedBuf {
    uint8_t* p = nullptr;
    size_t n = 0;
    void reserve(size_t bytes) {
        if (bytes <= n) return;
        std::free(p);
        p = static_cast<uint8_t*>(std::aligned_alloc(kAlign, (bytes + kAlign - 1) & ~(kAlign - 1)));
        if (!p) throw std::bad_alloc();
        n = bytes;
    }
    ~AlignedBuf() { std::free(p); }
};

// Reads [off, off+len) of fd into buf and returns a pointer to byte `off` inside it.
const uint8_t* read_range(int fd, uint64_t off, size_t len, AlignedBuf& buf, uint64_t& bytes_read) {
    const uint64_t a0 = off & ~uint64_t(kAlign - 1);
    const uint64_t a1 = (off + len + kAlign - 1) & ~uint64_t(kAlign - 1);
    buf.reserve(size_t(a1 - a0));
    size_t done = 0;
    while (done < a1 - a0) {
        const ssize_t r = pread(fd, buf.p + done, size_t(a1 - a0 - done), off_t(a0 + done));
        if (r < 0) throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
        if (r == 0) break;   // end of file inside the last aligned block
        done += size_t(r);
    }
    if (done < off - a0 + len) throw std::runtime_error("short read of expert slab");
    bytes_read += done;
    return buf.p + (off - a0);
}

}  // namespace

LoadStats load_experts(const Gguf& g, const Spec& s, ExpertArena& arena, int threads) {
    const q2_0::ExpertShape shape{s.d_model, s.d_ff_expert};
    if (arena.blob_bytes != q2_0::expert_bytes(shape) || arena.n_layer != s.n_layer || arena.n_expert != s.n_expert)
        throw std::runtime_error("load_experts: arena does not match the model");

    struct Slabs { const GgufTensor* t[3]; };   // gate, up, down
    std::vector<Slabs> layers(s.n_layer);
    for (int l = 0; l < s.n_layer; ++l) {
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        for (int k = 0; k < 3; ++k) {
            const std::string n = "blk." + std::to_string(l) + "." + names[k] + ".weight";
            layers[l].t[k] = g.tensor(n);
            if (!layers[l].t[k]) throw std::runtime_error("missing " + n);
        }
    }
    // bytes of one expert's slice of each slab, in ggml layout (same size as repacked)
    const size_t slice[3] = {q2_0::mat_bytes(s.d_ff_expert, s.d_model), q2_0::mat_bytes(s.d_ff_expert, s.d_model),
                             q2_0::mat_bytes(s.d_model, s.d_ff_expert)};

    const int items_per_layer = (s.n_expert + kExpertsPerItem - 1) / kExpertsPerItem;
    const int n_items = s.n_layer * items_per_layer;
    std::atomic<int> next{0};
    std::atomic<uint64_t> total_read{0};
    std::mutex err_mu;
    std::string err;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&] {
            try {
                std::vector<UniqueFd> fds;   // closed on every path, a failed read included
                for (size_t i = 0; i < g.shards.size(); ++i) {
                    fds.emplace_back(open(g.shards[i].c_str(), O_RDONLY | O_DIRECT));
                    if (fds.back().get() < 0) throw std::runtime_error("open " + g.shards[i] + ": " + std::strerror(errno));
                }
                AlignedBuf bufs[3];
                uint64_t read = 0;
                for (int it; (it = next.fetch_add(1)) < n_items;) {
                    const int l = it / items_per_layer;
                    const int e0 = (it % items_per_layer) * kExpertsPerItem;
                    const int ne = std::min(kExpertsPerItem, s.n_expert - e0);
                    const uint8_t* src[3];
                    for (int k = 0; k < 3; ++k) {
                        const GgufTensor* tt = layers[l].t[k];
                        src[k] = read_range(fds[tt->shard].get(), tt->file_offset + uint64_t(e0) * slice[k], size_t(ne) * slice[k],
                                            bufs[k], read);
                    }
                    for (int e = 0; e < ne; ++e)
                        q2_0::repack_expert(src[0] + size_t(e) * slice[0], src[1] + size_t(e) * slice[1],
                                            src[2] + size_t(e) * slice[2], shape, arena.blob(l, e0 + e));
                }
                total_read += read;
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = ex.what();
                next.store(n_items);
            }
        });
    }
    for (auto& t : ts) t.join();
    if (!err.empty()) throw std::runtime_error("load_experts: " + err);
    LoadStats st;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    st.bytes_read = total_read.load();
    return st;
}

}  // namespace flashrt::qwen4exp
