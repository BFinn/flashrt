// SPDX-License-Identifier: Apache-2.0
// qwen4exp dense weights in VRAM: every tensor the weight plan puts in the VramDense tier,
// in one device allocation. Each tensor starts 256-byte aligned and is followed by
// gemv::kWeightTailPad zero bytes (the mat-vec kernels may read past the last row).
#pragma once

#include "arch/qwen4exp/spec.hpp"
#include "core/formats.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

struct GpuTensor {
    void* dev = nullptr;
    uint32_t type = 0;              // ggml type id
    std::vector<int64_t> dims;      // ggml order: dims[0] is the row length
    size_t bytes = 0;

    int64_t cols() const { return dims.at(0); }
    int64_t rows() const {
        int64_t r = 1;
        for (size_t i = 1; i < dims.size(); ++i) r *= dims[i];
        return r;
    }
};

class GpuWeights {
public:
    GpuWeights() = default;
    ~GpuWeights();
    GpuWeights(const GpuWeights&) = delete;
    GpuWeights& operator=(const GpuWeights&) = delete;

    // Uploads all VramDense tensors of the plan. Throws on CUDA or I/O errors. With q3r, Q3_K
    // matrices (not the token embedding) are converted in place to Q3R (type kTypeQ3R). The layers'
    // hyper-connection down matrices (BF16) become Q8P (kTypeQ8P): half the bytes the decode reads
    // each token for them and the VRAM they take. The up matrices can too, at a KLD cost
    // (FLASHRT_HC_Q8=down (default) | 1 | up | 0; sw66, sw67).
    void load(const Gguf& g, const WeightPlan& plan, bool q3r = true);

    const GpuTensor& get(const std::string& name) const;         // throws if absent
    const GpuTensor* find(const std::string& name) const;        // nullptr if absent
    const GpuTensor& layer(int il, const std::string& suffix) const {   // "blk.<il>.<suffix>"
        return get("blk." + std::to_string(il) + "." + suffix);
    }
    size_t device_bytes() const { return total_; }
    double load_seconds() const { return seconds_; }

private:
    void* base_ = nullptr;
    std::vector<void*> host_bufs_;   // pinned mapped host memory of host-resident tensors
    size_t total_ = 0;
    double seconds_ = 0;
    std::map<std::string, GpuTensor> tensors_;
};

}  // namespace flashrt::qwen4exp
