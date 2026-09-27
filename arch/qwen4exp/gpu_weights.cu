// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/gpu_weights.hpp"

#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemv.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace flashrt::qwen4exp {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t slot_bytes(size_t bytes) { return (bytes + gemv::kWeightTailPad + 255) & ~size_t(255); }

}  // namespace

GpuWeights::~GpuWeights() {
    if (base_) cudaFree(base_);
}

void GpuWeights::load(const Gguf& g, const WeightPlan& plan) {
    const auto t0 = std::chrono::steady_clock::now();
    total_ = 0;
    for (const Placement& p : plan.tensors)
        if (p.tier == Tier::VramDense) total_ += slot_bytes(p.tensor->bytes);
    ck(cudaMalloc(&base_, total_), "cudaMalloc dense weights");
    ck(cudaMemset(base_, 0, total_), "cudaMemset dense weights");

    constexpr size_t kStage = size_t(64) << 20;
    void* stage[2] = {nullptr, nullptr};
    ck(cudaHostAlloc(&stage[0], kStage, cudaHostAllocDefault), "cudaHostAlloc");
    ck(cudaHostAlloc(&stage[1], kStage, cudaHostAllocDefault), "cudaHostAlloc");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "cudaStreamCreate");
    cudaEvent_t done[2];
    ck(cudaEventCreate(&done[0]), "cudaEventCreate");
    ck(cudaEventCreate(&done[1]), "cudaEventCreate");
    ck(cudaEventRecord(done[0], st), "cudaEventRecord");
    ck(cudaEventRecord(done[1], st), "cudaEventRecord");

    std::vector<int> fds(g.shards.size(), -1);
    for (size_t i = 0; i < g.shards.size(); ++i) {
        fds[i] = open(g.shards[i].c_str(), O_RDONLY);
        if (fds[i] < 0) throw std::runtime_error("open " + g.shards[i]);
    }

    size_t off = 0;
    int buf = 0;
    for (const Placement& p : plan.tensors) {
        if (p.tier != Tier::VramDense) continue;
        const GgufTensor& t = *p.tensor;
        GpuTensor gt;
        gt.dev = static_cast<char*>(base_) + off;
        gt.type = t.type;
        gt.dims = t.dims;
        gt.bytes = t.bytes;
        // double-buffered: read chunk k into one staging buffer while chunk k-1 copies
        for (size_t done_b = 0; done_b < t.bytes;) {
            const size_t n = std::min(kStage, t.bytes - done_b);
            ck(cudaEventSynchronize(done[buf]), "cudaEventSynchronize");
            for (size_t r = 0; r < n;) {
                const ssize_t got = pread(fds[t.shard], static_cast<char*>(stage[buf]) + r, n - r, off_t(t.file_offset + done_b + r));
                if (got <= 0) throw std::runtime_error("short read of " + t.name);
                r += size_t(got);
            }
            ck(cudaMemcpyAsync(static_cast<char*>(gt.dev) + done_b, stage[buf], n, cudaMemcpyHostToDevice, st), "cudaMemcpyAsync");
            ck(cudaEventRecord(done[buf], st), "cudaEventRecord");
            buf ^= 1;
            done_b += n;
        }
        tensors_.emplace(t.name, std::move(gt));
        off += slot_bytes(t.bytes);
    }
    ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
    for (int fd : fds) close(fd);
    cudaEventDestroy(done[0]);
    cudaEventDestroy(done[1]);
    cudaStreamDestroy(st);
    cudaFreeHost(stage[0]);
    cudaFreeHost(stage[1]);
    seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const GpuTensor* GpuWeights::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const GpuTensor& GpuWeights::get(const std::string& name) const {
    const GpuTensor* t = find(name);
    if (!t) throw std::runtime_error("GpuWeights: no tensor " + name);
    return *t;
}

}  // namespace flashrt::qwen4exp
