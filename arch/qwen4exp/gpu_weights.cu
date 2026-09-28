// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/gpu_weights.hpp"

#include "core/gguf.hpp"
#include "kernels/cuda/ggml_gemv.h"
#include "kernels/cuda/q3r.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace flashrt::qwen4exp {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t slot_bytes(size_t bytes) { return (bytes + gemv::kWeightTailPad + 255) & ~size_t(255); }

// Tall Q3_K matrices (>= 4096 rows, K <= 4096: attn_gate, attn_qkv, attn_q) are converted in
// place to Q3R (kernels/cuda/q3r.h), where it is 1.5x faster than ggml's MMVQ. Short or wide
// ones (ssm_out, the shared experts, attn_k/v) stay ggml Q3_K, which is faster there
// (bench/results/2026-09-28-sw16-q3r); so does the token embedding (read one row at a time).
bool q3r_eligible(const GgufTensor& t) {
    return t.type == 11 /* GGML_TYPE_Q3_K */ && t.name != "token_embd.weight" && t.dims.size() == 2 && t.dims[0] % 256 == 0 &&
           t.dims[0] <= 4096 && t.dims[1] >= 4096;
}
// The token embedding is read one row per token, so it lives in pinned mapped host memory (the
// GPU reads its rows over PCIe) and its 260 MB of VRAM go to the expert cache.
bool host_resident(const GgufTensor& t) { return t.name == "token_embd.weight"; }

// The layers' hyper-connection down and up matrices (BF16, 1.26 GB in all) are read in full every
// decode token; as Q8P they take half the bytes and half the VRAM (FLASHRT_HC_Q8=0: kept BF16).
bool hc_q8_on() {
    static const bool on = [] {
        const char* e = std::getenv("FLASHRT_HC_Q8");
        return !(e && e[0] == '0');
    }();
    return on;
}
bool hc_q8_eligible(const GgufTensor& t) {
    auto ends = [&](const char* suf) {
        const std::string sfx(suf);
        return t.name.size() > sfx.size() && t.name.compare(t.name.size() - sfx.size(), sfx.size(), sfx) == 0;
    };
    return hc_q8_on() && t.type == 30 /* GGML_TYPE_BF16 */ && t.dims.size() == 2 && t.dims[0] % 32 == 0 && t.name.rfind("blk.", 0) == 0 &&
           (ends(".hc_attn_down.weight") || ends(".hc_attn_up.weight") || ends(".hc_ffn_down.weight") || ends(".hc_ffn_up.weight"));
}
size_t q8p_bytes(const GgufTensor& t) { return size_t(t.dims[0]) * t.dims[1] * 17 / 16; }   // int8 values + an fp16 scale per 32

size_t tensor_slot(const GgufTensor& t, bool q3r_on) {
    size_t b = t.bytes;
    if (q3r_on && q3r_eligible(t)) b = std::max(b, q3r::bytes(t.dims[1], t.dims[0]));
    if (hc_q8_eligible(t)) b = q8p_bytes(t);
    return slot_bytes(b);
}

// BF16 -> Q8P (Q8_0's quantization: d = amax / 127, q = round(x / d)); one warp per 32 values
__global__ void k_bf16_to_q8p(const uint16_t* in, int8_t* q, __half* d, size_t n_blocks) {
    const size_t w = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= n_blocks) return;
    const float v = __uint_as_float(uint32_t(in[w * 32 + lane]) << 16);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f;
    q[w * 32 + lane] = int8_t(roundf(v * id));
    if (lane == 0) d[w] = __float2half(dd);
}

}  // namespace

GpuWeights::~GpuWeights() {
    if (base_) cudaFree(base_);
    for (void* h : host_bufs_) cudaFreeHost(h);
}

void GpuWeights::load(const Gguf& g, const WeightPlan& plan, bool q3r_on) {
    const auto t0 = std::chrono::steady_clock::now();
    total_ = 0;
    for (const Placement& p : plan.tensors)
        if (p.tier == Tier::VramDense && !host_resident(*p.tensor)) total_ += tensor_slot(*p.tensor, q3r_on);
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

    // hc matrices go through a BF16 staging copy on the device, converted into their Q8P slot
    void* hc_tmp = nullptr;
    {
        size_t hb = 0;
        for (const Placement& p : plan.tensors)
            if (p.tier == Tier::VramDense && hc_q8_eligible(*p.tensor)) hb = std::max<size_t>(hb, p.tensor->bytes);
        if (hb) ck(cudaMalloc(&hc_tmp, hb), "cudaMalloc hc staging");
    }
    size_t off = 0;
    int buf = 0;
    for (const Placement& p : plan.tensors) {
        if (p.tier != Tier::VramDense) continue;
        const GgufTensor& t = *p.tensor;
        GpuTensor gt;
        if (host_resident(t)) {
            void* h = nullptr;
            ck(cudaHostAlloc(&h, t.bytes, cudaHostAllocMapped), "cudaHostAlloc host-resident tensor");
            for (size_t r = 0; r < t.bytes;) {
                const ssize_t got = pread(fds[t.shard], static_cast<char*>(h) + r, t.bytes - r, off_t(t.file_offset + r));
                if (got <= 0) throw std::runtime_error("short read of " + t.name);
                r += size_t(got);
            }
            ck(cudaHostGetDevicePointer(&gt.dev, h, 0), "host-resident tensor device pointer");
            gt.type = t.type;
            gt.dims = t.dims;
            gt.bytes = t.bytes;
            host_bufs_.push_back(h);
            tensors_.emplace(t.name, std::move(gt));
            continue;
        }
        gt.dev = static_cast<char*>(base_) + off;
        gt.type = t.type;
        gt.dims = t.dims;
        gt.bytes = t.bytes;
        const bool to_q8 = hc_q8_eligible(t);
        char* dst = static_cast<char*>(to_q8 ? hc_tmp : gt.dev);
        // double-buffered: read chunk k into one staging buffer while chunk k-1 copies
        for (size_t done_b = 0; done_b < t.bytes;) {
            const size_t n = std::min(kStage, t.bytes - done_b);
            ck(cudaEventSynchronize(done[buf]), "cudaEventSynchronize");
            for (size_t r = 0; r < n;) {
                const ssize_t got = pread(fds[t.shard], static_cast<char*>(stage[buf]) + r, n - r, off_t(t.file_offset + done_b + r));
                if (got <= 0) throw std::runtime_error("short read of " + t.name);
                r += size_t(got);
            }
            ck(cudaMemcpyAsync(dst + done_b, stage[buf], n, cudaMemcpyHostToDevice, st), "cudaMemcpyAsync");
            ck(cudaEventRecord(done[buf], st), "cudaEventRecord");
            buf ^= 1;
            done_b += n;
        }
        if (to_q8) {   // after its copies on the same stream; the next tensor's copies wait for it
            const size_t ne = size_t(t.dims[0]) * t.dims[1];
            int8_t* q = static_cast<int8_t*>(gt.dev);
            k_bf16_to_q8p<<<unsigned((ne + 255) / 256), 256, 0, st>>>(static_cast<const uint16_t*>(hc_tmp), q,
                                                                       reinterpret_cast<__half*>(q + ne), ne / 32);
            ck(cudaGetLastError(), "hc to Q8P");
            gt.type = kTypeQ8P;
            gt.bytes = q8p_bytes(t);
        }
        tensors_.emplace(t.name, std::move(gt));
        off += tensor_slot(t, q3r_on);
    }
    if (hc_tmp) {
        ck(cudaStreamSynchronize(st), "hc to Q8P");
        cudaFree(hc_tmp);
    }
    // Q3_K -> Q3R in place, through one temporary (the slots were sized for the larger of the two)
    if (q3r_on) {
        size_t tmp_bytes = 0;
        for (const Placement& p : plan.tensors)
            if (p.tier == Tier::VramDense && q3r_eligible(*p.tensor))
                tmp_bytes = std::max(tmp_bytes, q3r::bytes(p.tensor->dims[1], p.tensor->dims[0]));
        if (tmp_bytes) {
            void* tmp = nullptr;
            ck(cudaMalloc(&tmp, tmp_bytes), "cudaMalloc q3r temporary");
            for (const Placement& p : plan.tensors) {
                if (p.tier != Tier::VramDense || !q3r_eligible(*p.tensor)) continue;
                GpuTensor& t = tensors_.at(p.tensor->name);
                const size_t qb = q3r::bytes(t.rows(), t.cols());
                q3r::repack(t.dev, tmp, t.rows(), t.cols(), st);
                ck(cudaMemcpyAsync(t.dev, tmp, qb, cudaMemcpyDeviceToDevice, st), "q3r copy back");
                t.type = kTypeQ3R;
                t.bytes = qb;
            }
            ck(cudaStreamSynchronize(st), "q3r repack");
            cudaFree(tmp);
        }
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
