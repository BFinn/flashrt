// SPDX-License-Identifier: Apache-2.0
// qwen4exp reference forward pass: the correctness-first blocks chained over one sequence, with
// the caches and recurrent states it needs. Every routed expert runs on the CPU (no VRAM expert
// cache yet). Used by the KL gate (tools/fr_kld) and as the baseline for the fast path.
#pragma once

#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/ple.hpp"
#include "core/cpu_pool.hpp"
#include "core/expert_arena.hpp"
#include "core/row_reader.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

class ForwardRef {
public:
    // max_ctx: KV capacity; max_batch: tokens per forward() call (scratch sizing).
    ForwardRef(const Gguf& g, const Spec& s, const GpuWeights& w, const ExpertArena& arena, CpuPool& pool, int max_ctx,
               int max_batch);
    ~ForwardRef();
    ForwardRef(const ForwardRef&) = delete;
    ForwardRef& operator=(const ForwardRef&) = delete;

    // Clears every cache and state: the next forward() starts a new sequence at position 0.
    void reset();

    // Runs seq[pos() .. pos() + T) (seq holds the whole sequence so far, for the n-gram context)
    // and advances pos(). Logits of rows [out_from, T) go to logits_dev, [T - out_from][n_vocab].
    void forward(const int32_t* seq, int T, int out_from, float* logits_dev);

    int pos() const { return pos_; }
    cudaStream_t stream() const { return stream_; }

private:
    const Spec& s_;
    const GpuWeights& w_;
    Ple ple_;
    std::unique_ptr<RowReader> reader_;
    PleHost ple_host_;
    MoeHost moe_host_;
    BlockScratch scratch_;
    cudaStream_t stream_ = nullptr;
    std::vector<GdnState> gdn_;
    std::vector<QsaCache> kv_;
    std::vector<PleState> ple_state_;
    float *emb_ = nullptr, *x_ = nullptr, *mixed_ = nullptr, *inject_ = nullptr, *blk_ = nullptr, *pemb_ = nullptr,
          *norm_ = nullptr;
    int max_batch_;
    int pos_ = 0;
};

}  // namespace flashrt::qwen4exp
