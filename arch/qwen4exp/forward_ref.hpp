// SPDX-License-Identifier: Apache-2.0
// qwen4exp reference forward pass: the correctness-first blocks chained over one sequence, with
// the caches and recurrent states it needs. Every routed expert runs on the CPU (no VRAM expert
// cache yet). Used by the KL gate (tools/fr_kld) and as the baseline for the fast path.
#pragma once

#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/ple.hpp"
#include "core/cpu_pool.hpp"
#include "core/expert_arena.hpp"
#include "core/row_reader.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
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

    // Decode (T == 1) uses the fast MoE path with this cache when set; batches keep the reference
    // path. Routing counts of the reference path accumulate into counts() (for a cache fill).
    void set_fast_moe(const ExpertCache* cache, MoeFastHost* host) { fast_cache_ = cache; fast_host_ = host; }
    // With a manager, every fast decode token also runs one adaptive-cache step.
    void set_cache_manager(CacheManager* m) { cache_mgr_ = m; }
    std::vector<uint32_t>& counts() { return counts_; }

    // Greedy token from a logits row on the device (GPU argmax; 4 bytes come back).
    int32_t argmax(const float* logits_row_dev);

    // Snapshot of the sequence state after a prefill (KV and pooled keys up to pos(), GDN and PLE
    // states, pos(), routing counts), for benchmarks at depth without re-running the prefill.
    // The file is only valid for the same model and the same kernels' numerics.
    void save_state(const std::string& path);
    void load_state(const std::string& path);

    int pos() const { return pos_; }
    cudaStream_t stream() const { return stream_; }

private:
    void state_file(const std::string& path, bool save);

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
    const ExpertCache* fast_cache_ = nullptr;
    MoeFastHost* fast_host_ = nullptr;
    CacheManager* cache_mgr_ = nullptr;
    bool have_access_ = false;   // fast_host_->access holds a token's routing
    std::vector<uint32_t> counts_;
    int32_t* argmax_dev_ = nullptr;
    int32_t* argmax_host_ = nullptr;   // pinned
};

}  // namespace flashrt::qwen4exp
