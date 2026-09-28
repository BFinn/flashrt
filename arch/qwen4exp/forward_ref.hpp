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
    // max_ctx: KV capacity; max_batch: tokens per forward() call (scratch sizing); kv_q8: the
    // QSA KV cache in Q8_0 (llama.cpp's q8_0 cache) instead of fp16.
    // kv_hot_blocks > 0 (q8 only): the KV cache in host memory with that many 4-cell blocks per
    // layer hot on the GPU (see QsaCache).
    ForwardRef(const Gguf& g, const Spec& s, const GpuWeights& w, const ExpertArena& arena, CpuPool& pool, int max_ctx,
               int max_batch, bool kv_q8 = false, int kv_hot_blocks = 0);
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
    void set_fast_moe(const ExpertCache* cache, MoeFastHost* host) {
        fast_cache_ = cache;
        fast_host_ = host;
        drop_graphs();
    }
    // Decode tokens (T == 1, fast path with doorbells) replay two captured CUDA graphs instead of
    // launching ~1,600 kernels: one up to the first PLE layer, one after it (the PLE rows are
    // read from the SSD in between). On by default; any other forward() drops them.
    void set_graphs(bool on) {
        use_graphs_ = on;
        drop_graphs();
    }
    long graph_captures() const { return graph_captures_; }
    // With a manager, every fast decode token also runs one adaptive-cache step.
    void set_cache_manager(CacheManager* m) { cache_mgr_ = m; }
    std::vector<uint32_t>& counts() { return counts_; }
    // The reference path halves counts() every `tokens` tokens (0 = never), so a long prompt's
    // counts favour its recent text, which predicts the decode better (default 4096).
    void set_count_half_life(int tokens) { count_half_life_ = tokens; }

    // Speculative verify windows of up to W tokens (allocates what a rewind needs; the fast MoE
    // host must have max_window >= W).
    void enable_windows(int W);
    // Runs seq[pos() .. pos() + T) (T <= W) on the fast path in one doorbell step, every row's
    // logits to logits_dev [T][n_vocab], and advances pos() by T. commit() must follow.
    void forward_window(const int32_t* seq, int T, float* logits_dev);
    // Keeps the first n (1 <= n <= T) tokens of the last window: the recurrent states are rewound
    // and pos() becomes the window's start + n. The KV caches need nothing: the positions past
    // it are rewritten before anything reads them.
    void commit(int n);

    // Greedy token from a logits row on the device (GPU argmax; 4 bytes come back).
    int32_t argmax(const float* logits_row_dev);

    // Snapshot of the sequence state after a prefill (KV and pooled keys up to pos(), GDN and PLE
    // states, pos(), routing counts), for benchmarks at depth without re-running the prefill.
    // The file is only valid for the same model and the same kernels' numerics. An fp16-KV file
    // loads into a q8 cache (converted on the GPU); not the other way.
    void save_state(const std::string& path);
    void load_state(const std::string& path);

    // The final hyper-connection streams [T][hc][d_model] of the last forward()'s rows (before the
    // output mix): the h an MTP draft head takes. Valid until the next forward().
    const float* streams() const { return x_; }

    int pos() const { return pos_; }
    cudaStream_t stream() const { return stream_; }

private:
    void state_file(const std::string& path, bool save);
    void enqueue_pre(const BlockCtx& c, const int32_t* seq, int T);
    void enqueue_post(const BlockCtx& c, int T, int out_from, float* logits_dev);
    void enqueue_layer(const BlockCtx& c, int il, int T);
    int first_ple_layer() const;
    bool graph_eligible(int T, int out_from, float* logits_dev) const;
    void capture_graphs(float* logits_dev);
    void drop_graphs();

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
    int count_half_life_ = 4096;
    long count_tokens_ = 0;   // tokens counted since the last halving
    int32_t* argmax_dev_ = nullptr;
    // graph mode
    bool use_graphs_ = true;
    int32_t* params_dev_ = nullptr;    // [token, position, doorbell seq]
    int32_t* params_host_ = nullptr;   // pinned; copied to params_dev_ by the first graph node
    cudaGraphExec_t graph_pre_ = nullptr, graph_post_ = nullptr;
    float* graph_logits_ = nullptr;
    const void* graph_ple_pinned_ = nullptr;
    const void* graph_ple_dev_ = nullptr;
    long graph_captures_ = 0;
    int32_t* argmax_host_ = nullptr;   // pinned
    // speculative windows
    int max_window_ = 0;
    bool in_window_ = false;          // enqueueing a window (the layers keep their rewind data)
    int window_T_ = 0, window_pos0_ = -1;
    std::vector<GdnWindow> gdn_win_;
    std::vector<PleWindow> ple_win_;
};

}  // namespace flashrt::qwen4exp
