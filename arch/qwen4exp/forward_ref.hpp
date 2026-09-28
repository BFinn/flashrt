// SPDX-License-Identifier: Apache-2.0
// qwen4exp reference forward pass: the correctness-first blocks chained over one sequence, with
// the caches and recurrent states it needs. Every routed expert runs on the CPU (no VRAM expert
// cache yet). Used by the KL gate (tools/fr_kld) and as the baseline for the fast path.
#pragma once

#include "arch/qwen4exp/blocks.hpp"
#include "arch/qwen4exp/moe_fast.hpp"
#include "arch/qwen4exp/moe_stream.hpp"
#include "arch/qwen4exp/ple.hpp"
#include "core/cpu_pool.hpp"
#include "core/expert_arena.hpp"
#include "core/row_reader.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <future>
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
    // T above max_batch is a prefill chunk: its own buffers, sized on first use, and the MoE on
    // the GPU with each layer's experts streamed from the host arena (moe_stream.hpp);
    // release_chunk_buffers() gives that memory back.
    void forward(const int32_t* seq, int T, int out_from, float* logits_dev);
    void release_chunk_buffers();
    size_t chunk_buffer_bytes() const;
    // Device memory a prefill chunk of T tokens takes, for a prompt ending at end_pos: its
    // buffers, the expert stream, the scratch its products and QSA grow, and in hot-set mode the
    // KV mirror (estimated from the allocation formulas, plus a margin).
    size_t chunk_bytes(int T, int end_pos) const;
    // A chunk length for n prompt tokens ending at end_pos: the largest (up to max_chunk, in steps
    // of 1,024) whose chunk_bytes fit free_bytes less 256 MiB, then evened out over the chunks it
    // takes. Longer chunks give each expert more tokens (fuller tensor-core tiles).
    int pick_chunk(int n, int end_pos, size_t free_bytes, int max_chunk = 16384) const;
    // The whole prompt a run of chunks comes from (valid until the prefill ends, or nullptr):
    // while a chunk computes, the next chunk's n-gram rows are read from the SSD.
    void set_prefill_lookahead(const int32_t* seq, int n) {
        look_seq_ = seq;
        look_n_ = n;
    }

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
    // Keeps the first n (0 <= n <= T) tokens of the last window: the recurrent states are rewound
    // and pos() becomes the window's start + n (n = 0 undoes the window). The KV caches need
    // nothing: the positions past it are rewritten before anything reads them.
    void commit(int n);

    // A device copy of the recurrent state at pos() (GDN states, conv and PLE histories, the
    // indexer rings); restore_checkpoint() returns the sequence to that position. As for
    // commit(), the KV caches need nothing. For prompt-prefix reuse across requests.
    void save_checkpoint();
    void restore_checkpoint();
    int checkpoint_pos() const { return ckpt_pos_; }

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
    struct Graphs;
    Graphs& capture_graphs(int T, float* logits_dev);
    void drop_graphs();

    const Spec& s_;
    const GpuWeights& w_;
    Ple ple_;
    std::unique_ptr<RowReader> reader_;
    PleHost ple_host_;
    MoeHost moe_host_;
    // working buffers: one set for decode steps and batches up to max_batch, one for chunks
    struct Bufs {
        float *emb = nullptr, *x = nullptr, *mixed = nullptr, *inject = nullptr, *blk = nullptr, *pemb = nullptr, *norm = nullptr;
        int cap = 0;
        BlockScratch scratch;
    };
    void alloc_bufs(Bufs& b, int T);
    void free_bufs(Bufs& b);
    void use_bufs(Bufs& b);
    Bufs dec_, chunk_;
    BlockScratch* scr_ = nullptr;   // the current set's scratch
    ExpertStream* estream_ = nullptr;
    uint32_t* counts_dev_ = nullptr;   // routing counts of chunks, [n_layer][n_expert]
    bool in_chunk_ = false;
    // PLE lookahead: the next chunk's rows, read on another thread into their own pinned buffer
    const int32_t* look_seq_ = nullptr;
    int look_n_ = 0;
    PleHost ple_next_;
    std::future<void> ple_next_rows_;
    int ple_next_pos_ = -1, ple_next_T_ = 0;
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
    // graph mode: one pair of graphs per (step length, window mode)
    bool use_graphs_ = true;
    int32_t* params_dev_ = nullptr;    // BlockCtx::dparams: [token 0, position, doorbell seq, token 1, ...]
    int32_t* params_host_ = nullptr;   // pinned; copied to params_dev_ by the first graph node
    struct Graphs {
        cudaGraphExec_t pre = nullptr, post = nullptr;
        float* logits = nullptr;
        const void* ple_pinned = nullptr;
        const void* ple_dev = nullptr;
        BlockScratch scratch;   // the buffers it was captured with
    };
    Graphs graphs_[kMaxGraphTokens + 1][2];
    long graph_captures_ = 0;
    int32_t* argmax_host_ = nullptr;   // pinned
    // speculative windows
    int max_window_ = 0;
    bool in_window_ = false;          // enqueueing a window (the layers keep their rewind data)
    int window_T_ = 0, window_pos0_ = -1;
    std::vector<GdnWindow> gdn_win_;
    std::vector<PleWindow> ple_win_;
    // checkpoint: one device buffer holding every piece, and where each piece lives
    void* ckpt_ = nullptr;
    size_t ckpt_bytes_ = 0;
    int ckpt_pos_ = -1;
    std::vector<std::pair<void*, size_t>> ckpt_parts();
};

}  // namespace flashrt::qwen4exp
