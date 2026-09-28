// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/forward_ref.hpp"

#include "core/gguf.hpp"
#include "core/platform.hpp"
#include "kernels/cuda/ggml_gemm.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>
#include <set>
#include <stdexcept>
#include <string>

namespace flashrt::qwen4exp {

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
float* dalloc(size_t elems) {
    float* p = nullptr;
    ck(cudaMalloc(&p, elems * 4), "cudaMalloc forward buffer");
    return p;
}
}  // namespace

ForwardRef::ForwardRef(const Gguf& g, const Spec& s, const GpuWeights& w, const ExpertArena& arena, CpuPool& pool,
                       int max_ctx, int max_batch, bool kv_q8, int kv_hot_blocks)
    : s_(s), w_(w), max_batch_(max_batch) {
    ple_ = parse_ple(g);
    reader_ = std::make_unique<RowReader>(g.shards[ple_.table_shard], ple_.table_offset, ple_.row_bytes, 64);
    ple_host_.ple = &ple_;
    ple_host_.reader = reader_.get();
    ple_next_.ple = &ple_;
    ple_next_.reader = reader_.get();
    moe_host_.arena = &arena;
    moe_host_.pool = &pool;
    counts_.assign(size_t(s.n_layer) * s.n_expert, 0);
    moe_host_.counts = &counts_;
    ck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    gdn_.resize(s.n_layer);
    kv_.resize(s.n_layer);
    ple_state_.resize(s.n_layer);
    for (int il : s.gdn_layers) gdn_[il] = alloc_gdn_state(s);
    for (int il : s.qsa_layers) kv_[il] = alloc_qsa_cache(s, max_ctx, kv_q8, kv_hot_blocks);
    for (int il : s.ple_layers) ple_state_[il] = alloc_ple_state(s, ple_);
    alloc_bufs(dec_, max_batch);
    use_bufs(dec_);
    ck(cudaMalloc(&argmax_dev_, 4), "cudaMalloc argmax");
    ck(cudaMalloc(&params_dev_, 16 * sizeof(int32_t)), "cudaMalloc decode params");
    ck(cudaHostAlloc(&params_host_, 16 * sizeof(int32_t), cudaHostAllocDefault), "cudaHostAlloc decode params");
    ck(cudaHostAlloc(&argmax_host_, 4, cudaHostAllocDefault), "cudaHostAlloc argmax");
}

ForwardRef::~ForwardRef() {
    for (int il : s_.gdn_layers) free_gdn_state(gdn_[il]);
    for (int il : s_.qsa_layers) free_qsa_cache(kv_[il]);
    for (int il : s_.ple_layers) free_ple_state(ple_state_[il]);
    for (GdnWindow& w : gdn_win_) free_gdn_window(w);
    for (PleWindow& w : ple_win_) free_ple_window(w);
    if (ckpt_) cudaFree(ckpt_);
    free_bufs(dec_);
    release_chunk_buffers();
    if (ple_next_rows_.valid()) ple_next_rows_.wait();
    if (ple_host_.raw_dev) cudaFree(ple_host_.raw_dev);
    if (ple_host_.raw_pinned) cudaFreeHost(ple_host_.raw_pinned);
    if (ple_next_.raw_pinned) cudaFreeHost(ple_next_.raw_pinned);
    drop_graphs();
    cudaFree(argmax_dev_);
    cudaFree(params_dev_);
    cudaFreeHost(params_host_);
    cudaFreeHost(argmax_host_);
    cudaStreamDestroy(stream_);
}

namespace {
// one device buffer to or from the file, through a host bounce buffer
void state_io(FILE* f, void* dev, size_t bytes, bool save, std::vector<uint8_t>& bounce) {
    bounce.resize(bytes);
    if (save) {
        ck(cudaMemcpy(bounce.data(), dev, bytes, cudaMemcpyDefault), "state to host");
        if (std::fwrite(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file write failed");
    } else {
        if (std::fread(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file truncated");
        ck(cudaMemcpy(dev, bounce.data(), bytes, cudaMemcpyDefault), "state to device");
    }
}
}  // namespace

void ForwardRef::save_state(const std::string& path) { state_file(path, true); }
void ForwardRef::load_state(const std::string& path) { state_file(path, false); }

void ForwardRef::state_file(const std::string& path, bool save) {
    ck(cudaStreamSynchronize(stream_), "state");
    FILE* f = std::fopen(path.c_str(), save ? "wb" : "rb");
    if (!f) throw std::runtime_error("cannot open state file " + path);
    const Spec& s = s_;
    // header: magic, n_layer, pos, n_counts, and (version 2) the KV format (0 fp16, 1 q8)
    const int64_t magic1 = 0x46525354, magic2 = 0x46525332;   // "FRST", "FRS2"
    const bool q8 = kv_[s.qsa_layers.front()].q8;
    bool file_q8 = q8;
    if (save) {
        const int64_t hdr[5] = {magic2, s.n_layer, pos_, int64_t(counts_.size()), q8 ? 1 : 0};
        std::fwrite(hdr, sizeof(hdr), 1, f);
    } else {
        int64_t h[5] = {0, 0, 0, 0, 0};
        const bool ok4 = std::fread(h, sizeof(int64_t), 4, f) == 4;
        if (ok4 && h[0] == magic2 && std::fread(&h[4], sizeof(int64_t), 1, f) != 1) h[0] = 0;
        if (!ok4 || (h[0] != magic1 && h[0] != magic2) || h[1] != s.n_layer || h[3] != int64_t(counts_.size())) {
            std::fclose(f);
            throw std::runtime_error("state file does not match this model");
        }
        file_q8 = h[0] == magic2 && h[4] == 1;
        if (file_q8 && !q8) {
            std::fclose(f);
            throw std::runtime_error("state file has a q8 KV cache; this cache is fp16");
        }
        pos_ = int(h[2]);
        if (pos_ > kv_[s.qsa_layers.front()].capacity) {
            std::fclose(f);
            throw std::runtime_error("state file holds more positions than the KV capacity");
        }
    }
    std::vector<uint8_t> bounce;
    const int gch = 2 * s.ssm_groups * s.ssm_state + s.ssm_heads * s.ssm_state;
    for (int il : s.gdn_layers) {
        state_io(f, gdn_[il].S, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4, save, bounce);
        state_io(f, gdn_[il].conv, size_t(s.ssm_conv - 1) * gch * 4, save, bounce);
    }
    for (int il : s.qsa_layers) qsa_state_io(f, s, kv_[il], pos_, save, file_q8, stream_);
    for (int il : s.ple_layers) state_io(f, ple_state_[il].hist, size_t(ple_.ngram) * 3 * size_t(s.hc_count) * s.d_model * 4 + 4, save, bounce);
    if (save) {
        std::fwrite(counts_.data(), 4, counts_.size(), f);
    } else if (std::fread(counts_.data(), 4, counts_.size(), f) != counts_.size()) {
        std::fclose(f);
        throw std::runtime_error("state file truncated");
    }
    std::fclose(f);
}

int32_t ForwardRef::argmax(const float* logits_row_dev) {
    argmax_dev(stream_, logits_row_dev, s_.n_vocab, argmax_dev_);
    ck(cudaMemcpyAsync(argmax_host_, argmax_dev_, 4, cudaMemcpyDeviceToHost, stream_), "argmax to host");
    ck(cudaStreamSynchronize(stream_), "argmax");
    return *argmax_host_;
}

void ForwardRef::reset() {
    for (int il : s_.gdn_layers) reset_gdn_state(s_, gdn_[il], stream_);
    for (int il : s_.ple_layers) reset_ple_state(s_, ple_, ple_state_[il], stream_);
    for (int il : s_.qsa_layers) {   // K/V and pooled keys are overwritten as positions advance; the ring is not
        ck(cudaMemsetAsync(kv_[il].idx_ring, 0, size_t(qsa_ring_slots(s_)) * s_.idx_dim * 4, stream_), "memset ring");
        reset_qsa_hot(s_, kv_[il], stream_);
    }
    ck(cudaStreamSynchronize(stream_), "reset");
    pos_ = 0;
}

// embedding, hyper-connection streams, and the layers before the first PLE layer
void ForwardRef::enqueue_pre(const BlockCtx& c, const int32_t* seq, int T) {
    embed(c, seq ? seq + pos_ : nullptr, T, emb_);
    hc_init(c, emb_, x_, T);
    for (int il = 0; il < first_ple_layer(); ++il) enqueue_layer(c, il, T);
}


// the PLE rows (already in ple_host_.raw_pinned), the remaining layers, and the head
void ForwardRef::enqueue_post(const BlockCtx& c, int T, int out_from, float* logits_dev) {
    const Spec& s = s_;
    const int n = s.d_model, hc = s.hc_count;
    if (!s.ple_layers.empty()) ple_upload(c, ple_host_, T, pemb_);
    for (int il = first_ple_layer(); il < s.n_layer; ++il) enqueue_layer(c, il, T);
    if (combine_pending_) {   // the last layer's
        hc_combine(c, x_, blk_, inject_, T);
        combine_pending_ = false;
    }
    if (out_from < T && logits_dev) {
        const int R = T - out_from;
        hc_mix(c, -1, 2, x_ + size_t(out_from) * hc * n, R, norm_, nullptr);
        head_logits(c, norm_, R, logits_dev);
    }
}

// In a prefill chunk the combine that ends a layer is deferred into the next layer's first mix
// (hc_combine_mix fuses it into the norm) unless a PLE block comes between.
void ForwardRef::enqueue_layer(const BlockCtx& c, int il, int T) {
    const Spec& s = s_;
    for (int pl : s.ple_layers)
        if (pl == il) {
            if (combine_pending_) {
                hc_combine(c, x_, blk_, inject_, T);
                combine_pending_ = false;
            }
            ple_block(c, il, ple_, pemb_, x_, T, ple_state_[il], in_window_ ? &ple_win_[il] : nullptr);
        }
    if (combine_pending_) {
        hc_combine_mix(c, il, 0, x_, blk_, inject_, T, mixed_, inject_);
        combine_pending_ = false;
    } else {
        hc_mix(c, il, 0, x_, T, mixed_, inject_);
    }
    if (s.mixer[il] == Mixer::QSA) qsa_mixer(c, il, mixed_, T, pos_, kv_[il], blk_);
    else gdn_mixer(c, il, mixed_, T, gdn_[il], blk_, nullptr, in_window_ ? &gdn_win_[il] : nullptr);
    hc_combine_mix(c, il, 1, x_, blk_, inject_, T, mixed_, inject_);
    if (in_chunk_) moe_block_stream(c, il, mixed_, T, *estream_, blk_, counts_dev_);
    else if ((T == 1 || in_window_) && fast_cache_) moe_block_fast(c, il, mixed_, *fast_cache_, *fast_host_, blk_, T);
    else moe_block(c, il, mixed_, T, moe_host_, blk_);
    if (in_chunk_) combine_pending_ = true;
    else hc_combine(c, x_, blk_, inject_, T);
}

void ForwardRef::enable_windows(int W) {
    if (W <= max_window_) return;
    if (W > max_batch_) throw std::runtime_error("enable_windows: window longer than the batch");
    for (GdnWindow& w : gdn_win_) free_gdn_window(w);
    for (PleWindow& w : ple_win_) free_ple_window(w);
    gdn_win_.assign(s_.n_layer, GdnWindow{});
    ple_win_.assign(s_.n_layer, PleWindow{});
    for (int il : s_.gdn_layers) gdn_win_[il] = alloc_gdn_window(s_, W);
    for (int il : s_.ple_layers) ple_win_[il] = alloc_ple_window(s_, ple_, W);
    max_window_ = W;
}

void ForwardRef::forward_window(const int32_t* seq, int T, float* logits_dev) {
    if (T < 1 || T > max_window_ || !fast_cache_ || !fast_host_->doorbell || T > fast_host_->max_window)
        throw std::runtime_error("forward_window: windows not enabled, or no fast MoE in doorbell mode");
    if (window_pos0_ >= 0) throw std::runtime_error("forward_window: the previous window was not committed");
    window_pos0_ = pos_;
    window_T_ = T;
    in_window_ = true;
    try {
        forward(seq, T, 0, logits_dev);
    } catch (...) {
        in_window_ = false;
        throw;
    }
    in_window_ = false;
}

void ForwardRef::commit(int n) {
    if (window_pos0_ < 0 || n < 0 || n > window_T_) throw std::runtime_error("commit: no window, or n out of range");
    if (n < window_T_) {
        const BlockCtx c{s_, w_, dec_.scratch, stream_};
        for (int il : s_.gdn_layers) gdn_rewind(c, gdn_[il], gdn_win_[il], window_T_, n);
        for (int il : s_.ple_layers) ple_rewind(c, il, ple_, ple_state_[il], ple_win_[il], window_T_, n);
        ck(cudaStreamSynchronize(stream_), "commit");
        pos_ = window_pos0_ + n;
    }
    fast_host_->access_prev_T = n;   // the cache learns from the kept tokens only
    window_pos0_ = -1;
}

std::vector<std::pair<void*, size_t>> ForwardRef::ckpt_parts() {
    const Spec& s = s_;
    const size_t gch = size_t(2 * s.ssm_groups * s.ssm_state + s.ssm_heads * s.ssm_state);
    std::vector<std::pair<void*, size_t>> parts;
    for (int il : s.gdn_layers) {
        parts.push_back({gdn_[il].S, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4});
        parts.push_back({gdn_[il].conv, size_t(s.ssm_conv - 1) * gch * 4});
    }
    for (int il : s.ple_layers) parts.push_back({ple_state_[il].hist, size_t(ple_.ngram) * 3 * size_t(s.hc_count) * s.d_model * 4 + 4});
    for (int il : s.qsa_layers) parts.push_back({kv_[il].idx_ring, size_t(qsa_ring_slots(s)) * s.idx_dim * 4});
    return parts;
}

void ForwardRef::save_checkpoint() {
    const auto parts = ckpt_parts();
    size_t total = 0;
    for (const auto& p : parts) total += (p.second + 255) & ~size_t(255);
    if (!ckpt_) {
        ck(cudaMalloc(&ckpt_, total), "cudaMalloc checkpoint");
        ckpt_bytes_ = total;
    }
    size_t off = 0;
    for (const auto& p : parts) {
        ck(cudaMemcpyAsync(static_cast<char*>(ckpt_) + off, p.first, p.second, cudaMemcpyDeviceToDevice, stream_), "checkpoint");
        off += (p.second + 255) & ~size_t(255);
    }
    ck(cudaStreamSynchronize(stream_), "checkpoint");
    ckpt_pos_ = pos_;
}

void ForwardRef::restore_checkpoint() {
    if (ckpt_pos_ < 0) throw std::runtime_error("restore_checkpoint: none saved");
    if (window_pos0_ >= 0) throw std::runtime_error("restore_checkpoint: a window is open");
    size_t off = 0;
    for (const auto& p : ckpt_parts()) {
        ck(cudaMemcpyAsync(p.first, static_cast<char*>(ckpt_) + off, p.second, cudaMemcpyDeviceToDevice, stream_), "restore");
        off += (p.second + 255) & ~size_t(255);
    }
    for (int il : s_.qsa_layers) reset_qsa_hot(s_, kv_[il], stream_);   // slots may hold rewritten positions' old values
    ck(cudaStreamSynchronize(stream_), "restore");
    pos_ = ckpt_pos_;
    have_access_ = false;
}

int ForwardRef::first_ple_layer() const { return s_.ple_layers.empty() ? s_.n_layer : s_.ple_layers.front(); }

bool ForwardRef::graph_eligible(int T, int out_from, float* logits_dev) const {
    return use_graphs_ && (T == 1 || in_window_) && T <= kMaxGraphTokens && out_from == 0 && logits_dev && fast_cache_ &&
           fast_host_->doorbell && embed_graph_capable(w_) && s_.idx_dim == 128;
}

void ForwardRef::drop_graphs() {
    for (auto& row : graphs_)
        for (Graphs& gs : row) {
            for (cudaGraphExec_t* g : {&gs.pre, &gs.post})
                if (*g) {
                    cudaGraphExecDestroy(*g);
                    *g = nullptr;
                }
            gs = Graphs{};
        }
}

ForwardRef::Graphs& ForwardRef::capture_graphs(int T, float* logits_dev) {
    Graphs& gs = graphs_[T][in_window_ ? 1 : 0];
    for (cudaGraphExec_t* g : {&gs.pre, &gs.post})
        if (*g) {
            cudaGraphExecDestroy(*g);
            *g = nullptr;
        }
    // everything the graphs touch is allocated now, not while capturing; a scratch that grows
    // later drops every graph (an eager pass does that)
    int capacity = 0;
    for (int il : s_.qsa_layers) capacity = kv_[il].capacity;
    qsa_scratch_reserve(s_, dec_.scratch, T, capacity / s_.qsa_block);
    if (!s_.ple_layers.empty() && (!ple_host_.raw_pinned || !ple_host_.raw_dev)) throw std::runtime_error("capture_graphs: run a prefill first");
    const size_t ple_bytes = size_t(T) * ple_.n_heads * ple_.row_bytes;
    if (!s_.ple_layers.empty() && (ple_host_.raw_pinned_bytes < ple_bytes || ple_host_.raw_dev_bytes < ple_bytes))
        throw std::runtime_error("capture_graphs: the PLE buffers are smaller than the step");
    const BlockCtx cg{s_, w_, dec_.scratch, stream_, params_dev_};
    auto capture = [&](auto&& body) {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        ck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
        body();
        ck(cudaStreamEndCapture(stream_, &g), "end capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate graph");
        cudaGraphDestroy(g);
        return ge;
    };
    gs.pre = capture([&] {
        ck(cudaMemcpyAsync(params_dev_, params_host_, size_t(2 + T) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "params");
        enqueue_pre(cg, nullptr, T);
    });
    gs.post = capture([&] { enqueue_post(cg, T, 0, logits_dev); });
    gs.logits = logits_dev;
    gs.ple_pinned = ple_host_.raw_pinned;
    gs.ple_dev = ple_host_.raw_dev;
    gs.scratch = dec_.scratch;
    ++graph_captures_;
    return gs;
}

void ForwardRef::alloc_bufs(Bufs& b, int T) {
    free_bufs(b);
    const size_t n = s_.d_model, hc = s_.hc_count, B = size_t(T);
    b.scratch = alloc_block_scratch(s_, T);
    b.emb = dalloc(B * n);
    b.x = dalloc(B * hc * n);
    b.mixed = dalloc(B * n);
    b.inject = dalloc(B * hc);
    b.blk = dalloc(B * n);
    b.pemb = dalloc(B * n);
    b.norm = dalloc(B * n);
    b.cap = T;
}

void ForwardRef::free_bufs(Bufs& b) {
    if (!b.cap) return;
    free_block_scratch(b.scratch);
    for (float* p : {b.emb, b.x, b.mixed, b.inject, b.blk, b.pemb, b.norm}) cudaFree(p);
    b = Bufs{};
}

void ForwardRef::use_bufs(Bufs& b) {
    emb_ = b.emb;
    x_ = b.x;
    mixed_ = b.mixed;
    inject_ = b.inject;
    blk_ = b.blk;
    pemb_ = b.pemb;
    norm_ = b.norm;
    scr_ = &b.scratch;
}

void ForwardRef::release_chunk_buffers() {
    if (ple_next_rows_.valid()) ple_next_rows_.wait();
    ple_next_pos_ = -1;
    look_seq_ = nullptr;
    look_n_ = 0;
    if (x_ == chunk_.x) use_bufs(dec_);
    cudaStreamSynchronize(stream_);
    for (int il : s_.qsa_layers) qsa_mirror_end(kv_[il]);
    free_bufs(chunk_);
    destroy_expert_stream(estream_);
    estream_ = nullptr;
    if (counts_dev_) cudaFree(counts_dev_);
    counts_dev_ = nullptr;
}

size_t ForwardRef::chunk_buffer_bytes() const {
    if (!chunk_.cap) return 0;
    const size_t n = s_.d_model, hc = s_.hc_count, B = size_t(chunk_.cap);
    return expert_stream_bytes(estream_) + chunk_.scratch.f32_elems * 4 + chunk_.scratch.q8_bytes + B * (6 * n + hc * n + hc) * 4;
}

size_t ForwardRef::chunk_bytes(int T, int end_pos) const {
    const Spec& s = s_;
    const size_t n = s.d_model, hc = s.hc_count, B = size_t(T);
    size_t b = block_scratch_bytes(s, T) + B * (6 * n + hc * n + hc) * 4;   // alloc_bufs
    b += expert_stream_bytes_for(s, *moe_host_.arena, T);
    b += gemm::workspace_bytes(int64_t(hc * n), T);                        // linear(): the widest input is the hc norm's
    b += size_t(s.n_layer) * s.n_expert * 4;                               // routing counts
    const int r = s.qsa_block, ldc = (s.idx_top_k + 2 * r - 2) / r * r;
    b += size_t(128) * (end_pos / r) * 8 + size_t(128) * ldc * 8;           // QSA sub-batches: scores, cell lists
    for (int il : s.qsa_layers)
        if (kv_[il].hot_blocks) {   // the VRAM mirror: K, V and their scales up to end_pos
            const size_t m = size_t(end_pos) * s.n_head_kv * s.head_dim_k;
            b += 2 * m + m / 8;
        }
    return b + b / 32 + (size_t(64) << 20);   // allocator rounding, the Q3_K weight copy
}

int ForwardRef::pick_chunk(int n, int end_pos, size_t free_bytes, int max_chunk) const {
    if (n <= max_batch_) return n;
    const size_t avail = free_bytes - std::min(free_bytes, size_t(256) << 20);
    int c = std::min(max_chunk, n);
    while (c > 1024 && chunk_bytes(c, end_pos) > avail) c = (c - 1) / 1024 * 1024;
    c = std::max(c, std::min(n, max_batch_ + 1));
    const int k = (n + c - 1) / c;
    return std::min(c, ((n + k - 1) / k + 255) / 256 * 256);
}

void ForwardRef::forward(const int32_t* seq, int T, int out_from, float* logits_dev) {
    if (T < 1) throw std::runtime_error("ForwardRef: bad batch size");
    const Spec& s = s_;
    in_chunk_ = T > max_batch_;
    if (in_chunk_) {
        if (in_window_) throw std::runtime_error("ForwardRef: a window cannot be a prefill chunk");
        if (chunk_.cap < T) {
            alloc_bufs(chunk_, T);
            destroy_expert_stream(estream_);
            estream_ = create_expert_stream(s, *moe_host_.arena, T);
        }
        if (!counts_dev_) {
            ck(cudaMalloc(&counts_dev_, counts_.size() * 4), "cudaMalloc routing counts");
            ck(cudaMemset(counts_dev_, 0, counts_.size() * 4), "memset routing counts");
        }
        use_bufs(chunk_);
        const int need = std::max(look_n_, pos_ + T);   // host KV: chunks attend from a VRAM mirror up to the prompt's end
        for (int il : s.qsa_layers) {
            if (kv_[il].mK && kv_[il].mcap < pos_ + T) qsa_mirror_end(kv_[il]);
            qsa_mirror_begin(s, kv_[il], pos_, need, stream_);
        }
        expert_stream_prefetch(estream_, 0);   // layer 0's experts copy while the embedding and PLE run
    } else {
        use_bufs(dec_);
    }
    const BlockCtx c{s, w_, *scr_, stream_};
    const bool graph = graph_eligible(T, out_from, logits_dev);
    if (!graph) drop_graphs();   // an eager pass may regrow scratch the graphs point at
    for (int il : s.qsa_layers)
        if (pos_ + T > kv_[il].capacity) throw std::runtime_error("ForwardRef: KV cache full");
    // the PLE rows come from the SSD: read them on another thread while the embedding and the
    // layers before the first PLE layer run
    std::future<void> ple_rows;
    if (ple_next_rows_.valid()) {   // a lookahead read: use it if it is this chunk's, else let it finish
        ple_next_rows_.get();
        if (ple_next_pos_ == pos_ && ple_next_T_ == T) {
            std::swap(ple_host_.raw_pinned, ple_next_.raw_pinned);
            std::swap(ple_host_.raw_pinned_bytes, ple_next_.raw_pinned_bytes);
        } else {
            ple_next_pos_ = -1;
        }
    }
    const bool have_rows = !s.ple_layers.empty() && ple_next_pos_ == pos_ && ple_next_T_ == T;
    ple_next_pos_ = -1;
    if (!s.ple_layers.empty() && !have_rows) ple_rows = std::async(std::launch::async, [&] {
            unpin_current_thread();
            ple_fetch(ple_host_, seq, pos_, T);
        });
    const bool fast = (T == 1 || in_window_) && fast_cache_;
    const bool db = fast && fast_host_->doorbell;
    if (db) doorbell_begin_token(*fast_host_, T);
    if (graph) {
        params_host_[0] = seq[pos_];
        params_host_[1] = pos_;
        params_host_[2] = int32_t(fast_host_->seq);
        for (int t = 1; t < T; ++t) params_host_[2 + t] = seq[pos_ + t];
        Graphs* gs = &graphs_[T][in_window_ ? 1 : 0];
        if (gs->pre && !same_buffers(gs->scratch, dec_.scratch)) {   // another length's capture grew the scratch
            drop_graphs();
            gs = &graphs_[T][in_window_ ? 1 : 0];
        }
        if (!gs->pre || gs->logits != logits_dev) {
            if (ple_rows.valid()) ple_rows.wait();   // the PLE buffers must exist before capture
            if (!s.ple_layers.empty() && ple_host_.raw_dev_bytes < ple_host_.raw_pinned_bytes) {   // grow the device side too
                if (ple_host_.raw_dev) cudaFree(ple_host_.raw_dev);
                ck(cudaMalloc(&ple_host_.raw_dev, ple_host_.raw_pinned_bytes), "cudaMalloc ple rows");
                ple_host_.raw_dev_bytes = ple_host_.raw_pinned_bytes;
                for (auto& row : graphs_)   // the old graphs point at the old buffer
                    for (Graphs& g : row)
                        if (&g != gs && g.pre) {
                            drop_graphs();
                            break;
                        }
            }
            gs = &capture_graphs(T, logits_dev);
            // capturing may have grown the scratch under the other lengths' graphs
            for (auto& row : graphs_)
                for (Graphs& g : row)
                    if (&g != gs && g.pre && !same_buffers(g.scratch, dec_.scratch))
                        for (cudaGraphExec_t* e : {&g.pre, &g.post})
                            if (*e) {
                                cudaGraphExecDestroy(*e);
                                *e = nullptr;
                            }
        }
        ck(cudaGraphLaunch(gs->pre, stream_), "launch graph (pre)");
        if (ple_rows.valid()) ple_rows.get();
        if (ple_host_.raw_pinned != gs->ple_pinned || ple_host_.raw_dev != gs->ple_dev)
            throw std::runtime_error("ForwardRef: PLE buffers moved under a captured graph");
        ck(cudaGraphLaunch(gs->post, stream_), "launch graph (post)");
    } else {
        enqueue_pre(c, seq, T);
        if (ple_rows.valid()) ple_rows.get();
        enqueue_post(c, T, out_from, logits_dev);
        // the next chunk's n-gram rows, while this one computes
        const int next = pos_ + T, nT = std::min(T, look_n_ - next);
        if (in_chunk_ && look_seq_ && !s.ple_layers.empty() && nT > max_batch_) {
            ple_next_pos_ = next;
            ple_next_T_ = nT;
            const int32_t* ls = look_seq_;
            ple_next_rows_ = std::async(std::launch::async, [this, ls, next, nT] {
                unpin_current_thread();
                ple_fetch(ple_next_, ls, next, nT);
            });
        }
    }
    // the adaptive cache learns from the previous step while this one runs on the GPU
    if (fast && cache_mgr_ && have_access_) cache_manager_step(cache_mgr_, *fast_host_, stream_);
    ck(cudaStreamSynchronize(stream_), "forward");
    if (db) doorbell_end_token(*fast_host_, s_);
    have_access_ = fast;
    if (have_access_) {
        fast_host_->access_prev = fast_host_->access;
        fast_host_->access_prev_T = T;
    }
    if (in_chunk_) {   // the chunk's routing counts join the host's
        std::vector<uint32_t> add(counts_.size());
        ck(cudaMemcpy(add.data(), counts_dev_, add.size() * 4, cudaMemcpyDeviceToHost), "routing counts");
        ck(cudaMemset(counts_dev_, 0, add.size() * 4), "memset routing counts");
        for (size_t i = 0; i < add.size(); ++i) counts_[i] += add[i];
        in_chunk_ = false;
    }
    if (!fast && count_half_life_ > 0 && (count_tokens_ += T) >= count_half_life_) {
        for (uint32_t& c : counts_) c >>= 1;
        count_tokens_ -= count_half_life_;
    }
    pos_ += T;
}

}  // namespace flashrt::qwen4exp
