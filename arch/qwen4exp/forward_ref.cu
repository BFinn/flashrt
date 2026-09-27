// SPDX-License-Identifier: Apache-2.0
#include "arch/qwen4exp/forward_ref.hpp"

#include "core/gguf.hpp"
#include "core/platform.hpp"

#include <algorithm>
#include <cstdio>
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
                       int max_ctx, int max_batch, bool kv_q8)
    : s_(s), w_(w), max_batch_(max_batch) {
    ple_ = parse_ple(g);
    reader_ = std::make_unique<RowReader>(g.shards[ple_.table_shard], ple_.table_offset, ple_.row_bytes, 16);
    ple_host_.ple = &ple_;
    ple_host_.reader = reader_.get();
    moe_host_.arena = &arena;
    moe_host_.pool = &pool;
    counts_.assign(size_t(s.n_layer) * s.n_expert, 0);
    moe_host_.counts = &counts_;
    scratch_ = alloc_block_scratch(s, max_batch);
    ck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    gdn_.resize(s.n_layer);
    kv_.resize(s.n_layer);
    ple_state_.resize(s.n_layer);
    for (int il : s.gdn_layers) gdn_[il] = alloc_gdn_state(s);
    for (int il : s.qsa_layers) kv_[il] = alloc_qsa_cache(s, max_ctx, kv_q8);
    for (int il : s.ple_layers) ple_state_[il] = alloc_ple_state(s, ple_);
    const size_t n = s.d_model, hc = s.hc_count, B = max_batch;
    emb_ = dalloc(B * n);
    x_ = dalloc(B * hc * n);
    mixed_ = dalloc(B * n);
    inject_ = dalloc(B * hc);
    blk_ = dalloc(B * n);
    pemb_ = dalloc(B * n);
    norm_ = dalloc(B * n);
    ck(cudaMalloc(&argmax_dev_, 4), "cudaMalloc argmax");
    ck(cudaMalloc(&params_dev_, 4 * sizeof(int32_t)), "cudaMalloc decode params");
    ck(cudaHostAlloc(&params_host_, 4 * sizeof(int32_t), cudaHostAllocDefault), "cudaHostAlloc decode params");
    ck(cudaHostAlloc(&argmax_host_, 4, cudaHostAllocDefault), "cudaHostAlloc argmax");
}

ForwardRef::~ForwardRef() {
    for (int il : s_.gdn_layers) free_gdn_state(gdn_[il]);
    for (int il : s_.qsa_layers) free_qsa_cache(kv_[il]);
    for (int il : s_.ple_layers) free_ple_state(ple_state_[il]);
    free_block_scratch(scratch_);
    for (float* p : {emb_, x_, mixed_, inject_, blk_, pemb_, norm_}) cudaFree(p);
    if (ple_host_.raw_dev) cudaFree(ple_host_.raw_dev);
    if (ple_host_.raw_pinned) cudaFreeHost(ple_host_.raw_pinned);
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
        ck(cudaMemcpy(bounce.data(), dev, bytes, cudaMemcpyDeviceToHost), "state to host");
        if (std::fwrite(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file write failed");
    } else {
        if (std::fread(bounce.data(), 1, bytes, f) != bytes) throw std::runtime_error("state file truncated");
        ck(cudaMemcpy(dev, bounce.data(), bytes, cudaMemcpyHostToDevice), "state to device");
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
    const size_t kvn = size_t(s.n_head_kv) * s.head_dim_k;   // K or V values of one cell
    const int gch = 2 * s.ssm_groups * s.ssm_state + s.ssm_heads * s.ssm_state;
    for (int il : s.gdn_layers) {
        state_io(f, gdn_[il].S, size_t(s.ssm_heads) * s.ssm_state * s.ssm_state * 4, save, bounce);
        state_io(f, gdn_[il].conv, size_t(s.ssm_conv - 1) * gch * 4, save, bounce);
    }
    for (int il : s.qsa_layers) {
        QsaCache& kv = kv_[il];
        if (!q8) {
            state_io(f, kv.K, size_t(pos_) * kvn * 2, save, bounce);
            state_io(f, kv.V, size_t(pos_) * kvn * 2, save, bounce);
        } else if (file_q8) {
            state_io(f, kv.K, size_t(pos_) * kvn, save, bounce);
            state_io(f, kv.Ks, size_t(pos_) * kvn / 32 * 2, save, bounce);
            state_io(f, kv.V, size_t(pos_) * kvn, save, bounce);
            state_io(f, kv.Vs, size_t(pos_) * kvn / 32 * 2, save, bounce);
        } else {   // fp16 file into a q8 cache: convert in chunks on the GPU
            const long rows = long(pos_) * s.n_head_kv, chunk = 1L << 16;
            void* tmp = nullptr;
            ck(cudaMalloc(&tmp, size_t(chunk) * s.head_dim_k * 2), "cudaMalloc state conversion");
            for (int which = 0; which < 2; ++which) {
                int8_t* dst = static_cast<int8_t*>(which ? kv.V : kv.K);
                uint16_t* dsc = which ? kv.Vs : kv.Ks;
                for (long r0 = 0; r0 < rows; r0 += chunk) {
                    const long nr = std::min(chunk, rows - r0);
                    state_io(f, tmp, size_t(nr) * s.head_dim_k * 2, false, bounce);
                    qsa_h2q8_rows(tmp, dst + size_t(r0) * s.head_dim_k, dsc + size_t(r0) * (s.head_dim_k / 32), nr, s.head_dim_k, stream_);
                    ck(cudaStreamSynchronize(stream_), "state conversion");
                }
            }
            cudaFree(tmp);
        }
        state_io(f, kv_[il].idx_pooled, size_t(pos_ / s.qsa_block + 1) * s.idx_dim * 4, save, bounce);
        state_io(f, kv_[il].idx_ring, size_t(s.qsa_block) * s.idx_dim * 4, save, bounce);
    }
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
    for (int il : s_.qsa_layers)   // K/V and pooled keys are overwritten as positions advance; the ring is not
        ck(cudaMemsetAsync(kv_[il].idx_ring, 0, size_t(s_.qsa_block) * s_.idx_dim * 4, stream_), "memset ring");
    ck(cudaStreamSynchronize(stream_), "reset");
    pos_ = 0;
}

// embedding, hyper-connection streams, and the layers before the first PLE layer
void ForwardRef::enqueue_pre(const BlockCtx& c, const int32_t* seq, int T) {
    const Spec& s = s_;
    const int n = s.d_model, hc = s.hc_count;
    embed(c, seq ? seq + pos_ : nullptr, T, emb_);
    for (int t = 0; t < T; ++t)
        for (int st = 0; st < hc; ++st)
            ck(cudaMemcpyAsync(x_ + (size_t(t) * hc + st) * n, emb_ + size_t(t) * n, size_t(n) * 4, cudaMemcpyDeviceToDevice, stream_),
               "hc init");
    for (int il = 0; il < first_ple_layer(); ++il) enqueue_layer(c, il, T);
}

// the PLE rows (already in ple_host_.raw_pinned), the remaining layers, and the head
void ForwardRef::enqueue_post(const BlockCtx& c, int T, int out_from, float* logits_dev) {
    const Spec& s = s_;
    const int n = s.d_model, hc = s.hc_count;
    if (!s.ple_layers.empty()) ple_upload(c, ple_host_, T, pemb_);
    for (int il = first_ple_layer(); il < s.n_layer; ++il) enqueue_layer(c, il, T);
    if (out_from < T && logits_dev) {
        const int R = T - out_from;
        hc_mix(c, -1, 2, x_ + size_t(out_from) * hc * n, R, norm_, nullptr);
        head_logits(c, norm_, R, logits_dev);
    }
}

void ForwardRef::enqueue_layer(const BlockCtx& c, int il, int T) {
    const Spec& s = s_;
    for (int pl : s.ple_layers)
        if (pl == il) ple_block(c, il, ple_, pemb_, x_, T, ple_state_[il]);
    hc_mix(c, il, 0, x_, T, mixed_, inject_);
    if (s.mixer[il] == Mixer::QSA) qsa_mixer(c, il, mixed_, T, pos_, kv_[il], blk_);
    else gdn_mixer(c, il, mixed_, T, gdn_[il], blk_);
    hc_combine(c, x_, blk_, inject_, T);
    hc_mix(c, il, 1, x_, T, mixed_, inject_);
    if (T == 1 && fast_cache_) moe_block_fast(c, il, mixed_, *fast_cache_, *fast_host_, blk_);
    else moe_block(c, il, mixed_, T, moe_host_, blk_);
    hc_combine(c, x_, blk_, inject_, T);
}

int ForwardRef::first_ple_layer() const { return s_.ple_layers.empty() ? s_.n_layer : s_.ple_layers.front(); }

bool ForwardRef::graph_eligible(int T, int out_from, float* logits_dev) const {
    return use_graphs_ && T == 1 && out_from == 0 && logits_dev && fast_cache_ && fast_host_->doorbell && embed_graph_capable(w_) &&
           s_.idx_dim == 128;
}

void ForwardRef::drop_graphs() {
    for (cudaGraphExec_t* g : {&graph_pre_, &graph_post_})
        if (*g) {
            cudaGraphExecDestroy(*g);
            *g = nullptr;
        }
}

void ForwardRef::capture_graphs(float* logits_dev) {
    drop_graphs();
    // everything the graphs touch is allocated now, not while capturing
    int capacity = 0;
    for (int il : s_.qsa_layers) capacity = kv_[il].capacity;
    qsa_scratch_reserve(s_, scratch_, 1, capacity / s_.qsa_block);
    if (!s_.ple_layers.empty() && (!ple_host_.raw_pinned || !ple_host_.raw_dev)) throw std::runtime_error("capture_graphs: run a prefill first");
    const BlockCtx cg{s_, w_, scratch_, stream_, params_dev_};
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
    graph_pre_ = capture([&] {
        ck(cudaMemcpyAsync(params_dev_, params_host_, 3 * sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "params");
        enqueue_pre(cg, nullptr, 1);
    });
    graph_post_ = capture([&] { enqueue_post(cg, 1, 0, logits_dev); });
    graph_logits_ = logits_dev;
    graph_ple_pinned_ = ple_host_.raw_pinned;
    graph_ple_dev_ = ple_host_.raw_dev;
    ++graph_captures_;
}

void ForwardRef::forward(const int32_t* seq, int T, int out_from, float* logits_dev) {
    if (T < 1 || T > max_batch_) throw std::runtime_error("ForwardRef: bad batch size");
    const Spec& s = s_;
    const BlockCtx c{s, w_, scratch_, stream_};
    const bool graph = graph_eligible(T, out_from, logits_dev);
    if (!graph) drop_graphs();   // an eager pass may regrow scratch the graphs point at
    for (int il : s.qsa_layers)
        if (pos_ + T > kv_[il].capacity) throw std::runtime_error("ForwardRef: KV cache full");
    // the PLE rows come from the SSD: read them on another thread while the embedding and the
    // layers before the first PLE layer run
    std::future<void> ple_rows;
    if (!s.ple_layers.empty()) ple_rows = std::async(std::launch::async, [&] {
            unpin_current_thread();
            ple_fetch(ple_host_, seq, pos_, T);
        });
    const bool db = T == 1 && fast_cache_ && fast_host_->doorbell;
    if (db) doorbell_begin_token(*fast_host_);
    if (graph) {
        params_host_[0] = seq[pos_];
        params_host_[1] = pos_;
        params_host_[2] = int32_t(fast_host_->seq);
        if (!graph_pre_ || graph_logits_ != logits_dev) {
            if (ple_rows.valid()) ple_rows.wait();   // the PLE buffers must exist before capture
            capture_graphs(logits_dev);
        }
        ck(cudaGraphLaunch(graph_pre_, stream_), "launch graph (pre)");
        if (ple_rows.valid()) ple_rows.get();
        if (ple_host_.raw_pinned != graph_ple_pinned_ || ple_host_.raw_dev != graph_ple_dev_)
            throw std::runtime_error("ForwardRef: PLE buffers moved under a captured graph");
        ck(cudaGraphLaunch(graph_post_, stream_), "launch graph (post)");
    } else {
        enqueue_pre(c, seq, T);
        if (ple_rows.valid()) ple_rows.get();
        enqueue_post(c, T, out_from, logits_dev);
    }
    // the adaptive cache learns from the previous token while this one runs on the GPU
    if (T == 1 && fast_cache_ && cache_mgr_ && have_access_) cache_manager_step(cache_mgr_, *fast_host_, stream_);
    ck(cudaStreamSynchronize(stream_), "forward");
    if (db) doorbell_end_token(*fast_host_, s_);
    have_access_ = T == 1 && fast_cache_;
    if (have_access_) fast_host_->access_prev = fast_host_->access;
    pos_ += T;
}

}  // namespace flashrt::qwen4exp
