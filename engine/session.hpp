// SPDX-License-Identifier: Apache-2.0
// One loaded model serving one sequence at a time: the engine side of the protocol in
// docs/design.md. It owns the weights, the host expert arena and the VRAM expert cache, the
// CPU pool with its doorbell miss server, the forward pass, and the optional MTP draft head.
//
// generate() reuses the longest usable prefix of the previous sequence: the whole of it when the
// new prompt extends it, else the checkpoint taken at the end of the previous prompt (the
// recurrent state cannot be rewound to an arbitrary position), else nothing. The prompt's new
// tokens are prefilled in batches (reference path), then decoding runs on the fast path, plain
// or speculative (the MTP head drafts K tokens, the target verifies them in one window, exact
// speculative sampling).
#pragma once

#include "kernels/cuda/sample.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {

struct SessionOptions {
    std::string model;                // first shard of the target GGUF
    std::string mtp;                  // the MTP draft GGUF ("" = no speculation)
    std::string draft_vocab;          // token ranking for the head's trimmed LM head ("" = full)
    int draft_vocab_n = 32768;
    int mtp_bits = 2;                 // the head's experts: 8, 4 or 2 (2 frees VRAM for cache slots; same acceptance, sw65)
    int spec_k = 2;                   // drafts per round (0 = plain decoding)
    int max_ctx = 262144;
    bool kv_q8 = true;
    int kv_hot = 4096;                // q8 host KV with this many GPU blocks per layer (0 = all in VRAM)
    int workers = 8;
    int reserve_mib = 512;            // VRAM left free after the expert cache: requests vary more than fr_bench's runs (256 there; sw64, sw86)
    int swap_budget = 8;
    int prefill_batch = 64;
    int prefill_chunk = 0;            // prompts adding at least chunk_min tokens prefill in chunks of this
    int chunk_min = 256;              // (the experts stream to the GPU; the expert cache is rebuilt after);
    int prefill_chunk_max = 16384;    // 0: the longest that fits the free VRAM, up to prefill_chunk_max
    std::string cache_prior;          // routing counts of a calibration prefill (fr_bench --save-counts)
};

struct GenerateRequest {
    std::vector<int32_t> prompt;
    int max_new = 256;
    sample::Params sampling;
    uint64_t seed = 0;
    std::vector<int32_t> stop_ids;
};

struct GenerateResult {
    int generated = 0, prompt_tokens = 0, reused = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish;               // "stop", "length", "cancelled"
    long drafts_proposed = 0, drafts_accepted = 0;
};

class Session {
public:
    explicit Session(const SessionOptions& o);   // loads everything; throws on failure
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    int max_context() const;
    int n_vocab() const;
    std::string arch() const;
    bool speculative() const;

    // on_token gets every generated token (the stop token is not reported); on_progress gets
    // (prompt tokens done, total) during the prefill. `cancel` is polled between steps.
    GenerateResult generate(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                            const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel);

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

}  // namespace flashrt
