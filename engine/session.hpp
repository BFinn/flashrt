// SPDX-License-Identifier: Apache-2.0
// One loaded model serving one sequence at a time: the engine side of the protocol in
// docs/design.md. It owns the weights, the host expert arena and the VRAM expert cache, the
// CPU pool with its doorbell miss server, the forward pass, and the optional MTP draft head.
//
// generate() reuses the longest usable prefix of the previous sequence: the whole of it when the
// new prompt extends it, else the latest checkpoint inside the shared prefix (the recurrent state
// cannot be rewound to an arbitrary position): the one at the end of the previous prompt, on the
// GPU, or one of those taken during earlier prefills, in host RAM. Else nothing. The prompt's new
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
    int swap_budget = 64;             // expert-cache uploads per step, as CachePolicyConfig (sw89: 8 -> 32, sw104: 32 -> 64)
    float cache_admit = 1.0f, cache_margin = 1.2f;   // the adaptive cache's admission, as CachePolicyConfig (sw100)
    float cache_seed_scale = 0.03f;                  // its warm-up, as CachePolicyConfig (sw104)
    // the fill after a chunked prompt: the routing of its last cache_tail_tokens tokens added at
    // cache_tail_weight times the whole prompt's (0: off; sw111)
    int cache_tail_tokens = 16;
    float cache_tail_weight = 0.0f;
    int prefill_batch = 64;
    int prefill_chunk = 0;            // prompts adding at least chunk_min tokens prefill in chunks of this
    int chunk_min = 256;              // (the experts stream to the GPU; the expert cache is rebuilt after);
    int prefill_chunk_max = 16384;    // 0: the longest that fits the free VRAM, up to prefill_chunk_max
    std::string cache_prior;          // routing counts of a calibration prefill (fr_bench --save-counts)
    // Prefix reuse beyond the end of the previous prompt: up to `ckpts` recurrent-state
    // checkpoints in pinned host RAM (113 MiB each for qwen4exp with the MTP head), taken during
    // a prefill at chunk ends at least ckpt_interval tokens apart. And once prompts have shown a
    // fixed tail after growing text (one diverged from the previous prompt within its last
    // ckpt_tail tokens), also before each prompt's last tokens, as many as that tail rounded up to
    // 8: the next prompt then reuses up to there. 0 turns the tail checkpoint off.
    int ckpts = 8;
    int ckpt_interval = 4096;
    int ckpt_tail = 64;
    bool cache_check = false;         // log the expert cache's consistency after each request (diagnostics)
};

struct GenerateRequest {
    std::vector<int32_t> prompt;
    int max_new = 256;
    sample::Params sampling;
    uint64_t seed = 0;
    std::vector<int32_t> stop_ids;
    // tests only (engine_smoke.py --faults, FLASHRT_TEST_HOOKS=1):
    int fail_at = 0;                  // throw after the first prefill step (1) or in the first decode step (2)
    bool first_top = false;           // report the first generated position's top logits
};

struct GenerateResult {
    int generated = 0, prompt_tokens = 0, reused = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish;               // "stop", "length", "cancelled"
    long drafts_proposed = 0, drafts_accepted = 0;
    long cache_hits = 0, cache_misses = 0;   // routed experts in the decode: in the VRAM cache, or not
    std::vector<std::pair<int32_t, float>> first_top;   // with GenerateRequest::first_top: (id, logit), best first
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
    // False after a failure the process cannot recover from: a doorbell timeout (the miss server
    // is out of step with the GPU) or a sticky CUDA error. The engine should exit.
    bool healthy() const;

    // on_token gets every generated token (the stop token is not reported); on_progress gets
    // (prompt tokens done, total) during the prefill. `cancel` is polled between steps.
    // An invalid request throws before anything changes, so the next request still reuses the
    // previous sequence. A failure during the request resets the session to an empty sequence
    // (the next request starts cold) and rethrows; if even that is impossible, healthy() turns
    // false.
    GenerateResult generate(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                            const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel);

private:
    GenerateResult run(const GenerateRequest& r, const std::function<void(int32_t)>& on_token,
                       const std::function<void(int, int)>& on_progress, const std::atomic<bool>& cancel);
    struct Impl;
    std::unique_ptr<Impl> m_;
};

}  // namespace flashrt
