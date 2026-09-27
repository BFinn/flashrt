// SPDX-License-Identifier: Apache-2.0
// qwen4exp (Qwen3.8-Flash-Next): the model description parsed from a GGUF, and the weight
// plan that places every tensor in a tier.
//
// Layer structure (from the GGUF metadata and tensor directory):
//   48 layers; layer i is QSA sparse attention when (i + 1) % full_attention_interval == 0
//   (i = 3, 7, ..., 47), otherwise a GDN linear-attention (recurrent) mixer. Every layer has
//   hyper-connections (hc_count residual streams, low-rank mixing), a 512-expert top-10 MoE
//   with one shared expert, and layers in ple.layers also read the n-gram (PLE) table.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
struct Gguf;
struct GgufTensor;
}

namespace flashrt::qwen4exp {

enum class Mixer { GDN, QSA };

struct Spec {
    std::string arch;                 // "qwen4exp"
    int n_layer = 0, d_model = 0, n_vocab = 0, max_context = 0;
    std::vector<Mixer> mixer;         // per layer
    std::vector<int> qsa_layers, gdn_layers;

    // attention (QSA layers)
    int n_head = 0, n_head_kv = 0, head_dim_k = 0, head_dim_v = 0;
    int qsa_block = 0;                // compress ratio: cells per indexer block
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    int rope_dims = 0;
    double rope_base = 0;

    // GDN layers
    int ssm_conv = 0, ssm_groups = 0, ssm_inner = 0, ssm_state = 0, ssm_heads = 0;

    // MoE
    int n_expert = 0, top_k = 0, d_ff_expert = 0, d_ff_shared = 0;

    // hyper-connections and PLE
    int hc_count = 0, hc_rank = 0;
    std::vector<int> ple_layers;
    int ple_ngram = 0, ple_heads = 0, ple_dim = 0;
    int64_t ple_rows = 0;

    double rms_eps = 0;
};

// Throws std::runtime_error when the GGUF is not qwen4exp or a required key is missing.
Spec parse(const Gguf& g);

enum class Tier {
    VramDense,    // read by the GPU every window: mixers, routers, shared experts, hc, head
    HostExperts,  // routed experts: pinned host arena, cached in VRAM slots
    SsdPle,       // the n-gram table, read by row
};
const char* tier_name(Tier t);

struct Placement {
    const GgufTensor* tensor;
    Tier tier;
    int layer;    // -1 for global tensors
};

struct WeightPlan {
    std::vector<Placement> tensors;
    uint64_t bytes[3] = {0, 0, 0};    // per Tier
    uint64_t expert_bytes = 0;        // one repacked expert (gate + up + down)
    int n_experts_total = 0;          // n_layer * n_expert
};

// Places every tensor; throws if an expected tensor is missing or has an unexpected shape.
WeightPlan plan(const Gguf& g, const Spec& s);

// Bytes the QSA KV cache needs per cell (one token) across all QSA layers, at 8-bit with one
// fp16 scale per 32 values; and the GDN recurrent state for one sequence.
uint64_t qsa_kv_bytes_per_cell(const Spec& s);
uint64_t gdn_state_bytes(const Spec& s);

}  // namespace flashrt::qwen4exp
