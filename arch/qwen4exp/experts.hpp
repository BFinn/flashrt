// SPDX-License-Identifier: Apache-2.0
// Loads qwen4exp's routed experts from the GGUF into a host arena of Q2_0 blobs.
#pragma once

#include "arch/qwen4exp/spec.hpp"
#include "core/expert_arena.hpp"

#include <cstdint>

namespace flashrt {
struct Gguf;
}

namespace flashrt::qwen4exp {

struct LoadStats {
    double seconds = 0;
    uint64_t bytes_read = 0;
};

// Reads the three expert slabs of every layer with O_DIRECT (no page cache) and repacks
// each expert into arena.blob(layer, e). The arena must be sized for n_layer x n_expert
// blobs of q2_0::expert_bytes(). Work is split across `threads` threads by (layer, range
// of experts). Throws std::runtime_error on I/O errors.
LoadStats load_experts(const Gguf& g, const Spec& s, ExpertArena& arena, int threads);

}  // namespace flashrt::qwen4exp
