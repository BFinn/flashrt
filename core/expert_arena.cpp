// SPDX-License-Identifier: Apache-2.0
#include "core/expert_arena.hpp"

namespace flashrt {

ExpertArena arena_alloc(int n_layer, int n_expert, size_t blob_bytes, PageMode mode, int touch_threads) {
    ExpertArena a;
    a.blob_bytes = blob_bytes;
    a.stride = (blob_bytes + 4095) & ~size_t(4095);
    a.n_layer = n_layer;
    a.n_expert = n_expert;
    a.buf = host_alloc(a.total_bytes(), mode, touch_threads);
    return a;
}

void arena_free(ExpertArena& a) {
    host_free(a.buf);
    a = ExpertArena{};
}

}  // namespace flashrt
