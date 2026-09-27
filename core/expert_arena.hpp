// SPDX-License-Identifier: Apache-2.0
// The host expert arena: one large allocation of repacked expert blobs, addressed by
// (layer, expert). Blobs start on 4 KiB boundaries so each is page-aligned for DMA.
// The quant pack defines the blob contents; the arena only sizes and addresses them.
#pragma once

#include "core/platform.hpp"

#include <cstddef>
#include <cstdint>

namespace flashrt {

struct ExpertArena {
    HostBuffer buf;
    size_t blob_bytes = 0;   // useful bytes per expert
    size_t stride = 0;       // blob_bytes rounded up to 4 KiB
    int n_layer = 0, n_expert = 0;

    uint8_t* blob(int layer, int expert) const {
        return static_cast<uint8_t*>(buf.ptr) + (size_t(layer) * n_expert + expert) * stride;
    }
    size_t total_bytes() const { return size_t(n_layer) * n_expert * stride; }
};

// Pages are touched by `touch_threads` threads so the allocation is resident before use;
// pass 0 when a loader writes every blob anyway.
// Returns an arena with buf.ptr == nullptr on failure.
ExpertArena arena_alloc(int n_layer, int n_expert, size_t blob_bytes, PageMode mode, int touch_threads);
void arena_free(ExpertArena& a);

}  // namespace flashrt
