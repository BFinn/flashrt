// SPDX-License-Identifier: Apache-2.0
// The numbers flashrt's files and tensors are identified by, in one place: the ggml tensor type
// ids it handles (ggml.h's enum ggml_type, MIT), flashrt's own converted types, and the magics of
// flashrt's own files.
#pragma once

#include <cstdint>

namespace flashrt {

namespace ggml_type {
constexpr uint32_t kQ4_0 = 2;
constexpr uint32_t kQ8_0 = 8;
constexpr uint32_t kQ3_K = 11;
constexpr uint32_t kIQ4_NL = 20;
constexpr uint32_t kBF16 = 30;
constexpr uint32_t kQ2_0 = 42;
}  // namespace ggml_type

// GpuTensor types outside ggml's ids: Q3_K converted to Q3R (kernels/cuda/q3r.h); BF16 converted
// to Q8P: int8 [rows][cols], then fp16 scales [rows][cols / 32] (Q8_0's values, planar)
constexpr uint32_t kTypeQ3R = 1000;
constexpr uint32_t kTypeQ8P = 1001;

// File magics (the first int64 of the file), ASCII little-endian
constexpr int64_t kStateMagicV1 = 0x46525354;   // "FRST": a Forward state file (fp16 KV)
constexpr int64_t kStateMagicV2 = 0x46525332;   // "FRS2": the same with the KV format
constexpr int64_t kCachePriorMagic = 0x50435246;   // "FRCP": routing counts (fr_bench --save-counts)

}  // namespace flashrt
