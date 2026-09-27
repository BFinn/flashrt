// SPDX-License-Identifier: Apache-2.0
// Stubs for ggml core functions the vendored kernels reference. The type-size helpers are
// real (the raw-pointer dispatch calls them) for every type flashrt uses; the rest belong to
// the tensor-based entry points (ggml_cuda_mul_mat_vec_q and friends), which flashrt never
// calls, and abort. Link this into
// flashrt binaries that do not link libggml. Do NOT link it next to libggml: a definition in
// the executable would interpose on libggml's own internal calls.
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace {
struct TypeInfo {
    int64_t blck;
    size_t size;
    bool quantized;
};
TypeInfo info(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return {1, 4, false};
        case GGML_TYPE_F16: return {1, 2, false};
        case GGML_TYPE_BF16: return {1, 2, false};
        case GGML_TYPE_Q4_0: return {32, 18, true};
        case GGML_TYPE_Q4_1: return {32, 20, true};
        case GGML_TYPE_Q5_0: return {32, 22, true};
        case GGML_TYPE_Q5_1: return {32, 24, true};
        case GGML_TYPE_Q8_0: return {32, 34, true};
        case GGML_TYPE_Q8_1: return {32, 36, true};
        case GGML_TYPE_Q2_0: return {64, 18, true};
        case GGML_TYPE_IQ4_NL: return {32, 18, true};
        case GGML_TYPE_Q2_K: return {256, 84, true};
        case GGML_TYPE_Q3_K: return {256, 110, true};
        case GGML_TYPE_Q4_K: return {256, 144, true};
        case GGML_TYPE_Q5_K: return {256, 176, true};
        case GGML_TYPE_Q6_K: return {256, 210, true};
        case GGML_TYPE_IQ4_XS: return {256, 136, true};
        default:
            std::fprintf(stderr, "flashrt: ggml type %d is not supported by the stubs\n", int(t));
            std::abort();
    }
}

[[noreturn]] void unused(const char* fn) {
    std::fprintf(stderr, "flashrt: ggml function %s is not available (tensor-based path not used)\n", fn);
    std::abort();
}
}  // namespace

extern "C" {
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::abort();
}
bool ggml_are_same_stride(const struct ggml_tensor*, const struct ggml_tensor*) { unused("ggml_are_same_stride"); }
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor*) { unused("ggml_backend_buffer_get_alloc_size"); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { unused("ggml_backend_buffer_get_usage"); }
int64_t ggml_blck_size(enum ggml_type t) { return info(t).blck; }
bool ggml_is_contiguous(const struct ggml_tensor*) { unused("ggml_is_contiguous"); }
bool ggml_is_contiguously_allocated(const struct ggml_tensor*) { unused("ggml_is_contiguously_allocated"); }
bool ggml_is_quantized(enum ggml_type t) { return info(t).quantized; }
size_t ggml_nbytes(const struct ggml_tensor*) { unused("ggml_nbytes"); }
int64_t ggml_nelements(const struct ggml_tensor*) { unused("ggml_nelements"); }
const char* ggml_type_name(enum ggml_type) { unused("ggml_type_name"); }
size_t ggml_type_size(enum ggml_type t) { return info(t).size; }
}
