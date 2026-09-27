// SPDX-License-Identifier: Apache-2.0
// Stubs for ggml core functions referenced only by the vendored kernels' tensor-based entry
// points (ggml_cuda_mul_mat_vec_q and friends), which flashrt never calls. Link this into
// flashrt binaries that do not link libggml. Do NOT link it next to libggml: a definition in
// the executable would interpose on libggml's own internal calls.
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace {
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
int64_t ggml_blck_size(enum ggml_type) { unused("ggml_blck_size"); }
bool ggml_is_contiguous(const struct ggml_tensor*) { unused("ggml_is_contiguous"); }
bool ggml_is_contiguously_allocated(const struct ggml_tensor*) { unused("ggml_is_contiguously_allocated"); }
bool ggml_is_quantized(enum ggml_type) { unused("ggml_is_quantized"); }
size_t ggml_nbytes(const struct ggml_tensor*) { unused("ggml_nbytes"); }
int64_t ggml_nelements(const struct ggml_tensor*) { unused("ggml_nelements"); }
const char* ggml_type_name(enum ggml_type) { unused("ggml_type_name"); }
size_t ggml_type_size(enum ggml_type) { unused("ggml_type_size"); }
}
