// SPDX-License-Identifier: Apache-2.0
// Q2_0 quant pack: repacked expert blobs, activation quantization, and the expert matvec
// for the CPU miss path.
//
// Format (ggml GGML_TYPE_Q2_0, 42): blocks of 64 weights, an fp16 scale d and 16 bytes of
// 2-bit codes; code q decodes to (q - 1) * d, i.e. {-1, 0, +1, +2} * d. Element j of a
// block sits in byte j/4 at bit 2*(j%4). ggml pairs it with Q8_0 activations (blocks of
// 32, int8 codes, fp16 scale amax/127), so each weight block meets two activation blocks.
//
// Repacked layout (same 18 bytes per block, reordered; our design):
//   codes  [rows][cols/64][16]  "planar": byte i of a block holds elements i, i+16, i+32,
//                               i+48 at bits 0, 2, 4, 6. One 128-bit broadcast, a per-lane
//                               shift of 0/2/4/6 and a mask give all 64 codes in order.
//   scales [rows][cols/64]      fp16, after all codes
// One row's codes are contiguous, so a row range is one sequential read.
#pragma once

#include <cstddef>
#include <cstdint>

namespace flashrt::q2_0 {

constexpr int kBlock = 64;        // weights per block
constexpr int kBlockBytes = 18;   // fp16 scale + 16 code bytes
constexpr int kQ8Block = 32;      // activation block

struct GgmlBlock {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(GgmlBlock) == kBlockBytes);

// A repacked rows x cols matrix (cols a multiple of 64).
struct Mat {
    const uint8_t* codes;     // [rows][cols/64][16]
    const uint16_t* scales;   // [rows][cols/64]
    int rows, cols;
    int nb() const { return cols / kBlock; }
};

size_t mat_bytes(int rows, int cols);
// Repacks `rows` rows of ggml blocks into `dst` (mat_bytes(rows, cols) bytes).
Mat repack(const void* ggml_rows, int rows, int cols, void* dst);
Mat mat_view(const void* src, int rows, int cols);

// One routed expert, SwiGLU: out = down(silu(gate x) * up x); gate and up are
// [d_ff x d_model], down is [d_model x d_ff]. The blob is gate, up, down back to back.
struct ExpertShape {
    int d_model, d_ff;
};
struct Expert {
    Mat gate, up, down;
};
size_t expert_bytes(ExpertShape s);
Expert repack_expert(const void* gate, const void* up, const void* down, ExpertShape s, void* blob);
Expert expert_view(const void* blob, ExpertShape s);

// Activations quantized exactly like ggml's quantize_row_q8_0_ref, plus the kernel's
// helpers: per-64-block scale vectors (lanes 0-7 the first half's scale, 8-15 the
// second's) and negated sums of each 4 consecutive codes.
struct Q8Act {
    int n = 0;                 // elements, a multiple of 64
    int8_t* qs = nullptr;      // [n]
    float* d = nullptr;        // [n/32], fp16-rounded, as float
    float* dvec = nullptr;     // [n/64][16]
    int32_t* negsum4 = nullptr;  // [n/4]
};
size_t q8_bytes(int n);                  // for one activation vector, 64-byte aligned parts
Q8Act q8_view(void* mem, int n);         // mem: q8_bytes(n), 64-byte aligned
void quantize_q8(const float* x, Q8Act& a);

// y[t * ldy + r] = row r of w . activation t, for r in [r0, r1) and t < n_tok (<= 4).
void matvec_ref(const Mat& w, const Q8Act* a, int n_tok, int r0, int r1, float* y, int ldy);
void matvec_avx512(const Mat& w, const Q8Act* a, int n_tok, int r0, int r1, float* y, int ldy);
bool have_avx512();

// ggml's own dot on ggml-layout blocks (mirrors ggml_vec_dot_q2_0_q8_0_generic), for tests.
float dot_ggml(const GgmlBlock* w, const Q8Act& a);

// The whole expert for n_tok tokens on one thread. scratch: expert_scratch_bytes(s).
size_t expert_scratch_bytes(ExpertShape s);
void expert_ffn(const Expert& e, const Q8Act* x, int n_tok, float* out, int ldo, void* scratch, bool use_avx512);

}  // namespace flashrt::q2_0
