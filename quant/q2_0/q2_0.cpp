// SPDX-License-Identifier: Apache-2.0
// Q2_0 pack: repack, activation quantization, scalar references and the expert driver.
// The format and ggml's reference semantics are from ggml (MIT): ggml-common.h,
// ggml-quants.c (quantize_row_q8_0_ref) and ggml-cpu/quants.c
// (ggml_vec_dot_q2_0_q8_0_generic); the code here is written for flashrt.
#include "quant/q2_0/q2_0.hpp"

#include "core/fp16.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace flashrt::q2_0 {

namespace {
constexpr size_t kAlign = 64;
size_t up64(size_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }

// ggml element j (byte j/4, bit 2*(j%4)) -> planar byte j%16, bit 2*(j/16)
void repack_block(const GgmlBlock& b, uint8_t* out16) {
    std::memset(out16, 0, 16);
    for (int j = 0; j < kBlock; ++j) {
        const int q = (b.qs[j / 4] >> (2 * (j % 4))) & 3;
        out16[j % 16] |= uint8_t(q << (2 * (j / 16)));
    }
}
int planar_code(const uint8_t* blk16, int j) { return (blk16[j % 16] >> (2 * (j / 16))) & 3; }
}  // namespace

size_t mat_bytes(int rows, int cols) { return size_t(rows) * (cols / kBlock) * kBlockBytes; }

Mat mat_view(const void* src, int rows, int cols) {
    const auto* p = static_cast<const uint8_t*>(src);
    const size_t nblk = size_t(rows) * (cols / kBlock);
    return Mat{p, reinterpret_cast<const uint16_t*>(p + nblk * 16), rows, cols};
}

Mat repack(const void* ggml_rows, int rows, int cols, void* dst) {
    assert(cols % kBlock == 0);
    const auto* in = static_cast<const GgmlBlock*>(ggml_rows);
    auto* codes = static_cast<uint8_t*>(dst);
    const size_t nblk = size_t(rows) * (cols / kBlock);
    auto* scales = reinterpret_cast<uint16_t*>(codes + nblk * 16);
    for (size_t i = 0; i < nblk; ++i) {
        repack_block(in[i], codes + i * 16);
        scales[i] = in[i].d;
    }
    return mat_view(dst, rows, cols);
}

size_t expert_bytes(ExpertShape s) { return 2 * mat_bytes(s.d_ff, s.d_model) + mat_bytes(s.d_model, s.d_ff); }

Expert expert_view(const void* blob, ExpertShape s) {
    const auto* p = static_cast<const uint8_t*>(blob);
    const size_t gu = mat_bytes(s.d_ff, s.d_model);
    return Expert{mat_view(p, s.d_ff, s.d_model), mat_view(p + gu, s.d_ff, s.d_model),
                  mat_view(p + 2 * gu, s.d_model, s.d_ff)};
}

Expert repack_expert(const void* gate, const void* up, const void* down, ExpertShape s, void* blob) {
    auto* p = static_cast<uint8_t*>(blob);
    const size_t gu = mat_bytes(s.d_ff, s.d_model);
    repack(gate, s.d_ff, s.d_model, p);
    repack(up, s.d_ff, s.d_model, p + gu);
    repack(down, s.d_model, s.d_ff, p + 2 * gu);
    return expert_view(blob, s);
}

size_t q8_bytes(int n) {
    return up64(size_t(n)) + up64(size_t(n / kQ8Block) * 4) + up64(size_t(n / kBlock) * 16 * 4) + up64(size_t(n / 4) * 4);
}

Q8Act q8_view(void* mem, int n) {
    assert(n % kBlock == 0);
    auto* p = static_cast<uint8_t*>(mem);
    Q8Act a;
    a.n = n;
    a.qs = reinterpret_cast<int8_t*>(p);
    p += up64(size_t(n));
    a.d = reinterpret_cast<float*>(p);
    p += up64(size_t(n / kQ8Block) * 4);
    a.dvec = reinterpret_cast<float*>(p);
    p += up64(size_t(n / kBlock) * 16 * 4);
    a.negsum4 = reinterpret_cast<int32_t*>(p);
    return a;
}

void quantize_q8(const float* x, Q8Act& a) {
    for (int i = 0; i < a.n / kQ8Block; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < kQ8Block; ++j) amax = std::max(amax, std::fabs(x[i * kQ8Block + j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        a.d[i] = fp16_to_fp32(fp32_to_fp16(d));   // ggml stores d as fp16 and reads it back
        for (int j = 0; j < kQ8Block; ++j) a.qs[i * kQ8Block + j] = int8_t(std::roundf(x[i * kQ8Block + j] * id));
    }
    for (int b = 0; b < a.n / kBlock; ++b)
        for (int l = 0; l < 16; ++l) a.dvec[b * 16 + l] = a.d[2 * b + (l >= 8)];
    for (int i = 0; i < a.n / 4; ++i)
        a.negsum4[i] = -(a.qs[4 * i] + a.qs[4 * i + 1] + a.qs[4 * i + 2] + a.qs[4 * i + 3]);
}

// Same arithmetic, in the same order, as ggml_vec_dot_q2_0_q8_0_generic.
void matvec_ref(const Mat& w, const Q8Act* a, int n_tok, int r0, int r1, float* y, int ldy) {
    const int nb = w.nb();
    for (int t = 0; t < n_tok; ++t) {
        for (int r = r0; r < r1; ++r) {
            float sumf = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = w.codes + (size_t(r) * nb + b) * 16;
                const float d0 = fp16_to_fp32(w.scales[size_t(r) * nb + b]);
                float sumi = 0.0f;
                for (int k = 0; k < 2; ++k) {
                    int s = 0;
                    for (int j = 0; j < kQ8Block; ++j)
                        s += (planar_code(blk, k * kQ8Block + j) - 1) * a[t].qs[b * kBlock + k * kQ8Block + j];
                    sumi += a[t].d[2 * b + k] * float(s);
                }
                sumf += d0 * sumi;
            }
            y[size_t(t) * ldy + r] = sumf;
        }
    }
}

float dot_ggml(const GgmlBlock* w, const Q8Act& a) {
    float sumf = 0.0f;
    for (int b = 0; b < a.n / kBlock; ++b) {
        const float d0 = fp16_to_fp32(w[b].d);
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            int s = 0;
            for (int i = 0; i < 8; ++i) {
                const uint8_t byte = w[b].qs[k * 8 + i];
                for (int e = 0; e < 4; ++e) s += (((byte >> (2 * e)) & 3) - 1) * a.qs[b * kBlock + k * kQ8Block + 4 * i + e];
            }
            sumi += a.d[2 * b + k] * float(s);
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

size_t expert_scratch_bytes(ExpertShape s) {
    return 4 * (2 * up64(size_t(s.d_ff) * 4) + q8_bytes(s.d_ff) + kAlign) + kAlign;
}

void expert_ffn(const Expert& e, const Q8Act* x, int n_tok, float* out, int ldo, void* scratch, bool use_avx512) {
    assert(n_tok >= 1 && n_tok <= 4);
    const int ff = e.gate.rows;
    auto base = (reinterpret_cast<uintptr_t>(scratch) + kAlign - 1) & ~uintptr_t(kAlign - 1);
    auto* g = reinterpret_cast<float*>(base);
    auto* u = g + 4 * up64(size_t(ff) * 4) / 4;
    auto* hq = reinterpret_cast<uint8_t*>(u + 4 * up64(size_t(ff) * 4) / 4);
    const int ld = int(up64(size_t(ff) * 4) / 4);
    auto mv = use_avx512 ? matvec_avx512 : matvec_ref;

    mv(e.gate, x, n_tok, 0, ff, g, ld);
    mv(e.up, x, n_tok, 0, ff, u, ld);
    Q8Act h[4];
    for (int t = 0; t < n_tok; ++t) {
        float* gt = g + size_t(t) * ld;
        const float* ut = u + size_t(t) * ld;
        for (int i = 0; i < ff; ++i) gt[i] = gt[i] / (1.0f + std::exp(-gt[i])) * ut[i];   // silu(g) * u
        h[t] = q8_view(hq + size_t(t) * q8_bytes(ff), ff);
        quantize_q8(gt, h[t]);
    }
    mv(e.down, h, n_tok, 0, e.down.rows, out, ldo);
}

}  // namespace flashrt::q2_0
