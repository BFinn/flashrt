// SPDX-License-Identifier: Apache-2.0
// Q2_0 x Q8 matvec with AVX-512 (F, BW, VNNI) over the planar repacked layout.
//
// Per 64-weight block and token:
//   codes: one 16-byte load broadcast to four 128-bit lanes, shifted right by 0/2/4/6 per
//          lane and masked to 2 bits: all 64 codes in element order, as unsigned bytes 0..3
//   dot:   vpdpbusd(codes, activation int8) accumulating onto the negated 4-element sums,
//          so each of the 16 dword lanes holds sum((q - 1) * a) over 4 elements exactly
//   scale: lanes 0-7 belong to the first activation half, 8-15 to the second; convert,
//          multiply by the per-lane activation scale, then FMA with the block's weight scale
// The weight codes are unpacked once per block and reused for up to four tokens, so a
// multi-token call reads each expert once.
#include "quant/q2_0/q2_0.hpp"

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

#include <immintrin.h>

namespace flashrt::q2_0 {

namespace {

template <int NT>
void matvec_nt(const Mat& w, const Q8Act* a, int r0, int r1, float* y, int ldy) {
    const int nb = w.nb();
    const __m512i shifts = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
    const __m512i mask3 = _mm512_set1_epi8(3);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* codes = w.codes + size_t(r) * nb * 16;
        const uint16_t* scales = w.scales + size_t(r) * nb;
        __m512 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
        for (int b0 = 0; b0 < nb; b0 += 16) {
            const int nblk = nb - b0 < 16 ? nb - b0 : 16;
            const __m256i sh = _mm256_maskz_loadu_epi16(__mmask16((1u << nblk) - 1), scales + b0);
            const __m512 dw16 = _mm512_cvtph_ps(sh);
            for (int k = 0; k < nblk; ++k) {
                const int b = b0 + k;
                __m512i q = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + size_t(b) * 16)));
                q = _mm512_and_si512(_mm512_srlv_epi64(q, shifts), mask3);
                const __m512 dw = _mm512_permutexvar_ps(_mm512_set1_epi32(k), dw16);
                for (int t = 0; t < NT; ++t) {
                    const __m512i ns = _mm512_load_si512(a[t].negsum4 + b * 16);
                    const __m512i av = _mm512_loadu_si512(a[t].qs + b * kBlock);
                    const __m512i s = _mm512_dpbusd_epi32(ns, q, av);
                    const __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(s), _mm512_load_ps(a[t].dvec + b * 16));
                    acc[t] = _mm512_fmadd_ps(f, dw, acc[t]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) y[size_t(t) * ldy + r] = _mm512_reduce_add_ps(acc[t]);
    }
}

}  // namespace

void matvec_avx512(const Mat& w, const Q8Act* a, int n_tok, int r0, int r1, float* y, int ldy) {
    switch (n_tok) {
        case 1: matvec_nt<1>(w, a, r0, r1, y, ldy); break;
        case 2: matvec_nt<2>(w, a, r0, r1, y, ldy); break;
        case 3: matvec_nt<3>(w, a, r0, r1, y, ldy); break;
        default: matvec_nt<4>(w, a, r0, r1, y, ldy); break;
    }
}

bool have_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vnni");
}

}  // namespace flashrt::q2_0

#else  // no AVX-512 VNNI at compile time: fall back to the reference

namespace flashrt::q2_0 {
void matvec_avx512(const Mat& w, const Q8Act* a, int n_tok, int r0, int r1, float* y, int ldy) {
    matvec_ref(w, a, n_tok, r0, r1, y, ldy);
}
bool have_avx512() { return false; }
}  // namespace flashrt::q2_0

#endif
