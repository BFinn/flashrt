// SPDX-License-Identifier: Apache-2.0
// IEEE half <-> float, portable and exact (round to nearest even), for scales in quant packs.
// Hot loops convert with vcvtph2ps instead.
#pragma once

#include <cstdint>
#include <cstring>

namespace flashrt {

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                   // subnormal: normalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);    // inf / nan
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const uint32_t absx = x & 0x7fffffff;
    if (absx >= 0x7f800000) return uint16_t(sign | 0x7c00 | (absx > 0x7f800000 ? 0x200 : 0));   // inf / nan
    if (absx >= 0x477ff000) return uint16_t(sign | 0x7c00);                                      // overflow -> inf
    if (absx < 0x38800000) {                                                                     // subnormal or zero
        if (absx < 0x33000000) return uint16_t(sign);
        const uint32_t e = absx >> 23, m = (absx & 0x7fffff) | 0x800000;
        const uint32_t shift = 126 - e;                                  // 14..24
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = ((absx >> 13) - ((127 - 15) << 10));
    const uint32_t rem = absx & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return uint16_t(sign | h);
}

}  // namespace flashrt
