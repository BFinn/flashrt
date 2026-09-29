// SPDX-License-Identifier: Apache-2.0
// The MTP head's Q8_0 -> Q2_0 conversion (convert_q8_0_to_q2_0, run at load with --mtp-bits 2)
// against the conversion as first written, which was 10x slower (sw98): the outputs must be
// bit-identical. Synthetic blocks cover the edge cases (zero scales,
// scales that round to zero or overflow in fp16, exact halves, one large value). With
// FLASHRT_TEST_MTP=<the MTP GGUF> the head's real expert tensors are compared too, and both
// conversions are timed. A .cu file so that nvcc compiles it as it compiles mtp.cu: the float
// contractions, and so the bits, depend on the compiler's flags.
//
//   test_mtp_convert
#include "arch/qwen4exp/mtp.hpp"
#include "core/fp16.hpp"
#include "core/gguf.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace flashrt;

namespace {
// verbatim: the conversion as mtp.cu had it before sw98
void reference(const uint8_t* src, uint8_t* dst, size_t nblocks64) {
    for (size_t b = 0; b < nblocks64; ++b, src += 68, dst += 18) {
        float x[64], amax = 0.0f;
        for (int h = 0; h < 2; ++h) {
            const uint8_t* s8 = src + 34 * h;
            const float d8 = fp16_to_fp32(uint16_t(s8[0] | (s8[1] << 8)));
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * h + j] = d8 * float(int8_t(s8[2 + j]))));
        }
        float best_d = 0.0f, best_e = INFINITY;
        for (int c = 0; c < 24 && amax > 0.0f; ++c) {
            const float d = amax / (2.5f - 1.5f * float(c) / 23.0f);
            const uint16_t dh = fp32_to_fp16(d);
            const float dq = fp16_to_fp32(dh);
            float e = 0.0f;
            for (int j = 0; j < 64; ++j) {
                const int q = std::min(2, std::max(-1, int(std::lround(x[j] / dq))));
                const float r = x[j] - float(q) * dq;
                e += r * r;
            }
            if (e < best_e) { best_e = e; best_d = dq; }
        }
        const uint16_t dh = fp32_to_fp16(best_d);
        dst[0] = uint8_t(dh & 0xff);
        dst[1] = uint8_t(dh >> 8);
        std::memset(dst + 2, 0, 16);
        for (int j = 0; j < 64; ++j) {
            const int q = best_d > 0.0f ? std::min(2, std::max(-1, int(std::lround(x[j] / best_d)))) : 0;
            dst[2 + j / 4] |= uint8_t((q + 1) << (2 * (j % 4)));
        }
    }
}

void reference_mt(const uint8_t* src, uint8_t* dst, size_t n, int threads) {
    std::vector<std::thread> th;
    for (int k = 0; k < threads; ++k)
        th.emplace_back([=] {
            const size_t a = n * size_t(k) / size_t(threads), e = n * size_t(k + 1) / size_t(threads);
            reference(src + a * 68, dst + a * 18, e - a);
        });
    for (auto& t : th) t.join();
}

// blocks of 64 values that differ from the reference
size_t compare(const std::vector<uint8_t>& src, size_t n, int threads, double* t_ref, double* t_new) {
    std::vector<uint8_t> a(n * 18), b(n * 18);
    auto t0 = std::chrono::steady_clock::now();
    reference_mt(src.data(), a.data(), n, 8);   // as the head's load ran it
    auto t1 = std::chrono::steady_clock::now();
    qwen4exp::convert_q8_0_to_q2_0(src.data(), b.data(), n, threads);
    auto t2 = std::chrono::steady_clock::now();
    if (t_ref) *t_ref += std::chrono::duration<double>(t1 - t0).count();
    if (t_new) *t_new += std::chrono::duration<double>(t2 - t1).count();
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff += std::memcmp(a.data() + 18 * i, b.data() + 18 * i, 18) != 0;
    return diff;
}
}  // namespace

int main() {
    const int threads = int(std::max(8u, std::thread::hardware_concurrency()));
    int fail = 0;
    // synthetic: 2^20 blocks of 64 with the edge cases mixed in
    {
        const size_t n = size_t(1) << 20;
        std::vector<uint8_t> src(n * 68);
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0, 40);
        for (size_t i = 0; i < n * 2; ++i) {
            uint8_t* s = src.data() + 34 * i;
            float d8;
            switch (i % 97) {
                case 0: d8 = 0; break;              // a zero scale
                case 1: d8 = 3e-8f; break;          // d rounds to 0 in fp16
                case 2: d8 = 6e-8f; break;          // subnormal
                case 3: d8 = 60000.f; break;        // d overflows fp16
                default: d8 = std::ldexp(1.0f, -int(rng() % 20)) * 0.01f;
            }
            const uint16_t h = fp32_to_fp16(d8);
            s[0] = uint8_t(h);
            s[1] = uint8_t(h >> 8);
            for (int j = 0; j < 32; ++j) {
                int v = i % 13 == 5 ? (j == 0 ? 127 : 0) : int(std::lround(nd(rng)));
                if (i % 11 == 7) v = (j % 2) ? 64 : -64;   // exact halves after scaling
                s[2 + j] = uint8_t(int8_t(std::max(-127, std::min(127, v))));
            }
        }
        const size_t d = compare(src, n, threads, nullptr, nullptr);
        std::printf("synthetic: %zu blocks of 64, %zu differ from the reference %s\n", n, d, d ? "FAIL" : "ok");
        fail += d != 0;
    }
    // the head's expert tensors
    if (const char* path = std::getenv("FLASHRT_TEST_MTP")) {
        const Gguf g = Gguf::open(path);
        size_t total = 0, diff = 0;
        double t_ref = 0, t_new = 0;
        for (const GgufTensor& t : g.tensors) {
            if (t.type != 8 || t.name.find("_exps.weight") == std::string::npos) continue;   // Q8_0 experts
            const size_t n = size_t(t.n_elements()) / 64;
            std::vector<uint8_t> src(n * 68);
            const int fd = open(g.shards[t.shard].c_str(), O_RDONLY);
            for (size_t r = 0; r < src.size();) {
                const ssize_t got = pread(fd, src.data() + r, src.size() - r, off_t(t.file_offset + r));
                if (got <= 0) {
                    std::printf("short read of %s\n", t.name.c_str());
                    return 2;
                }
                r += size_t(got);
            }
            close(fd);
            diff += compare(src, n, threads, &t_ref, &t_new);
            total += n;
        }
        std::printf("the head's experts: %zu blocks of 64, %zu differ %s; reference (8 threads) %.1f s, now (%d threads) %.1f s\n",
                    total, diff, diff || !total ? "FAIL" : "ok", t_ref, threads, t_new);
        fail += diff != 0 || total == 0;
    } else {
        std::printf("FLASHRT_TEST_MTP not set: the real head not compared\n");
    }
    std::printf("%s\n", fail ? "FAILED" : "all passed");
    return fail ? 1 : 0;
}
