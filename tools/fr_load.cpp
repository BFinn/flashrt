// SPDX-License-Identifier: Apache-2.0
// fr_load: load every routed expert of a qwen4exp GGUF into the host arena and check it.
//
//   fr_load MODEL.gguf [--threads N] [--pages 4k|thp] [--verify N]
//
// Reports load time and read rate. Verification works independently of the repack: for N
// random (layer, expert) pairs, a random row of each matrix is read straight from the GGUF
// and dotted with ggml's semantics (dot_ggml), and compared with the AVX-512 kernel on the
// arena's repacked blob. One expert also runs through the whole FFN, AVX-512 vs reference.
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace flashrt;
using namespace flashrt::q2_0;

namespace {

struct Act {
    std::vector<uint8_t> mem;
    Q8Act a;
    Act(const std::vector<float>& x) : mem(q8_bytes(int(x.size())) + 64) {
        auto p = (reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63);
        a = q8_view(reinterpret_cast<void*>(p), int(x.size()));
        quantize_q8(x.data(), a);
    }
};

std::vector<GgmlBlock> read_row(const Gguf& g, const GgufTensor& t, uint64_t off, int cols) {
    std::vector<GgmlBlock> row(cols / kBlock);
    const int fd = open(g.shards[t.shard].c_str(), O_RDONLY);
    if (fd < 0 || pread(fd, row.data(), row.size() * sizeof(GgmlBlock), off_t(t.file_offset + off)) !=
                      ssize_t(row.size() * sizeof(GgmlBlock))) {
        std::perror("read_row");
        std::exit(1);
    }
    close(fd);
    return row;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fr_load MODEL.gguf [--threads N] [--pages 4k|thp] [--verify N]\n");
        return 2;
    }
    int threads = 12, verify = 64;
    PageMode mode = PageMode::THP;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--pages")) mode = parse_page_mode(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--verify")) verify = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    const Gguf g = Gguf::open(argv[1]);
    const qwen4exp::Spec s = qwen4exp::parse(g);
    const ExpertShape shape{s.d_model, s.d_ff_expert};

    const auto ta = std::chrono::steady_clock::now();
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, expert_bytes(shape), mode, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    const double alloc_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
    std::printf("arena: %d x %d blobs of %zu B (stride %zu), %.2f GiB, pages=%s, mapped in %.2f s\n", s.n_layer,
                s.n_expert, arena.blob_bytes, arena.stride, arena.total_bytes() / 1073741824.0, page_mode_name(mode), alloc_s);

    const qwen4exp::LoadStats st = qwen4exp::load_experts(g, s, arena, threads);
    std::printf("huge-page share after load: %.2f\n", huge_page_fraction(arena.buf));
    std::printf("loaded %d experts in %.1f s: %.2f GB read at %.2f GB/s (%d threads, O_DIRECT)\n", s.n_layer * s.n_expert,
                st.seconds, st.bytes_read / 1e9, st.bytes_read / 1e9 / st.seconds, threads);

    // verification against ggml semantics on the original bytes
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xd(s.d_model), xf(s.d_ff_expert);
    for (auto& v : xd) v = nd(rng);
    for (auto& v : xf) v = nd(rng);
    Act ad(xd), af(xf);
    int bad = 0;
    double worst = 0;
    for (int n = 0; n < verify; ++n) {
        const int l = int(rng() % s.n_layer), e = int(rng() % s.n_expert);
        const Expert ex = expert_view(arena.blob(l, e), shape);
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        const Mat* mats[3] = {&ex.gate, &ex.up, &ex.down};
        for (int k = 0; k < 3; ++k) {
            const GgufTensor* t = g.tensor("blk." + std::to_string(l) + "." + names[k] + ".weight");
            const Mat& m = *mats[k];
            const int r = int(rng() % m.rows);
            const uint64_t off = (uint64_t(e) * m.rows + r) * (m.cols / kBlock) * sizeof(GgmlBlock);
            const auto row = read_row(g, *t, off, m.cols);
            const Q8Act& a = m.cols == s.d_model ? ad.a : af.a;
            const float want = dot_ggml(row.data(), a);
            std::vector<float> y(m.rows);
            matvec_avx512(m, &a, 1, r, r + 1, y.data(), m.rows);
            const float got = y[r];
            const double err = std::fabs(got - want) / (std::fabs(want) + 1e-2);
            worst = std::max(worst, err);
            if (err > 1e-4) {
                if (bad < 5) std::printf("  MISMATCH layer %d expert %d %s row %d: arena %g, gguf %g\n", l, e, names[k], r, got, want);
                ++bad;
            }
        }
    }
    std::printf("verify: %d random experts x 3 matrices, %d mismatches, worst rel err %.2e\n", verify, bad, worst);

    // one real expert end to end
    std::vector<uint8_t> scratch(expert_scratch_bytes(shape) + 64);
    std::vector<float> o_ref(s.d_model), o_avx(s.d_model);
    const Expert ex = expert_view(arena.blob(s.n_layer / 2, 123), shape);
    expert_ffn(ex, &ad.a, 1, o_ref.data(), s.d_model, scratch.data(), false);
    expert_ffn(ex, &ad.a, 1, o_avx.data(), s.d_model, scratch.data(), true);
    double num = 0, den = 0;
    bool finite = true;
    for (int i = 0; i < s.d_model; ++i) {
        num += double(o_avx[i] - o_ref[i]) * (o_avx[i] - o_ref[i]);
        den += double(o_ref[i]) * o_ref[i];
        finite &= std::isfinite(o_avx[i]);
    }
    std::printf("expert (layer %d, 123) FFN: rms %.4g, avx512 vs ref rel L2 %.2e, finite %s\n", s.n_layer / 2,
                std::sqrt(den / s.d_model), std::sqrt(num / den), finite ? "yes" : "NO");
    const bool ok = bad == 0 && finite && std::sqrt(num / den) < 1e-4;
    std::printf("%s\n", ok ? "fr_load OK" : "fr_load FAILED");
    arena_free(arena);
    return ok ? 0 : 1;
}
