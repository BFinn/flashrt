// SPDX-License-Identifier: Apache-2.0
// fr_kld: KL divergence of flashrt's logits against a llama-perplexity --kl-divergence-base
// file (the P1 correctness gate), on the same tokens and chunks.
//
//   fr_kld MODEL.gguf BASE.bin [--ctx N] [--chunks K] [--batch B]
//
// The base file holds: the magic "_logits_", int32 ctx, int32 n_vocab, int32 n_chunk, the
// tokens of all chunks (n_chunk * ctx int32), then for every chunk the scored positions ctx/2 .. ctx-2, each as a float scale and a
// float minimum log-prob (4 uint16) and n_vocab uint16 log-probs (padded to even). Each chunk
// is run from a fresh state; KLD, same-top-1 and PPL follow llama-perplexity's formulas
// (reference probabilities below e^-16 are ignored).
#include "arch/qwen4exp/experts.hpp"
#include "arch/qwen4exp/forward_ref.hpp"
#include "arch/qwen4exp/gpu_weights.hpp"
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace flashrt;
using namespace flashrt::qwen4exp;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: fr_kld MODEL.gguf BASE.bin --ctx N [--chunks K] [--batch B]\n");
        return 2;
    }
    int ctx = 0, chunks = 0, batch = 64;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--ctx")) ctx = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--chunks")) chunks = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--batch")) batch = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    FILE* bf = std::fopen(argv[2], "rb");
    if (!bf) { std::perror(argv[2]); return 1; }
    char magic[8];
    int32_t file_ctx = 0, n_vocab = 0, n_chunk = 0;
    if (std::fread(magic, 1, 8, bf) != 8 || std::memcmp(magic, "_logits_", 8) != 0 || std::fread(&file_ctx, 4, 1, bf) != 1 ||
        std::fread(&n_vocab, 4, 1, bf) != 1 || std::fread(&n_chunk, 4, 1, bf) != 1) {
        std::fprintf(stderr, "not a llama-perplexity KL base file\n");
        return 1;
    }
    if (ctx == 0) ctx = file_ctx;
    if (ctx != file_ctx) { std::fprintf(stderr, "--ctx %d does not match the base file's %d\n", ctx, file_ctx); return 1; }
    std::vector<int32_t> tokens(size_t(n_chunk) * ctx);
    if (std::fread(tokens.data(), 4, tokens.size(), bf) != tokens.size()) { std::fprintf(stderr, "base file too short\n"); return 1; }
    if (chunks <= 0 || chunks > n_chunk) chunks = n_chunk;
    const int first = ctx / 2, n_scored = ctx - 1 - first;
    const size_t nv = size_t(2 * ((n_vocab + 1) / 2) + 4);
    const long data0 = std::ftell(bf);

    const Gguf g = Gguf::open(argv[1]);
    const Spec s = parse(g);
    if (s.n_vocab != n_vocab) { std::fprintf(stderr, "vocab mismatch: model %d, base %d\n", s.n_vocab, n_vocab); return 1; }
    const WeightPlan plan = qwen4exp::plan(g, s);
    GpuWeights w;
    w.load(g, plan);
    ExpertArena arena = arena_alloc(s.n_layer, s.n_expert, q2_0::expert_bytes({s.d_model, s.d_ff_expert}), PageMode::THP, 0);
    if (!arena.buf.ptr) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    load_experts(g, s, arena, 12);
    CpuPool pool(8, physical_cpus());
    ForwardRef fwd(g, s, w, arena, pool, ctx + 8, batch);

    float* logits_dev = nullptr;
    cudaMalloc(&logits_dev, size_t(batch) * n_vocab * 4);
    std::vector<float> logits(size_t(batch) * n_vocab);
    std::vector<uint16_t> base(nv);
    std::vector<double> klds;
    double sum_nll = 0, sum_nll_base = 0;
    long same_top = 0, count = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (int ch = 0; ch < chunks; ++ch) {
        const int32_t* seq = tokens.data() + size_t(ch) * ctx;
        fwd.reset();
        for (int p0 = 0; p0 < ctx; p0 += batch) {
            const int T = std::min(batch, ctx - p0);
            // rows whose logits are scored: positions first .. ctx-2
            const int lo = std::max(first, p0), hi = std::min(ctx - 2, p0 + T - 1);
            const int out_from = lo <= hi ? lo - p0 : T;
            fwd.forward(seq, T, out_from, lo <= hi ? logits_dev : nullptr);
            if (lo > hi) continue;
            const int R = T - out_from;
            cudaMemcpy(logits.data(), logits_dev, size_t(R) * n_vocab * 4, cudaMemcpyDeviceToHost);
            for (int p = lo; p <= hi; ++p) {
                const float* lg = logits.data() + size_t(p - lo) * n_vocab;
                const long rec = long(ch) * n_scored + (p - first);
                std::fseek(bf, data0 + rec * long(nv * 2), SEEK_SET);
                if (std::fread(base.data(), 2, nv, bf) != nv) { std::fprintf(stderr, "base record %ld missing\n", rec); return 1; }
                float scale, min_lp;
                std::memcpy(&scale, &base[0], 4);
                std::memcpy(&min_lp, &base[2], 4);
                const uint16_t* bl = base.data() + 4;
                // ours: log-softmax
                float mx = lg[0];
                int imax = 0;
                for (int v = 1; v < n_vocab; ++v)
                    if (lg[v] > mx) { mx = lg[v]; imax = v; }
                double se = 0;
                for (int v = 0; v < n_vocab; ++v) se += std::exp(double(lg[v] - mx));
                const double lse = std::log(se) + mx;
                const int tok = seq[p + 1];
                sum_nll += lse - lg[tok];
                sum_nll_base += -(scale * bl[tok] + min_lp);
                double kl = 0;
                int imax_b = 0;
                float lp_max = -INFINITY;
                for (int v = 0; v < n_vocab; ++v) {
                    const float lpb = scale * bl[v] + min_lp;
                    if (lpb > lp_max) { lp_max = lpb; imax_b = v; }
                    if (lpb > -16.0f) kl += std::exp(double(lpb)) * (lpb - (lg[v] - lse));
                }
                klds.push_back(kl);
                same_top += imax == imax_b;
                ++count;
            }
            if ((p0 / batch) % 16 == 15 || p0 + T >= ctx) {
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double m = 0;
                for (double k : klds) m += k;
                std::printf("chunk %d pos %5d: %ld scored, mean KLD %.5f, same top %.2f%%, %.1f tok/s\n", ch, p0 + T, count,
                            count ? m / count : 0.0, count ? 100.0 * same_top / count : 0.0,
                            (double(ch) * ctx + p0 + T) / el);
                std::fflush(stdout);
            }
        }
    }
    std::vector<double> sorted = klds;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0;
    for (double k : klds) mean += k;
    mean /= std::max<size_t>(1, klds.size());
    auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted[size_t(q * (sorted.size() - 1))]; };
    std::printf("\nfr_kld: %ld tokens over %d chunk(s) of %d\n", count, chunks, ctx);
    std::printf("  PPL flashrt %.4f, base %.4f, ratio %.5f\n", std::exp(sum_nll / count), std::exp(sum_nll_base / count),
                std::exp((sum_nll - sum_nll_base) / count));
    std::printf("  KLD mean %.6f, median %.6f, p99 %.6f, p99.9 %.6f, max %.6f\n", mean, pct(0.5), pct(0.99), pct(0.999),
                sorted.empty() ? 0.0 : sorted.back());
    std::printf("  same top token %.3f%%\n", 100.0 * same_top / std::max(1L, count));
    std::fclose(bf);
    cudaFree(logits_dev);
    arena_free(arena);
    return 0;
}
