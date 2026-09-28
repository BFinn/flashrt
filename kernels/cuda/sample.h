// SPDX-License-Identifier: Apache-2.0
// Token sampling on the GPU, one logits row per CUDA block.
//
// The chain follows llama.cpp's default order: top-k, then top-p over the softmax of the top-k
// candidates (at temperature 1; the smallest prefix whose mass reaches top_p, at least one
// token), then min-p (drop candidates below min_p times the top one's probability), then the
// temperature, then one draw. temperature <= 0 is greedy (argmax, lowest index on ties).
//
// The draw for the token at position p uses u = rng(seed, p): a counter-based generator, so a
// token gets the same draw whether it is sampled in a plain decode step or in a speculative
// verify window. That makes speculative sampling reproduce plain sampling token for token
// wherever the logits agree (exact speculative sampling with argmax drafts: sample every verify
// row, accept while the sample equals the draft).
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace flashrt::sample {

constexpr int kMaxTopK = 64;

struct Params {
    float temperature = 1.0f;   // <= 0: greedy
    int top_k = 20;             // 1 .. kMaxTopK
    float top_p = 0.95f;        // 1: off
    float min_p = 0.0f;         // 0: off
};

// out_dev[r] = the token sampled from row r of logits [rows][n_vocab], at position pos0 + r.
void sample_rows(const float* logits, int rows, int n_vocab, const Params& p, uint64_t seed, int64_t pos0, int32_t* out_dev,
                 cudaStream_t stream);

// The uniform draw in [0, 1) for position pos (host and device agree bit for bit).
__host__ __device__ inline float draw(uint64_t seed, int64_t pos) {
    uint64_t z = seed ^ (uint64_t(pos) * 0x9E3779B97F4A7C15ull) ^ 0xD1B54A32D192ED03ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return float(z >> 40) * (1.0f / 16777216.0f);
}

// The chain after top-k (device and host, so tests can check the kernel): given the top-k
// candidates' logits (sorted descending, ties by id) and ids, the chosen id for draw u.
__host__ __device__ inline int32_t choose(const float* cl, const int32_t* ci, int k, const Params& p, float u) {
    if (p.temperature <= 0.0f || k == 1) return ci[0];
    // top-p over the softmax at temperature 1
    float pr[kMaxTopK], sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += pr[i] = expf(cl[i] - cl[0]);
    int n = k;
    if (p.top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += pr[i] / sum;
            if (cum >= p.top_p) {
                n = i + 1;
                break;
            }
        }
    }
    // min-p against the top candidate (pr[0] == 1)
    if (p.min_p > 0.0f)
        for (int i = 1; i < n; ++i)
            if (pr[i] < p.min_p) {
                n = i;
                break;
            }
    // temperature, then one draw
    float q[kMaxTopK], qs = 0.0f;
    for (int i = 0; i < n; ++i) qs += q[i] = expf((cl[i] - cl[0]) / p.temperature);
    const float target = u * qs;
    float cum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cum += q[i];
        if (target < cum) return ci[i];
    }
    return ci[n - 1];
}

}  // namespace flashrt::sample
