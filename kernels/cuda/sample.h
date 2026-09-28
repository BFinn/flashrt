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

// The kept candidates' probabilities under the chain above (top-p, min-p, temperature), for
// cl sorted descending: returns n and prob[0..n) (summing to 1); the distribution choose() draws
// from.
__host__ __device__ inline int chain_probs(const float* cl, int k, const Params& p, float* prob) {
    if (p.temperature <= 0.0f || k == 1) {
        prob[0] = 1.0f;
        return 1;
    }
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
    if (p.min_p > 0.0f)
        for (int i = 1; i < n; ++i)
            if (pr[i] < p.min_p) {
                n = i;
                break;
            }
    float qs = 0.0f;
    for (int i = 0; i < n; ++i) qs += prob[i] = expf((cl[i] - cl[0]) / p.temperature);
    for (int i = 0; i < n; ++i) prob[i] /= qs;
    return n;
}

// ---- Sampled drafts with speculative sampling (Leviathan et al. 2023; sw85)
// The drafter's distribution q is its logits through the same chain as the target's. A draft is
// drawn from q with a draw of its own (kDraftSalt); the verifier accepts it with probability
// min(1, p(d) / q(d)) (a third draw, kAccSalt), and on a rejection samples from max(0, p - q),
// normalised, with the plain sampler's draw. The output distribution equals plain sampling's,
// but tokens no longer match plain sampling one for one.
constexpr uint64_t kDraftSalt = 0x5DEECE66DA3F29C1ull, kAccSalt = 0xA24BAED4963EE407ull;

// What draft_row reads from device memory, so a captured draft graph serves any request
struct DraftCfg {
    Params p;
    uint64_t seed;
};

// One row of drafter logits (V entries; ids maps entry -> token id, or nullptr): samples the draft
// for position dp[1] + 1 into v[1] (as a token id) and stores the step's q, the kept candidates
// (token ids, probabilities, count) at q_ids / q_p [dp[3]][kMaxTopK] and q_n[dp[3]]. dp is the draft
// chain's parameter block [token, position, -, step] (device).
void draft_row(const float* logits, int V, const DraftCfg* cfg_dev, const int32_t* dp, const int32_t* ids, int32_t* v, int32_t* q_ids,
               float* q_p, int32_t* q_n, cudaStream_t stream);

// The verify rows of a window: rows - 1 drafts (drafts[j], with q at q_ids / q_p / q_n [j]) and a
// last row. out_dev[j] (j < rows) is the token for position pos0 + j: the draft if accepted, else
// the residual sample; the last row's is a plain sample. out_dev[rows + j] (j < rows - 1) is 1 if
// draft j was accepted. The caller keeps drafts up to the first rejection and emits its token (or
// the last row's if none was rejected).
void spec_verify(const float* logits, int rows, int n_vocab, const Params& p, uint64_t seed, int64_t pos0, const int32_t* drafts,
                 const int32_t* q_ids, const float* q_p, const int32_t* q_n, int32_t* out_dev, cudaStream_t stream);

}  // namespace flashrt::sample
