// SPDX-License-Identifier: Apache-2.0
// fr_inspect: parse a qwen4exp GGUF into the spec and weight plan, and estimate the VRAM
// budget: how many expert slots remain after the dense weights, KV and state.
//
//   fr_inspect MODEL.gguf [--vram-mib 15840] [--overhead-mib 1024] [--mtp-mib 834] [--kv-resident 32768]
//
// --overhead-mib covers the CUDA context, graph and activation buffers (an estimate until
// the engine measures it). With --kv-resident N, QSA layers keep N cells in VRAM and the rest
// of the KV in pinned host RAM (KV streaming); 0 keeps the whole context in VRAM.
#include "arch/qwen4exp/spec.hpp"
#include "core/gguf.hpp"
#include "quant/q2_0/q2_0.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace flashrt;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fr_inspect MODEL.gguf [--vram-mib N] [--overhead-mib N] [--mtp-mib N] [--kv-resident N]\n");
        return 2;
    }
    double vram_mib = 15840, overhead_mib = 1024, mtp_mib = 834;
    long kv_resident = 32768;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--vram-mib")) vram_mib = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--overhead-mib")) overhead_mib = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--mtp-mib")) mtp_mib = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--kv-resident")) kv_resident = std::atol(argv[i + 1]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    const Gguf g = Gguf::open(argv[1]);
    const qwen4exp::Spec s = qwen4exp::parse(g);
    const qwen4exp::WeightPlan p = qwen4exp::plan(g, s);
    const double MiB = 1048576.0;

    std::printf("qwen4exp: %d layers (%zu QSA, %zu GDN), d_model %d, vocab %d, context %d\n", s.n_layer,
                s.qsa_layers.size(), s.gdn_layers.size(), s.d_model, s.n_vocab, s.max_context);
    std::printf("  QSA: %d heads, %d kv heads, head dim %d/%d, block %d cells, indexer %d heads x %d, top-%d cells\n",
                s.n_head, s.n_head_kv, s.head_dim_k, s.head_dim_v, s.qsa_block, s.idx_heads, s.idx_dim, s.idx_top_k);
    std::printf("  GDN: conv %d, %d groups, inner %d, state %d, %d value heads\n", s.ssm_conv, s.ssm_groups, s.ssm_inner,
                s.ssm_state, s.ssm_heads);
    std::printf("  MoE: %d experts, top-%d, expert ff %d, shared ff %d\n", s.n_expert, s.top_k, s.d_ff_expert, s.d_ff_shared);
    std::printf("  hyper-connections: %d streams, rank %d; PLE layers %zu, %d-gram, %d heads x %d dims, %lld rows\n",
                s.hc_count, s.hc_rank, s.ple_layers.size(), s.ple_ngram, s.ple_heads, s.ple_dim, (long long) s.ple_rows);

    std::printf("\nweight plan\n");
    for (int t = 0; t < 3; ++t)
        std::printf("  %-13s %9.1f MiB\n", qwen4exp::tier_name(qwen4exp::Tier(t)), p.bytes[t] / MiB);
    const q2_0::ExpertShape es{s.d_model, s.d_ff_expert};
    std::printf("  one expert: %llu bytes (%s the Q2_0 repacked blob, %zu); %d experts in all\n",
                (unsigned long long) p.expert_bytes, p.expert_bytes == q2_0::expert_bytes(es) ? "matches" : "DIFFERS FROM",
                q2_0::expert_bytes(es), p.n_experts_total);

    const double kv_cell = double(qwen4exp::qsa_kv_bytes_per_cell(s));
    const double gdn = double(qwen4exp::gdn_state_bytes(s));
    std::printf("\nVRAM budget (estimate): %.0f MiB total, %.0f overhead, %.0f MTP head, %.1f dense, %.1f GDN state, "
                "%.0f B of QSA KV per cell\n", vram_mib, overhead_mib, mtp_mib, p.bytes[0] / MiB, gdn / MiB, kv_cell);
    std::printf("  %9s  %14s  %9s  %12s  %9s\n", "context", "KV in VRAM", "KV MiB", "expert slots", "of all");
    for (long ctx : {32768L, 131072L, 262144L}) {
        for (long res : {0L, kv_resident}) {
            if (res && res >= ctx) continue;
            const long cells = res ? res : ctx;
            const double kv = cells * kv_cell / MiB;
            const double free_mib = vram_mib - overhead_mib - mtp_mib - p.bytes[0] / MiB - gdn / MiB - kv;
            const long slots = std::max(0L, long(free_mib * MiB / double(p.expert_bytes)));
            std::printf("  %9ld  %14s  %9.0f  %12ld  %8.1f%%\n", ctx, res ? ("streamed, " + std::to_string(res)).c_str() : "all",
                        kv, slots, 100.0 * slots / p.n_experts_total);
        }
    }
    return 0;
}
