# flashrt sweet spots (2026-09-28)

Where tuning stopped and why, the configurations that came out best, and the paths not yet
tested. Numbers come from `bench/results/<folder>` as cited. The rationale for each piece of the
engine is in `docs/engine.md`.

## The best configurations measured

RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600 (EXPO off), Qwen3.8-Flash-Next GSQ Q2_0.

**Decode, one user.**

| Setting | Value | Why | Evidence |
|---|---|---|---|
| Speculation | `--spec 1` or `--spec 2` with sampled drafts (the default at temperature > 0) | Sampled drafts with speculative sampling raised first-draft acceptance from ~51% to ~76% at 32K. With them a second draft is break-even or slightly ahead (32K 146.8 against 143.1, 245K 97.6 against 97.1). | sw85 |
| | `--spec 2`-`3` only for greedy decoding of repetitive text | At depth, 2.7-3.5 tokens per round. | sw30, sw33 |
| Draft head | Q2_0 experts (`--mtp-bits 2`) | Acceptance equal to Q4_0/Q8_0; frees ~500 cache slots (+5%). | sw26, sw65 |
| | LM head trimmed to 32,768 ranked + prompt tokens | Halves the draft step; no measurable acceptance loss. | sw26, sw84 |
| KV | q8 host KV with a GPU hot set of 4,096 blocks at long context | | sw18, sw21 |
| Expert-cache swap budget | 32 uploads per step (engine and `fr_bench`) | +9-12% when the generation routes unlike its prompt (window 9's protocol), -1% when it continues the prompt's text | sw89, sw90 |
| VRAM reserve | 256 MiB in `fr_bench`, 512 MiB in the engine | Every MiB is expert-cache slots: about +0.3% speed per +1% of capacity. The engine keeps more for varied requests (a server at 256 failed its first request before the checkpoints were allocated up front; sw86). | sw64, sw86 |
| CPU miss pool | 8 workers (6 is marginally better for plain decode, within noise) | | p1-moe-cpu, sw79 |
| Kernels | all defaults on (see the toggles below) | | sw73-sw78 |

**With sampled drafts at temperature 1.0 (sw85, sampled text, 6 windows of 128):**

| Arm | tok/s |
|---|---|
| 32K `--spec 1` | 143.1 |
| 32K `--spec 2` | 146.8 |
| 245K `--spec 1` | 97.1 |
| 245K `--spec 2` | 97.6 |

**Measured before sampled drafts (teacher-forced, same tokens every arm, 6 windows of 128):**

| Arm | tok/s |
|---|---|
| 32K plain | ~107 |
| 32K `--spec 1` | ~112.8 |
| 245K `--spec 1` | ~88 |

For comparison, at sw68 the same arms gave 99.6 / 106.4 / 82.9. The P2 gate (≥ 80 at 32K, ≥ 72 at
250K, temperature 1.0) was already met by plain decoding (sw33).

**On the reference engines' protocol** (window 9's prompts, 384 tokens per depth, 3 runs;
`2026-09-29-sw91-depthbench`), decode at 1K / 32K / 134K / 250K:

| Arm | 1K | 32K | 134K | 250K |
|---|---|---|---|---|
| greedy, `--spec 2` | 106.6 | 83.7 | 81.0 | 74.8 |
| temperature 1.0, `--spec 2`, sampled drafts | 82.2 | 77.7 | 81.0 | 76.4 |

Strata greedy there: 87.0 / 96.0 / 85.0 / 80.4. flashrt trails it from 32K on because the expert
cache is warmed from a prompt that predicts the answer's routing poorly (sw88).

**Prefill.** Automatic chunks (the longest that fits the free VRAM, up to 16,384) and q8 KV:
32K 6,216 and 64K 6,239 tok/s (sw83). The last 245K measurement is 5,309 (sw61), before the
chunked GDN and later work.

## Where each track reached its knee

**Decode is bound by host DRAM and verify misses, not the GPU (sw78, sw79).**
- **The CPU-miss window.** In a MoE layer with CPU misses, the GPU's hits and shared expert run
  while the host computes the misses, and `k_moe_combine_db` then waits. Faster kernels there
  only lengthen the wait: the combine fold and moe-hits v2 gained nothing end to end.
- **The miss path itself.** A CPU miss costs 41-46 µs, about 43 GB/s per expert, near the
  host-DRAM ceiling.
- **Launch overhead inside CUDA graphs.**
  - A small kernel still costs ~1.4 µs.
  - Programmatic dependent launch saves only ~0.25 µs per boundary (`bench_pdl`).
  - So fusing kernels paid (sw73-sw75: +4.3% and +2.7%), and "PDL everywhere" would not.
- **What remains outside the window:**
  - dense mat-vecs (at bandwidth except Q3R at ~600 GB/s);
  - the LM head at ~730 GB/s;
  - the hc mix at ~590 GB/s;
  - attention.
  Each is worth a few percent at most.

**Prefill is bound by arithmetic and bandwidth in roughly equal parts (sw80).**

| Part | Share | Limit |
|---|---|---|
| Dense MMQ (ggml) | ~22% | 120-167 TOPS; a new int8 GEMM would be needed to beat it. |
| hc | ~21% | At memory bandwidth; only a BF16 residual would cut it (KLD risk). |
| Routed experts (moe_q2) | ~19% | Gate/up at ~175 TOPS against a ~283 TOPS ceiling. The floor is the per-32 scale arithmetic (7 ALU ops per accumulator element per 64-weight block); per-64 activation scales would lift it but failed KLD (sw57). |
| GDN (chunked, fp16 tensor cores) | ~9% | Prep is DRAM-bound; state is mma-bound and imbalanced. |
| Attention | ~9% | L2-bound gathers; grouping tokens was rejected (sw60). |

**Kernel-level sweet spots found by sweeping.**

| Kernel | Best setting | Beaten alternatives | Evidence |
|---|---|---|---|
| Chunked GDN | chunks of 64, slabs of 8 chunks, state blocks of 32 columns x 8 warps, 3 per SM | slabs of 4 or 16; 64 or 128 columns | sw72 |
| hc decode | 2 rows per warp for down; up as a PDL dependent, one wave | | sw75 |
| hc decode up kernel | weights loaded first for one token, after the preamble for windows | | sw75 |
| BF16 multi-mat-vec | a block of 4 warps per row | a warp per row lost to two MMVF launches | sw73 |
| Prefill routing | a warp per token | 26x faster than the block kernel | sw81 |
| Prefill chunk length | the longest that fits (16K at 32K-64K) | tile fill of the expert GEMM grows with it | sw52-sw54 |

## Untested paths

Ranked by expected value. None of these has been implemented or measured end to end.

| Path | Expected gain | Evidence so far | Effort, risk |
|---|---|---|---|
| ~~Sampled drafts with speculative sampling~~ | **Done (sw85): +20% at 32K (119.0 → 143.1 tok/s), +6% at 245K (91.7 → 97.1)**, `--spec 1`, temperature 1.0 | | |
| **Adaptive draft length** from the head's calibrated q and the round's expected new experts | Unknown; depends on the above | Gating on the argmax head's probability did not beat a fixed K (sw29); a sampled q is better calibrated. The MoE literature reports verify cost 2.4x → ~1.5x (vault: "Speculative Decoding with MoE"). | Medium. |
| **N-gram / prompt-lookup drafts stacked with the MTP head** | Large on repetitive content (code, RAG, long documents), none on fresh prose | Greedy at 245K keeps 3.5 tokens per round because the text repeats earlier context (sw33). | Medium. |
| **Expert-cache warm-up for answers that route unlike the prompt** (an adaptive swap budget, larger while the hit rate is low; the routing of the prompt's last part, such as an instruction, weighted up) | Up to the 5-13% gap to Strata at 32K-250K on window 9's protocol | The hit rate there is 66% against 93% on wikitext (sw88); swap budget 32 recovered 9-12% (sw89) | Small-medium; needs both kinds of prompt measured |
| **The MTP head's pass over the prompt on the chunk path** | Prefill with the head 2,150 → toward 5,700 tok/s at 250K | The head runs over every prompt token on a slower path (sw87, sw91) | Medium |
| **Worker count per mode** (6 for one token, 8-11 for windows) | ~1-2% | Plain decode with 6 workers measured best but within noise (sw79). | Small. |
| **The grouped window hit kernels** | Only zero-miss layers of verify rounds | 39 µs at T = 2 against 25 µs at T = 1 for the same 10 experts (`test_moe_hits`). | Small-medium. |
| **GDN prep traffic** (raw Q/K^T per key group; V and T applied in the state step) | ~1-2% of prefill | Prep is DRAM-bound at ~70 MB per 512-token slab (sw71). | Medium. |
| **BF16 xn in the prefill gated mean** | ~1.5% of prefill | The kernel is at bandwidth; this removes 20 of 70 KB per token and mix. | Small; needs a KLD gate. |
| **A custom int8 dense GEMM** for prefill (Q8_0 / IQ4_XS shapes) | up to ~10% of prefill | ggml MMQ at 120-167 TOPS against 490 int8 peak; our moe_q2 reached ~175 with the same scale arithmetic. | Large. |
| **Distilling the MTP head against the quantized target** | Unknown | The head was trained against the full-precision model, not this 2-bit target. The head's MoE is ~2.5B parameters. | Large; needs training data from the target and more than this box. |
| ~~Re-measure the server end to end; compare on the reference protocol~~ | Done: server smoke 11/11 (sw86); window 9's protocol (sw87-sw91) | | |
| **Faster host memory (EXPO)** | Estimated ~10% decode at 245K, a few % at 32K (misses scale with DRAM bandwidth) | STREAM 33.6 GB/s at DDR5-3600. | Owner decision: a BIOS change and reboot. |

## Toggles

All default to the tuned setting. Setting one to `0` restores the older path for A/B runs.

| Variable | Default | What it switches | Evidence |
|---|---|---|---|
| `FLASHRT_GDN_CHUNK` | on | chunked GDN on fp16 tensor cores (prefill, 64+ tokens) | sw71 |
| `FLASHRT_GDN_COL` | on | the column kernel for 16-63 tokens | sw47 |
| `FLASHRT_ROUTE_WARP` | on | prefill routing a warp per token | sw81 |
| `FLASHRT_Q3_Q8` | on | Q3_K multiplied as Q8_0 in prefill | sw69 |
| `FLASHRT_MOE_Q2MMA` | on | own int8 expert kernels on the planar arena | sw55 |
| `FLASHRT_MOE_YD16` | on | BF16 per-slot expert outputs | sw59 |
| `FLASHRT_MOE_AB64` | off | per-64 activation scales (faster, fails KLD) | sw57 |
| `FLASHRT_MOE_GU_STAGES` | 2 | gate/up pipeline stages | sw55 |
| `FLASHRT_MOE_J` | 0 (the widest) | ggml MoE token-tile width (MMQ path) | sw51, sw52 |
| `FLASHRT_HC_GATE16` | on | BF16 hc gate in prefill | sw59 |
| `FLASHRT_HC_Q8` | down | hc matrices as Q8P (down; up costs KLD) | sw66-sw68 |
| `FLASHRT_HC_DOWN2` / `FLASHRT_HC_UP2` | on | hc decode kernels v2 | sw75 |
| `FLASHRT_HC_COMB2` | on | layer-end combine folded into the hc decode kernels | sw78 |
| `FLASHRT_LINEAR_MULTI` | on | BF16 mat-vecs of one input in one launch (decode) | sw73 |
| `FLASHRT_FUSE_EPI` | on | decode/prefill epilogue fusions and shared input conversions | sw74, sw82 |
| `FLASHRT_DB_SKIP` | on | tokens without CPU misses skip the doorbell round trip | sw77 |
| `FLASHRT_ATTN_TC` / `FLASHRT_IDX_TC` | on | tensor-core attention and indexer scores in prefill | sw46, sw49 |
