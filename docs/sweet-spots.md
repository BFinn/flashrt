# flashrt sweet spots

Last updated 2026-09-30 (through sw112).

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
| Expert-cache swap budget | 64 uploads started per step (`CachePolicyConfig`, engine and `fr_bench`; 8 until sw89, 32 until sw104) | With the seed scale, teacher-forced: window 9 +7.7%, wikitext +4.4%, window 9 with the head +23.8% (sw104) | sw89, sw90, sw104 |
| Expert-cache admission | a missed expert at count 1, and 1.2x the weakest resident's (was 2 and 1.5) | Teacher-forced: window 9 +3.0%, wikitext +1.6% (sw100) | sw99-sw101 |
| Expert-cache seed scale | the policy's counts start at 0.03x the prompt's routing counts; the fill still follows them | The warm-up: at 1x an answer's experts could not beat the residents for ~70 tokens (sw99, sw104) | sw104 |
| Expert-cache tail fill | off (`--cache-tail-weight 0`) | Weight 2 on the prompt's last 16 tokens: window 9 with the head +10%, the agent's first short turns +11-33%, but wikitext -4% and the whole agent session flat (sw111) | sw111 |
| Expert-cache commits | an upload commits at the next step (waiting there), so runs repeat | Query-based commits made the KLD vary 0.00876-0.00920 between runs (sw106); two steps cost 1-2 hit points on short agent turns (sw109) | sw106-sw109 |
| Prefix reuse | 8 host checkpoints (113 MiB each), every 4,096 tokens, and before a fixed tail once one is seen | Window 9: reuse 32,768 / 134,004 as the reference engines; 250K first token 117 → 25.8 s with the head | sw95, sw96 |
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
| greedy, `--spec 2` (sw101, n = 3) | 112.1 | 87.0 | 83.7 | 74.6 |
| greedy, no head (sw96, n = 5, the older admission) | 94.5 | 83.1 | 78.3 | 73.4 |
| temperature 1.0, `--spec 2`, sampled drafts (sw101, n = 3) | 102.3 | 83.8 | 82.2 | 76.9 |

Strata greedy there: 87.0 / 96.0 / 85.0 / 80.4 (its build warns that its cache path changes its
outputs; timings as measured). flashrt trails it at 32K and 250K. Prefix reuse now matches
Strata's (sw96) and decode did not move with it: the gap is the expert cache's warm-up. The cache
is filled from a prompt that predicts the answer's routing poorly, and it reaches 68-70% hits
with the head where the optimum is about 90% (sw99). The swap budget of 32 and the admission
were tuned on this protocol (sw89, sw100).

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
| ~~Adaptive draft length~~ from the head's probability or the last round | **Tried (sw114, simulated on logged rounds): at most ~2% for one rule across contexts, not consistent, below the model's error.** An oracle would gain 7-21%; the head's probability (well calibrated for argmax drafts) and acceptance streaks do not predict enough of it. Sampled drafts may not gate on q(d) at all (it breaks exactness). | A better predictor (target-side signals, q's shape) could revisit it: `fr_bench --round-log` and `simulate.py` are the harness. | |
| **N-gram / prompt-lookup drafts stacked with the MTP head** | Large on repetitive content (code, RAG, long documents), none on fresh prose | Greedy at 245K keeps 3.5 tokens per round because the text repeats earlier context (sw33). | Medium. |
| **Expert-cache warm-up for answers that route unlike the prompt** (done: sw100-sw109, window 9 with the head +8-25%, agent session +12%; left: uploads that cost less DRAM, or a prime that predicts the answer) | Up to the 1.5-9% gap to Strata at 32K-250K on window 9's protocol; per-turn in agent sessions (sw97: 73 tok/s at 44% hits, 125 at 86%) | sw99's simulator (`tools/cache_sim.py --policies engine`) reproduces the engine's hit rate: the loss is the first ~128 tokens, the optimum ~90%. Tail-weighted primes and generation priors trade one text against the other there. | Medium; try in the simulator first |
| ~~The MTP head's pass over the prompt on the chunk path~~ | **Done: the head's KV mirror (sw98) and calls of 1,024 rows with grouped expert GEMMs (sw102): 131K prompt with the head 35.3 → 23.6 s; the head costs 5-7% of prefill at depth (was 27-60%)** | | |
| **Worker count per mode** (6 for one token, 8-11 for windows) | ~1-2% | Plain decode with 6 workers measured best but within noise (sw79). | Small. |
| **The grouped window hit kernels** | Only zero-miss layers of verify rounds | 39 µs at T = 2 against 25 µs at T = 1 for the same 10 experts (`test_moe_hits`). | Small-medium. |
| **GDN prep traffic** (raw Q/K^T per key group; V and T applied in the state step) | ~1-2% of prefill | Prep is DRAM-bound at ~70 MB per 512-token slab (sw71). | Medium. |
| **BF16 xn in the prefill gated mean** | ~1.5% of prefill | The kernel is at bandwidth; this removes 20 of 70 KB per token and mix. | Small; needs a KLD gate. |
| **A custom int8 dense GEMM** for prefill (Q8_0 / IQ4_XS shapes) | up to ~10% of prefill | ggml MMQ at 120-167 TOPS against 490 int8 peak; our moe_q2 reached ~175 with the same scale arithmetic. | Large. |
| **Distilling the MTP head against the quantized target** | Unknown | The head was trained against the full-precision model, not this 2-bit target. The head's MoE is ~2.5B parameters. | Large; needs training data from the target and more than this box. |
| ~~Re-measure the server end to end; compare on the reference protocol~~ | Done: server smoke 11/11 (sw86); window 9's protocol (sw87-sw91) | | |
| **Faster host memory (EXPO)** | Estimated ~10% decode at 245K, a few % at 32K (misses scale with DRAM bandwidth) | STREAM 33.6 GB/s at DDR5-3600. | Owner decision: a BIOS change and reboot. |

## Toggles

The A/B toggles of paths that won or lost were deleted in phase 4 (H-3, sw112): the winning path
of each stays, and the results folders are the record. sw112 checked that the build without them
reproduces every output exactly. Removed: `FLASHRT_GDN_CHUNK` (sw71), `GDN_COL` (sw47),
`ROUTE_WARP` (sw81), `Q3_Q8` (sw69), `MOE_Q2MMA` (sw55), `MOE_YD16` (sw59), `MOE_GU_STAGES` (2,
sw55), `MOE_J` (the widest tile, sw48), `HC_GATE16` (sw59), `HC_DOWN2` / `HC_UP2` (sw75),
`HC_COMB2` (sw78), `LINEAR_MULTI` (sw73), `FUSE_EPI` (sw74, sw82), `DB_SKIP` (sw77), `ATTN_TC` /
`IDX_TC` (sw46, sw49), `MTP_CHUNK` (sw102) and `MTP_MIRROR` (sw98).

What remains:

| Variable | Default | What it switches | Evidence |
|---|---|---|---|
| `FLASHRT_MOE_AB64` | off | per-64 activation scales (faster, fails KLD) | sw57 |
| `FLASHRT_HC_Q8` | down | hc matrices as Q8P (down; up costs KLD) | sw66-sw68 |
| `FLASHRT_ARGMAX_DRAFTS` | off | `1`: argmax drafts in sampled runs (as `--argmax-drafts`) | sw85 |
