# Research: a flashrt-native dynamic quant for Flash-Next (2026-09-27)

A parallel research track to the main workstream. Question: can the GSQ-RCO Q2_0 build
become a "dynamic" quant in the style of Unsloth's UD quants, and shrink further so more
of the model lives in VRAM? Nothing here was run on the box. The labels are the ones from
`docs/background.md`:

- **[M]** = measured on the box.
- **[Q]** = quoted from a source.
- **[E]** = estimate.

## Short answer

1. **GSQ-RCO is already a dynamic quant, and a stronger one than UD.** RCO picks a type
   for each of 352 tensors by gradient search on end-to-end KL, under an exact size budget
   [Q]. UD picks types with heuristics plus KLD checks. Running a UD-style recipe over this
   model would re-derive a worse version of what RCO already did.
2. **The headroom GGUF leaves is per-expert precision.** GGUF allows one type per tensor, so
   all 512 experts of a layer share a type. Every routed-expert tensor in the Q2_0 build is
   Q2_0 [Q]. flashrt already repacks experts into its own per-expert blobs, so it can give
   each expert its own format. llama.cpp cannot. This is the one real opening, and it
   needs our own format.
3. **Smaller experts pay twice in this engine:** more cache slots, and fewer bytes per miss.
   On the measured hit-rate curve, a 1.75 bpw average expert cuts miss bytes per token to
   about 53% of today's, with no change to the cache's VRAM [E, table below].
4. **Quality is the risk, and it is steep in this range.**
   - Going from 2.40 to 3.00 bpw (Q2_0 build to IQ3_XXS build) buys +3.5 task-average
     points [Q].
   - Going down from 2.25 bpw on the experts will probably cost points of the same order
     if done uniformly [E].
   - The bet is that allocating bits by routing mass and sensitivity recovers most of that
     loss.
   - Every step is KLD-gated.

## Where the bytes are (shard 1, from the GGUF header) [Q]

Parsed from the published header (`general.file_type` Q2_0, 1223 tensors, 37.61 GB):

| Class | GB | Params | bpw | Where it lives in flashrt |
|---|---:|---:|---:|---|
| `ffn_{gate,up,down}_exps` | 33.98 | 120.8B | 2.25 | host arena + VRAM cache |
| `hc_{attn,ffn}_{up,down}` (hyper-connections) | 1.26 | 0.63B | **16 (BF16)** | VRAM, read every window |
| `attn_qkv`, `attn_gate`, `attn_q/k/v/output` | 1.01 | | 3.5-5.5 | VRAM |
| `output` (head) | 0.44 | 0.64B | 5.5 (Q5_K) | VRAM |
| `ssm_out` | 0.32 | | 4.5 | VRAM |
| `token_embd` | 0.27 | 0.64B | 3.4 (Q3_K) | VRAM today; a lookup |
| `ffn_gate_inp` (routers) | 0.13 | | 16 (BF16) | VRAM |
| shared experts | 0.11 | | 3.3-4.1 | VRAM |

Shard 2 is the n-gram (PLE) table: 28.8 GB, IQ4_NL, excluded from ISTA's search [Q].

- **Expert geometry:** 640 × 2560 for gate and up, and 2560 × 640 for down. That is
  4,915,200 weights, or 1.382 MB per expert at Q2_0.
- **`ffn_down_exps` rows are 640 wide.** 640 is not divisible by 256, which rules out every
  K-quant, IQ-quant and TQ quant. Only 32- or 64-block formats fit: Q2_0, IQ4_NL, Q4_0,
  Q8_0 [Q]. So any sub-2-bit expert format for this model has to be a new 64-block type.

## What ISTA already found (model cards) [Q]

| Build | Transformer bpw | Experts | Task avg (AIME25 / GPQA-D / LCB v6) |
|---|---:|---|---:|
| BF16 | 16 | | 93.12 (100 / 91.92 / 87.43) |
| IQ3_S | 3.50 | gate/up IQ2_S-IQ3_S, down mostly IQ4_NL | 93.26 |
| IQ3_XXS | 3.00 | gate/up IQ2_XXS-IQ3_S, down 18 layers IQ4_NL | 92.57 |
| IQ2_XS | 2.50 | gate/up: 34 IQ2_S, 11 IQ2_XXS, 3 IQ1_M layers; down Q2_0 | 89.16 |
| **Q2_0 (ours)** | 2.40 | all Q2_0 | 89.07 (96.67 / 89.39 / 81.14) |
| Coder: 50% of experts pruned by RCO, rest 3.5 bpw | 1.89 effective | 256 per layer | LCB v6 86.28, SWE-V 75.6 vs 82.8 |

- **The IQ lookup-table formats cost real speed.** Q2_0 has 3.4× the prefill and 1.33× the
  decode of IQ2_XS in llama.cpp [Q]. flashrt should stay with integer-grid formats that
  keep the VNNI int8 dot and exact-int8 prefill GEMM.
- **RCO already lowered 3 layers of gate/up to IQ1_M** (1.75 bpw) in the IQ2_XS build.
  Some layers tolerate sub-2-bit experts [Q].
- **The Coder build is the strongest data point for this track.**
  - 256 experts at 3.5 bpw is about the byte budget of 512 experts at 1.75 bpw.
  - On code it beats our full Q2_0 build: LCB v6 86.28 against 81.14.
  - General capability degrades by design [Q].
  - So on this model, "fewer bits on unimportant experts, more on important ones" beats
    uniform 2-bit. flashrt does not need GGUF's equal-experts-per-layer constraint.
- **The metadata records an imatrix:** `quantize.imatrix.*`, 1000 chunks,
  `calib_qwen3.8-flash-next.tokids`. The file itself is not published.

## Why smaller experts pay twice here [E]

Hit rate is interpolated from `bench/results/2026-09-27-p0/cache_sim.txt` (wikitext at 32K,
LRU). The cache's VRAM is held at today's 7.60 GB (5,500 × 1.382 MB). The average bpw is
uniform over experts.

| Expert bpw | Blob MB | Slots | Hit | Miss MB/token | vs now | Prefill bytes per chunk |
|---:|---:|---:|---:|---:|---:|---:|
| 2.25 (now) | 1.382 | 5,500 | 0.865 | 89.6 | 1.00 | 1.00 |
| 2.125 (Q2_0 with fp8 scales) | 1.306 | 5,824 | 0.875 | 78.4 | 0.87 | 0.94 |
| 2.00 | 1.229 | 6,188 | 0.886 | 67.5 | 0.75 | 0.89 |
| 1.75 | 1.075 | 7,071 | 0.908 | 47.2 | 0.53 | 0.78 |
| 1.50 | 0.922 | 8,250 | 0.931 | 30.4 | 0.34 | 0.67 |

- **Mixed allocation should do better than uniform.**
  - Hot experts, which are mostly cache-resident, keep more bits.
  - Cold experts, which make up most misses, get fewer bits.
  - The misses then shrink more than the average bpw suggests.
- **Local skew is large, global skew is moderate.**
  - In our own traces the hottest 20% of experts take 67-86% of routes
    (`trace_stats.txt`) [M]. Those are 2,048-token windows, so this is within-context skew,
    which the LRU cache already exploits.
  - Over a diverse corpus, the top 20% hold only about 50% of routing mass (range 36-56% by
    layer). That was computed from Unsloth's imatrix counts for Qwen3-Next-80B: 512
    experts, top-10, 762K tokens [Q, computed].
  - Consequence [E]: putting the cold 80% at about 1.5 bpw puts about half of routed
    compute on the low tier, and cuts miss bytes by about a third, not more.
- **Decode gain is bounded.**
  - In Strata's breakdown the CPU pool is 5.2 ms of a 19.6 ms round, against 10.9 ms of GPU
    wait [M].
  - Halving miss bytes shortens the CPU leg, but the dense GPU path stays the critical path.
- **Prefill barely moves (correction).** An earlier version of this note called prefill
  transfer-bound. At the measured 45-55 GB/s host-to-device rate, Strata's 32K prefill
  moved about 298 GB of expert blobs (215,293 × 1.38 MB) in about 6 s of its 47 s. The
  rest is PLE stalls (21.6 s) and GPU compute. Smaller experts save at most about 1-1.5 s
  per 32K prompt, and nothing if the DMA is overlapped [E].

## Speed estimates in flashrt [E]

- **The anchor** is Strata's measured 32K round (greedy, MTP, `2026-09-27-p0c`): 1.66 tokens
  per round in 19.6 ms. That is:
  - 10.9 ms of GPU wait, split by estimate as dense ≈ 8.6 and expert hits ≈ 2.3;
  - 5.2 ms in the CPU miss pool, and 3.5 ms of other work;
  - at 7,002 slots and 82.8% hits.
- **Scaling rules:**
  - the dense term scales with dense bytes read;
  - the hit term with hit rate × blob size;
  - the CPU term with miss rate × blob size.
- **Miss rate vs slots** follows the wikitext 32K LRU simulation, scaled to Strata's
  measured 17.2% at 7,002 slots. Beyond 9,000 slots it is extrapolated, as miss ∝ slots⁻².
- **Held constant:** tokens per round. A quality drop that lowers draft acceptance would eat
  into these numbers.

| Option | Slots | Share of experts | Hit | Round | Greedy tok/s at 32K | vs now |
|---|---:|---:|---:|---:|---:|---:|
| Current Q2_0 (Strata-shaped anchor) | 7,002 | 28% | 82.8% | 19.6 ms | 85 | 1.00 |
| Q2_0 with fp8 scales, plus dense fixes | 8,070 | 33% | 87% | 16.6 ms | 100 | 1.18 |
| Mix at ~1.95, plus dense fixes | 8,800 | 36% | 89% | 15.6 ms | 106 | 1.26 |
| Cold tier at ~1.75, plus dense fixes | 9,800 | 40% | 91% | 14.6 ms | 114 | 1.34 |
| 1.5 (quality risk) | 11,440 | 47% | 94% | 13.6 ms | 122 | 1.44 |

- **Roughly half of the second row's gain is the dense fixes.** Hyper-connection reads are
  on the GPU critical path.
- **Apply the ratio to whatever flashrt reaches.** It is not additive to the phase gates.
  With the ~1.95 mix, the P2 gate (≥80 at 32K, t=1.0) would become about 100, and P4's ≥95
  about 120. The background physics ceiling is about 140.
- **Context:** all rows keep the native 262,144 with KV streaming (32K cells resident, about
  0.4 GiB).
  - Without streaming, 262K of KV costs 3.19 GiB of VRAM (13,056 B per cell): about 2,480
    Q2_0 slots, or 3,180 at 1.75 bpw.
  - The host arena shrinks from 34 GB to 26-29 GB, which leaves room for the host-side KV
    (3.4 GB at 262K).

## What Unsloth's UD recipe actually is [Q]

- **The method.** Types are picked per tensor from heuristics plus KLD checks, with a large
  hand-curated imatrix. They publish the imatrix, not the search. The recipe can be read
  from GGUF headers and reproduced with `llama-quantize --tensor-type`.
- **The closest analogue is Qwen3-Next-80B**, parsed from Unsloth's headers.
  - gate/up get the low type; down sits 1-2 steps higher.
  - The first ~5 and last ~8 layers are bumped.
  - The shared expert is kept at 4-8 bit, and routers at F32/BF16.
- **The names overstate the compression.** Average bpw on the experts:

  | File | gate/up | down | All experts |
  |---|---:|---:|---:|
  | UD-TQ1_0 | 1.69 | 2.57 | 1.98 |
  | UD-IQ1_S | 1.72 | 3.27 | 2.23 |
  | UD-IQ1_M | 1.85 | 3.43 | 2.38 |

  So "UD-IQ1_S" carries about the same expert bytes as our Q2_0 build. Beating Q2_0 on size
  means gate/up near 1.75 bpw with down at about 2.1-2.6.

## Quality evidence for going below 2.25 bpw on experts [Q unless marked]

- **Q-Strata** (arXiv 2608.30564), Qwen3-30B-A3B, wikitext perplexity, effective bpw:
  - uniform GPTQ: 12.05 at 2.25 bpw and 14.97 at 2.0;
  - allocated: 8.56 at 2.25, 8.97 at 2.0 and 10.23 at 1.75.
  - Allocation at 1.75-2.0 bpw puts 25-59% of expert linears at 1 bit.
- **GEMQ** (arXiv 2605.23078), same model, FP16 8.71:
  - at 1.5 bpw per expert, uniform gives 21.1 and gradient-based global allocation 12.2.
  - Router fine-tuning is a large part of the gain: at 1.5 bit, over 40% of tokens change
    experts.
- **AlphaQ** (arXiv 2606.04980): allocating per projection beats allocating per expert.
  Allocation driven by calibration data overfits the calibration domain.
- **Large MoE, ik_llama trellis quants** (ubergarm's Kimi-K2 cards):
  - IQ2_KS mix at 2.43 bpw: PPL 3.68;
  - gate/up IQ1_KT with down IQ2_KT, 1.96 bpw: 3.97 (+8%).
- **Best estimate [E]:**
  - 2.25 → ~1.8 average expert bpw with sensitivity allocation: roughly +8-15% perplexity
    on a large MoE.
  - Uniform allocation at 1.5 bpw is 2-4× worse than allocated.
  - On Flash-Next, 2.40 → 3.00 bpw was worth +3.5 task points, so expect a visible drop on
    LCB/GPQA before the KLD gate allows anything.

## Format candidates for the CPU miss path

| Format | bpw | Fits `ffn_down` (640 wide)? | CPU decode cost |
|---|---:|---|---|
| Q2_0 (now) | 2.25 | yes (64-block) | trivial unpack, DRAM-bound at 6 cores [M] |
| Q2_0 with fp8 scales (new) | 2.125 | yes | same |
| Ternary, 64-block, 5 trits per byte (new) | 1.75-1.875 | yes | base-3 unpack, cheap in AVX-512 [E] |
| IQ2_XXS / IQ1_M / IQ1_S | 2.06 / 1.75 / 1.56 | **no** (256-block) | grid lookup and gather; 3.4× slower prefill in llama.cpp [Q] |
| IQ1_KT / IQ2_KT (trellis) | 1.75 / 2.125 | no | 6-8 SIMD ops per 8 weights to generate values; probably compute-bound at 6 cores, might reach the DRAM limit at 12 [E from code] |

Because of the 640-wide rows, down must be a 64-block format anyway. gate/up (2560 wide)
could use 256-block codebook formats, but that means a second kernel family. Prefer the
integer-grid route first.

## Cheap wins before any new expert format [E unless marked]

These need no BF16 weights and no GPU search:

1. **`token_embd` to host RAM:** −0.27 GB of VRAM. It is a row lookup per token, and the
   quality cost is zero.
2. **Hyper-connections from BF16 to Q8_0:** −0.59 GB of VRAM, and −0.59 GB of dense reads
   per window.
   - Those reads are on the GPU critical path: about 16% of dense bytes.
   - RCO treated these as fixed, not searched [Q, allocation file header].
   - Needs a KLD check.
3. **Q2_0 with 8-bit block scales** (fp8 E4M3 relative to a per-row fp16 scale): 2.125 bpw,
   −5.6% on every expert.
   - Codes stay unchanged, so it can be derived from the existing file.
   - It adds about 1% RMS weight error on top of the 2-bit quantization noise.
   - Needs a KLD check.

Items 1 and 2 together are about 0.86 GB, or about 620 more Q2_0 slots (+1.5-2.5 hit points
by the background estimate).

## Building our own quant: the path

- **Formats.** Keep the count at 2-3, each with a CPU AVX-512 kernel plus a CUDA GEMV and
  grouped GEMM. All use 64-weight blocks and integer grids, exact in int8.
  - `Q2_0` as today, or with fp8 scales.
  - A 64-block ternary `{-1,0,+1}·d`: 5 trits per byte, 13 bytes plus scale. That is 1.875
    bpw with an fp16 scale, or 1.75 with an fp8 scale.
  - Optionally a 3-bit `{-4..3}·d` for the hottest experts, about 3.1-3.25 bpw.
- **Allocation**, in two stages.
  1. Per (layer, projection) first. This is the safer step, per AlphaQ and the UD pattern:
     gate/up low, down higher, edge layers higher.
  2. Then a per-expert cold tier by global routing mass. Minimise Σ routing-mass × weighted
     error under a byte budget. This is a multiple-choice knapsack, solved greedily with a
     Lagrangian.
  - RCO's code supports per-expert groups for a later end-to-end-KL refinement [Q].
  - Pruning, as zero bits with the router masked, is the same knapsack with one more choice.
  - Router drift: GEMQ gets much of its gain from router fine-tuning. Measure the top-10
    overlap alongside KLD.
- **Source weights.** Downward moves can come from the existing Q2_0, but that compounds
  error. Anything better needs BF16.
  - Qwen/Qwen3.8-Flash-Next: 360 GB, 131 safetensors shards [Q].
  - Stream shard by shard. The box has about 149 GB free; this Mac has 403 GB free.
  - Weighted-MSE quantization of one expert is cheap on a CPU. Full GSQ is not: it is
    8+ H200-class GPUs for a day or more, and its public code has no GGUF or Q2_0 writer
    [Q/E].
- **Calibration.**
  - Per-expert routing mass over a diverse corpus, from `tools/route_trace`.
  - A per-expert imatrix: diagonal input second moments. llama.cpp's `llama-imatrix`
    already keeps per-expert entries for `_exps`.
  - Both need one box window with the current Q2_0 model. Heed the RAM rules.
- **Validation without new kernels: a "carrier" GGUF.**
  - Q2_0 and ternary values are exact in Q4_0 with the same scale, since Q4_0 covers
    `{-8..7}·d`.
  - Writing the fake-quantized mixed experts into Q4_0 tensors (about 68 GB) gives stock
    llama.cpp an exact functional twin of the flashrt format for KLD.
  - Reference logits come from IQ3_S (task average 93.26, at parity with BF16) [Q].

## Proposed sequence (each step KLD-gated; none started)

| Step | Needs | Expected gain [E] |
|---|---|---|
| 0. `token_embd` to host, HC to Q8_0 | existing file; flashrt loader | about 0.86 GB of VRAM, about 620 slots; −0.59 GB of dense reads per window |
| 1. Q2_0 with fp8 scales | existing file; a scale re-encode | −5.6% expert bytes |
| 2. Calibration window on the box: global routing mass, per-expert imatrix, IQ3_S reference logits | one GPU window, after the main track allows | the inputs for everything below |
| 3. Imatrix-aware 64-block ternary plus Q2_0 requant from BF16 experts, allocated per (layer, projection) | BF16 streamed shard by shard; CPU only | a 1.9-2.0 bpw expert average; misses −25-35% |
| 4. Per-expert cold tier plus a 3-bit hot tier | step 2's routing mass | the same bytes at better KLD, or fewer bytes at the same KLD |
| 5. Optional RCO end-to-end refinement or pruning | a rented multi-GPU box | Coder-style targeted builds |

Validation for steps 3-4 before any kernel work: the Q4_0 carrier GGUF in llama.cpp.

## Open questions

- **Global routing skew over a diverse corpus.** This decides how much frequency-aware
  allocation beats uniform.
- **How much ternary costs per matrix class** (gate/up vs down) at 64-weight blocks,
  measured as KLD per GB saved.
- **Whether frequency-based allocation hurts rare domains,** such as vision or multilingual.
  The Coder card's vision lesson says it will unless the calibration mix covers them [Q].
- **License.** The base model is Qwen Community License 1.0, not Apache-2.0 as the GGUF card
  says. It matters only if a derived quant is published.

## Sources

- Model cards and allocation files:
  - https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  - https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
- GSQ: arXiv 2604.18556, https://github.com/IST-DASLab/GSQ
- RCO: arXiv 2605.00649, https://github.com/IST-DASLab/RCO
- Q2_0 in ggml: llama.cpp PR #24448. `quantize_q2_0` ignores the imatrix and never emits
  code 3 (`ggml/src/ggml-quants.c:74-111, 2113-2124` in the local clone).
- Base model: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
