# flashrt engine as built

Last updated 2026-09-29 (through sw91).

This describes what the code does today, why each piece is shaped the way it is, what was tried
and rejected, and how to measure it. `design.md` holds the plan and the phase gates. The
per-experiment evidence is in `bench/results/2026-09-2*`, each folder with a README.

**Status:**
- **P1 is complete.** The gates are met: 32K 94.9 / 96.2 / 99.2 tok/s, 250K 61-64 tok/s with
  fp16 KV, fast-path KLD 0.0087.
- **P4's KV work is done:** q8 KV and host-resident KV with a GPU hot set. With them, 245K runs at
  66-80 tok/s over 6 windows.
- **P2 is complete (2026-09-28).** At temperature 1.0 with one MTP draft per round: 32K 108-120
  tok/s, 245K 82-101 tok/s over 6 windows (gates ≥ 80 / ≥ 72). The distribution test and the
  window KLD gate pass.
- **The engine process serves the fast path** (`flashrt-engine`, JSON lines): prefix reuse,
  checkpoints, sampling, speculation, cancellation.
- **The Rust server works end to end (2026-09-28):** OpenAI and Anthropic APIs with streaming,
  reasoning, tool calls and stop strings over the engine (`bench/results/2026-09-28-sw35-server`).
- **P3 is complete (2026-09-28):** prefill in chunks with the experts streamed to the GPU, 2,135-
  2,272 tok/s at 32K and 1,940-1,959 at 245K (was 109-123 on the CPU path).
- **Prefill kernels (2026-09-28):**
  - tensor-core attention, a GDN column kernel, tensor-core indexer scores;
  - BF16 activations written by the hyper-connection norm, flashrt's own expert grouping;
  - the arena registered at load.

  - the chunk length chosen from free VRAM;
  - the routed experts on flashrt's own int8 kernels, straight from the planar arena layout
    (`kernels/cuda/moe_q2.cu`);
  - less hyper-connection traffic, and BF16 expert outputs and hc gate.

  Prefill runs **5,580 tok/s at 32K, 5,629 at 64K and 5,170-5,309 at 245K** (sw61); KLD
  0.0084-0.0087. Through the engine, a 32K prompt takes 7.9 s.
- **Decode round (2026-09-28, sw63-sw68):**
  - A verify window's second token costs about 3.6 ms of kernels, mostly its own experts (sw63).
  - Forecasting routing to prefetch misses does not pay: uploads compete with the CPU misses for
    host DRAM (sw63).
  - Capacity is the lever: the VRAM reserve is now 256 MiB, the head's experts Q2_0, and the hc
    down matrices Q8P. Each is measured teacher-forced (`fr_bench --teacher`) at +2-5%, at no
    KLD cost.
- **Next:** see "Next steps".

## One decode token (fast path)

`ForwardRef::forward(seq, T = 1, ...)` in `arch/qwen4exp/forward_ref.cu`:

1. **The host posts the token.**
   - `doorbell_begin_token` bumps the doorbell sequence number and wakes the miss server.
   - The decode parameters (token, position, seq) go into a pinned 12-byte block.
   - A helper thread starts reading the token's PLE n-gram rows from the SSD (`ple_fetch`,
     RowReader, O_DIRECT).
2. **Graph A** (`cudaGraphLaunch`):
   - it copies the parameter block to the device;
   - it gathers the embedding row from the host-resident Q3_K table by the device-side token id
     (`k_embed_q3k`);
   - it initialises the 4 hyper-connection streams;
   - it runs layer 0.
3. **The host waits for the PLE rows.** The read (about 0.4 ms) overlaps layer 0.
4. **Graph B:** the PLE upload and dequantize, layers 1-47, the output hyper-connection mix, and the
   head (Q5_K).
5. **After graph B, the cache manager learns from the previous token's routing** and enqueues
   table updates and expert uploads (see below).
6. **The host synchronises the stream,** checks the doorbell mailboxes for errors, and runs a GPU
   argmax (4 bytes come back).

Any `forward()` that is not graph-eligible (prefill batches, the reference path) drops the
graphs, and the next decode token captures them again (about 0.1 ms).

### One layer

- **Hyper-connection mix (4 streams, rank 320), two fused kernels (v2, sw75, sw78):**
  - `k_hc_down2`:
    - the previous block's combine (x + out * 2 sigmoid(inject / 4));
    - the RMS norm, with 1/rms applied after the dot products;
    - the Q8P down-projection and BF16 inject projection, 2 rows per warp, weights loaded
      before the norm; per-stream partial sums.
  - `k_hc_up_mix2`, a PDL dependent:
    - sums the partials, then scale+SiLU;
    - the BF16 up-projection and the gated mean;
    - stores the combined residual.
  - 17.1 µs per mix at one token (590 GB/s), against 20.2 for v1.
- **Mixer, one of two:**
  - **GDN** (36 layers): conv, L2 norm, a register-resident delta rule (`k_gdn_delta_reg`, the
    state read and written once per call), gated RMS norm.
  - **QSA** (12 layers, `il % 4 == 3`): Q/K/V, norm+rope, the KV write, the indexer (pooled keys,
    scores, select), then split-K flash-decode attention (see QSA below).
- **The FFN-side hyper-connection mix** (with the combine folded in, as above).
- **MoE fast path** (`arch/qwen4exp/moe_fast.cu`):
  1. BF16 router, then `k_route`:
     - softmax;
     - top-10 by rank counting over candidates (the k-th largest warp maximum filters them);
     - the hit list as **weight pointers** (cache slots);
     - it publishes the CPU misses and x to the layer's mailbox in mapped host memory and raises
       the `routed` flag.
  2. The GPU runs the hits and the shared expert:
     - `moe_hits`: `k_moe_gate_up` (dp4a, x quantized to int8 per 64 values, SwiGLU, hidden rows
       quantized in the kernel), then `k_moe_down`;
     - the shared expert through ggml MMVQ.
  3. `k_moe_combine_db` waits on the GPU for the mailbox `done` flag, then sums hits, misses and
     the gated shared expert. The wait is bounded (10 s and 10M polls, signed timer arithmetic).
     On timeout it writes an error word, which the host reports with the miss server's state.
- **Miss server** (`MissServer`, a thread pinned to physical CPU 0):
  1. spin on the `routed` flag;
  2. `moe_cpu` on `CpuPool` (8 workers, 2 ms spin before sleeping), AVX-512 VNNI on the arena's
     planar Q2_0 blobs;
  3. write the result into the mailbox and raise `done`;
  4. log the layer's routing into `access` (read by the cache manager).
- **Thread placement:** the enqueue thread sits on CPU 8, and helper threads unpin themselves.

## Speculative decoding with the MTP head (P2)

A round, in `fr_bench --spec K` and `flashrt-engine --spec K`:

1. **Draft** (`MtpHead`, `arch/qwen4exp/mtp.cu`): the head runs over the rows the target kept in
   the last round (its catch-up, with the target's final streams as h), and the last row's
   logits give draft 1. Drafts 2..K come from `draft_chain`: one captured graph per step, fed its
   token and position on the device by the previous step's draft. One host sync per round.
   At temperature > 0 each draft is **sampled** from the head's q: its logits through the
   target's sampler chain, with a salted per-position draw (`sample::draft_row`). q is kept on
   the GPU. Greedy decoding and `--argmax-drafts` / `FLASHRT_ARGMAX_DRAFTS=1` use the argmax.
2. **Verify** (`ForwardRef::forward_window`): the target runs the window `x_p, d_1 .. d_K` (T =
   K + 1 tokens) in one doorbell step on the fast path, captured as a graph pair per T.
3. **Accept:**
   - **Sampled drafts** (`sample::spec_verify`, sw85): draft j is kept when u * q(d) < p(d);
     otherwise the token is drawn from max(0, p - q), normalised, and the round ends. The last
     row is a plain sample.
     - Exact in distribution (`test_spec_sample`; the distribution test).
     - Tokens no longer equal plain sampling's one for one.
     - First-draft acceptance at 32K rose from ~51% to ~76%.
   - **Argmax drafts:** every row is sampled (greedy is argmax), and drafts are kept while the
     sampled token equals the draft. With a deterministic draft this is also exact, and
     token-for-token equal to plain sampling (position-keyed draws, `test_sample`).
4. **Commit** (`ForwardRef::commit(a + 1)`): the recurrent states are rewound to the kept tokens.

**The head** is the NextN block of the draft GGUF (`-noembd`: the target's embedding and LM head
are borrowed). Its input at position p is the target's final hyper-connection streams at p - 1
and the token at p (llama.cpp's graph_mtp semantics); its attention is a QSA layer with its own
KV cache and indexer. Its 512 experts live in VRAM, requantized Q8_0 → Q4_0 at load (1,449 MiB
in all). Its head is a gathered copy of the target's LM-head rows for the top 32,768 tokens of a
frequency ranking (`bench/mtp_vocab.py`) plus the prompt's distinct tokens.

**Rewinding a window** (what `commit` needs):
- **GDN:** the delta-rule kernel saves the state before the window (one 113 MB backup per
  window, not one per token); a partial accept replays the kept tokens from the saved per-token
  inputs. The conv history is rebuilt from the saved old history plus the window's inputs.
- **PLE:** the n-gram conv history, the same way.
- **KV caches (target and head):** nothing: rejected positions are rewritten before anything
  reads them. The indexer's raw-key ring holds 2 blocks of positions for this, so a rewind never
  finds an older position's key overwritten.

**The MoE in a window:** routing per token (one `k_route` block each), the GPU hits grouped by
expert (each distinct expert read once for all its tokens), and the CPU misses grouped by expert
too (`moe_cpu` takes up to 4 tokens per expert). Each missed expert is read once per window, but
the union of a window's misses grows almost linearly with its length: the CPU misses dominate a
window's cost.

## Prefill in chunks (P3)

A `forward()` of more tokens than the decode batch is a chunk. The engine picks the length
(`ForwardRef::pick_chunk`): the longest, up to 16,384, whose buffers fit the free VRAM, estimated
from the allocation formulas (`chunk_bytes`). That is 16,384 at 32K and about 10-11K beside 245K
of KV. `fr_bench --prefill-chunk auto` does the same.
- **Dense layers** run as matrix-matrix products (`kernels/cuda/ggml_gemm.h`): ggml's MMQ int8
  tensor-core kernels for the quantized types, launched by flashrt (`mmq_launch.cuh`, one file per
  weight type), cuBLAS for BF16. Q3R matrices are unpacked back to Q3_K for it.
- **The MoE** (`moe_stream.cu`): a chunk routes to essentially every expert, so the layer's whole
  expert slice (676 MB) is copied from the host arena while the previous layer computes, converted
  on the GPU from the arena's planar Q2_0 to ggml's layout, and multiplied by grouped MMQ (the grid
  sized by the largest expert's token count). About 32 GB cross PCIe per chunk, at 41-50 GB/s.
- **QSA** runs its scoring, selection and attention in sub-batches of 128 tokens. With host KV,
  the chunks attend from a VRAM mirror of the cache (sized to the prompt, freed after).
  - **Attention** (`k_attn_tc`): one CTA per (kv head, token). The GQA group's 12 query heads are
    the M = 16 rows of fp16 `mma.m16n8k16`, with fp32 accumulation. K and V go from the token's
    cell list straight into fragments, with no shared staging: the dot product's k order and the
    output's column order are permuted so each lane reads contiguous bytes (64 dims of one cell
    for K, one Q8_0 block of four cells for V). The softmax is online in base 2, and there are no
    split-K partials.
  - **Scores** (`k_idx_scores_tc`): 32 tokens x 4 heads as M rows against tiles of 64 pooled keys
    in fp16, with relu and the head sum fused.
  - Decode and verify windows keep the FP32 split-K kernels.
- **GDN** (calls of 64+ tokens): the chunked (WY) delta rule on fp16 tensor cores
  (`gdn_delta_prefill`, sw71).
  - Chunks are 64 tokens; `k_gdn_chunk_prep` does the S-free part of every (chunk, head) in
    parallel:
    - it forms K K^T and Q K^T by mma, with the decays and beta in the epilogue;
    - it solves T = (I + A)^-1 in fp32 by column substitution in registers;
    - it forms W = T diag(beta gamma) K and U~ = T diag(beta) V by mma;
    - it writes Q^, K^T, W and P as ready mma A fragments.
  - `k_gdn_chunk_state`, one block per (head, 32 value columns), carries the state through
    the chunks, fp32 in registers:
    - U = U~ - W S0;
    - O = gamma Q S0 + P U;
    - S0 = gamma_C S0 + K^T U;
    - an fp16 copy of S0 feeds the mma. This is FLA's precision split.
  - In slabs of 8 chunks it is 1.8x the column kernel, which remains the fallback
    (`FLASHRT_GDN_CHUNK=0`) and the path for 16-63 tokens: four lanes share two state columns, and
    q, k, v arrive in tiles of 8 through shared memory.
- **Hyper-connections:** the RMS norm writes `xn` in BF16 as well, for the down and inject
  products. The combine after the mixer is fused into the FFN mix's norm (`k_hc_combine_norm4`).
- **Expert grouping** (`moe_prepare`) is flashrt's own: per-block histograms, a scan and a stable
  placement, O(T K). ggml's helper scanned every slot once per expert.
- **The routed experts** (`moe_q2::run`) read the streamed planar Q2_0 slice directly: there is
  no conversion to ggml's layout.
  - One 32-bit load per lane gives both k-steps' `mma.m16n8k32` A fragments of a 64-weight block
    (shifts 0/2/4/6).
  - The codes enter the MMA unsigned; Q2_0's -1 is folded into each activation block's
    `kMagic - sum`, which also turns int32 into float without I2F.
  - Gate and up run in one kernel; its epilogue does SwiGLU and the int8 quantization of the
    down input.
  - Down keeps the tile's activations in shared memory and streams the weights over all row
    tiles.
  - The tile list is built on the GPU. Per-slot outputs are BF16.
  - `FLASHRT_MOE_Q2MMA=0` switches back to the MMQ path.
- **Hyper-connections in prefill:**
  - the norm writes 1/rms, and the gated mean recomputes xn from x;
  - one block per token does the combine, the norm and the 4-output inject product;
  - the layer-end combine is deferred into the next layer's first mix;
  - the up product writes the gate in BF16.
- **The n-gram rows** of the next chunk are read from the SSD while a chunk computes.
- **Routing counts** accumulate on the GPU and feed the expert cache after the prefill.

The engine lends the expert cache's VRAM to the chunk buffers (about 4-7 GB) and rebuilds the
cache after, from the prefill's routing counts and the startup prior.

## Why it is shaped like this (with the evidence)

| Decision | Why | Evidence |
|---|---|---|
| **Doorbells instead of a host sync per layer** | The host sync serialised GPU work, CPU misses and launches. With doorbells the whole token is enqueued (now captured) at once. | sw3: 60.5 → 63.5 tok/s at 2K |
| **CUDA graphs, two per token** | About 1,600 launches per token left roughly 1.1 ms of gaps. The split around PLE keeps the SSD read overlapped. | sw17: 89.8 → 101.3 tok/s at 2K, same tokens; 32K 79 → 95-99 |
| **Per-token values in a device block** | Graph replay needs fixed launch arguments. The QSA kernels, doorbell kernels and embedding read position, seq and token from it. | sw17 |
| **Expert cache in the arena's planar layout** | Uploads are plain copies (no staging, no unrepack), and one layout serves the CPU and GPU kernels. | sw14 |
| **dp4a hit kernels** (int8 activations per 64 values) | ggml's grouped MMVQ ran the down-projection at about 300 GB/s and always covered 10 entries, pads included. | sw14: MoE hit time 1.6 → 0.97 ms per token; `test_moe_hits` 0.8-1.4% relative L2 |
| **Q3R for tall Q3_K matrices, converted in place** | ggml's Q3_K MMVQ reaches 364-388 GB/s on sm_120. The limit is decode arithmetic, not bytes; a planar dp4a layout reaches 549-611 GB/s. Only matrices with ≥ 4096 rows and K ≤ 4096 qualify: ggml wins on short or wide ones. | sw10 (v1 kept a duplicate copy: net loss), sw16 |
| **Adaptive decayed-LFU cache, swap budget 32** | A static cache from the prompt's routing falls to 66% hits on new text; the adaptive one holds about 89%. The budget was 8 until sw89: when the answer routes unlike the prompt (window 9's chat prompts: 66% hits against 93% on wikitext), 32 uploads per step recover 9-12%; on wikitext continuation it costs 1%. | sw11-12, sw21-22, sw88-sw90 |
| **Prefill routing counts halve every 4,096 tokens** | At depth, the whole prompt's counts are a poor prior. | sw21: first-window hit rate at 245K 67-69% → 75-79% |
| **Split-K flash-decode attention** | The first kernel spent 7.1 ms per token on attention. | sw2: 7.1 → 0.23 ms |
| **Deterministic indexer selection** (block order, no float atomics anywhere) | Atomic slot order made the attention sum order, and so the output, vary from run to run. | sw5: runs bit-reproducible since |
| **Select: 8-bit radix, warp-aggregated histogram, segmented ordered output** | At 245K (61K blocks, one CTA per layer) selection took 1.2 ms per token. | sw20: 1.21 → 0.85 ms; selection unchanged (`fr_parity qsa`) |
| **fp16 KV storage** | The values were fp16-rounded already. Half the VRAM, same numerics. | sw12: parity identical line for line |
| **q8_0 KV** (`--kv q8`) | llama.cpp's own q8_0 cache is the noise floor (KLD 0.0078). It frees about 2,000 expert slots at 245K. | sw18: KLD 0.0092; 245K 61-64 → 63-71 tok/s |
| **Host KV + GPU hot set** (`--kv-hot 4096`) | QSA reads only 2,052 cells per layer per token. Promoting the selected blocks *before* attention keeps attention on GPU memory. | sw19-20 (v1 read misses zero-copy inside attention: slower), sw22 |
| **Token embedding in host memory** | Decode reads one 1.1 KB row per token, so its 260 MB of VRAM is better spent on the expert cache. | sw22 |
| **Graph-mode QSA always runs the selection path** | It keeps the graph valid at every position; the select kernel falls back to dense below the width. | sw17 |
| **MTP head experts at Q4_0, head over 32K ranked + prompt tokens** | Q4_0 halves the head's VRAM (more expert slots), the trimmed head halves the draft step, and neither costs measurable acceptance. | sw26 |
| **GDN backup + replay for rewinds** | One state copy per window instead of one per token; a partial accept replays only the kept tokens. | sw25 (KLD with rewinds 0.0087) |
| **Fused hc kernels for windows, each weight read once** | The generic path read the BF16 hc weights at 300 GB/s: 4.6 ms of a 3-token window. | sw28, sw29 |
| **Window graphs, one pair per length** | Eager windows paid about 1.4 ms of launches per round. | sw27 |
| **Prefill: every layer's experts streamed, not only the misses** | A 4K-8K chunk touches all 512 experts of a layer; one 676 MB copy per layer overlaps the previous layer, and the cache's VRAM is free for the chunk's buffers. | sw36-sw39 (P3): 136 → 2,135 tok/s at 32K |
| **A VRAM mirror of host KV during prefill** | Chunk attention reading the host store over PCIe ran 2.6× slower. | sw42, sw43: 245K 738 → 1,940 tok/s |
| **Tensor-core prefill attention, K/V gathered into fragments** | The FP32 split-K kernel ran at about 7 TFLOPS and wrote 0.8 MB of partials per token-layer. | sw46: 32K 2,193 → 2,644 tok/s; 64K attention 6.1 → 0.95 s |
| **Sampled drafts at temperature > 0** | Argmax drafts accept with p(argmax q), sampled drafts with sum min(p, q): 0.48-0.58 against 0.66-0.71 (probe). 32K `--spec 1` 119.0 → 143.1 tok/s, 245K 91.7 → 97.1. With them a second draft is break-even or slightly ahead. | sw84, sw85 |
| **Decode: work inside the CPU-miss window does not pay** | In a layer with CPU misses, the hits and the shared expert run while the host computes the misses, and `k_moe_combine_db` waits for it. Faster kernels there mostly lengthen the wait. Gains have to come from work outside that window (hc, mixers, dense mat-vecs), or from fewer misses. A CPU miss costs 41-46 µs in decode, near host-DRAM bandwidth. | sw78, sw79 (moe hits v2 reverted) |
| **Decode hc kernels v2** | v1 ran at ~500 GB/s: shared-memory bound at T >= 2, with a serial norm preamble. v2: 2 rows per warp, weights before the norm, 1/rms after the dot product, up kernel as a PDL dependent in one wave. | sw75 (32K plain +2.7%) |
| **Doorbell skip for tokens without misses** | 61% of layers at 32K have no CPU miss; they skip the x copy and the mailbox wait and read (the host still serves them, for statistics). | sw77 (`--spec 1` 32K +1.8%) |
| **Decode launch fusions** (`linear_multi`, `FLASHRT_FUSE_EPI`) | A decode step is hundreds of 2-6 µs kernels. The BF16 projections of one input share a launch (a block of 4 warps per row: a warp per row lost to two MMVF launches), the q/k norm runs in the conv kernel, and SwiGLU and the gated norm write the next mat-vec's q8_1 input. | sw73, sw74 (32K plain 99.8 → 104.1; KLD 0.00891) |
| **GDN prefill: the chunked form on tensor cores** | fp32 chunked is exact but 5x slower: it needs 2.4x the recurrence's FLOPs, and the fp32 tensor paths are no faster than the CUDA cores (TF32 61, BF16 122 TFLOPS). fp16 mma with an fp32 state is 1.8x the column kernel. Prep is DRAM-bound (about 70 MB per 512-token slab); state is mma-bound. | sw70, sw71 (+3.1-3.5% prefill, KLD 0.0084), sw72 |
| **GDN: lanes own columns, tokens tiled through shared memory** | The block kernel waited on 4 barriers per token. v1, which prefetched one token into registers, waited on DRAM (slower). v2 was bound by shared-memory reads (24 per lane-token, over 4 addresses). | sw47, sw50: 32K 3,216 → 3,531 tok/s |
| **Tensor-core indexer scores** | FP32 scoring is linear in depth: 1.65 s at 64K, an estimated 20 s at 245K. | sw49: 64K 2,984 → 3,197 tok/s; 0.16 s |
| **BF16 from the hc norm; own expert grouping; hc combine fused into the norm** | The conversions cost as much as the BF16 GEMMs (1.17 s at 64K); ggml's grouping took 0.67 s. | sw48, sw51 (bit-identical) |
| **The arena registered at load** | `cudaHostRegister` takes 1.8 s and fell into the first long prompt. | sw50 |
| **Experts from the planar layout on own int8 kernels** | ggml's MMQ ran at about 57-77 TOPS, and the layout conversion, SwiGLU and quantization passes cost more. | sw55: 32K 4,019 → 5,111 tok/s, 245K 3,622 → 4,871; test_moe_q2 1.8e-4 against MMQ |
| **hc traffic: no xn write, inject in the norm block, deferred combine** | The hc elementwise kernels were at bandwidth (about 2 s at 64K). | sw58, sw59: 5,156 → 5,432 tok/s |
| **BF16 per-slot expert outputs and hc gate** | The per-slot outputs were 100 KB per token, written and read back; KLD unchanged within the spread. | sw59: 5,432 → 5,606 tok/s; 0.8 GB less at 16K chunks |
| **Decode A/B runs teacher-forced** (`--teacher`) | With sampled text, other slot counts produce other text, whose hit rate moves 20 points; forced tokens route alike. | sw64 |
| **More expert-cache slots: reserve 256 MiB, Q2_0 head, hc down Q8P** | Misses are bound by host DRAM, so fewer misses is the lever. | sw64 +2.5%, sw65 +5%, sw68 +2.3-2.6% |
| **Chunk length from free VRAM** | Longer chunks fill the expert tiles better (+8.6% from 8K to 16K), but 16K does not fit beside 245K of KV. | sw52-sw54: 245K 3,362 → 3,623 tok/s |
| **Sampling draws keyed by (seed, position)** | A position's sample is the same in a plain step and in a verify window, so speculative output can be checked against plain output token for token. | test_sample, sw31 |

## Tried and rejected, or parked

- **PCIe zero-copy misses** (`--pcie-frac`, Strata's idea): no gain at 2K or 245K. A 1.32 MB
  expert read over PCIe lengthens the GPU's part of each layer more than it saves the CPU. The
  flag remains, off by default. (sw15)
- **Q3R v1:** a second copy of the weights plus float activations. The VRAM it took from the
  expert cache outweighed the kernel gain. (sw10)
- **Float activations for sub-4-bit decode:** compute-bound (int-to-float conversion at quarter
  rate). Use dp4a. (sw10, sw14)
- **Hot set v1** (zero-copy miss reads inside attention, one-CTA promotion afterwards): slower
  than plain q8. (sw19)
- **Swap budget 32 as the default** was rejected in sw22 (churn at 32K) and adopted in sw89-sw90,
  once window 9's protocol showed the warm-up cost on prompts that predict the answer's routing
  poorly. The churn cost measured 1% (sw90).
- **A cache prior from routing statistics** (sw89): no gain over the prompt's own routing.
- **Q4 KV with a Hadamard rotation** (planned for P4): not done. With the hot set, KV VRAM is
  about 0.2 GB, so it matters little now.
- **Fewer experts per token** (from P0): rejected at 2-bit, because the KLD cost is too high.
- **Gating drafts on the head's own probability** (`--draft-pmin`): no better than a fixed K at
  2K (sw29). The probability over the trimmed vocabulary is not calibrated enough.
- **Token-id prefix as the drafter's vocabulary:** the first 40K ids cover only 92.7% of wiki
  tokens. A frequency ranking is used instead.
- **Narrower expert MMQ tiles** (J = 64 or 32 instead of 128, for fuller tiles at about 160 tokens
  per expert): slower, 2,640 and 2,211 tok/s against 2,896 at 32K. Weight reuse matters more
  than tile fill. (sw48)
- **MMQ without stream-k for the expert calls:** not possible without patching ggml. The
  kernel's tiling is chosen at compile time, and stream-k already spreads the experts evenly.
- **Activation scales per 64 in moe_q2** (`FLASHRT_MOE_AB64=1`): +2.3% at 32K, but KLD +0.0003
  (fp16) and +0.0007 (q8), outside the spread. Kept off. (sw57)
- **More pipeline stages for moe_q2's gate/up** (3 or 4): no change. The kernel is not
  load-latency bound. (`test_moe_q2`)
- **GDN chunk variants** (sw72):
  - U~ in fp16: no faster, KLD 0.008744 against 0.008396;
  - slabs of 4 or 16 chunks, and 64 or 128 columns per state block: all slower.
- **GDN with 4 accumulators or 8 lanes per column:** no gain (sw60). The kernel runs at about 3x
  its instruction-issue estimate for reasons not found without performance counters.
- **Prefetching forecast experts** (layer L's router on layer L-1's output): top-16 catches
  66-68% of misses, but uploads read host DRAM, the CPU misses' bottleneck, about twice per miss
  removed. (sw63)
- **The hc up matrices as Q8P:** +5% at 32K, but KLD +0.0011 (the down matrices convert free).
  Opt-in: `FLASHRT_HC_Q8=1`. (sw66, sw67)
- **Attention over groups of adjacent tokens:** their selections overlap too little (4 tokens:
  union 1.72x one list), and building the union per group costs about what a CTA does now. (sw60)

## Current numbers (all exact within the KLD gate)

| Arm | tok/s | Folder |
|---|---|---|
| 2K, 256 tokens, 3 runs | 100.8 / 102.0 / 101.9 | `2026-09-28-sw22-defaults` |
| 32K fp16 KV, fresh prefill, 3 windows | 94.9 / 96.2 / 99.2 | `2026-09-28-sw17-graphs` |
| 245K fp16 KV, saved state, 3 windows | 61.3 / 62.0 / 64.2 | `2026-09-28-sw17-graphs` |
| 245K q8 KV, 3 windows | 70.2 / 63.5 / 71.5 | `2026-09-28-sw18-kv-q8` |
| 245K q8 + hot set 4096, fresh prefill, decayed prior, 6 windows | 68.7 / 70.5 / 75.9 / 67.3 / 80.2 / 76.3 | `2026-09-28-sw21-warmup` (budget 32) |
| Fast-path KLD (2 × 8K wikitext against the FP16-KV llama.cpp base) | 0.0087-0.0092 | sw17-sw22 |
| **2K greedy, `--spec 1` / `--spec 2`** (MTP) | 123.1 / 121.0 | `2026-09-28-sw29-fused-hc` |
| **32K, temperature 1.0: plain / `--spec 1` / `--spec 2`, 6 windows, means** | 97.9 / 114.2 / 115.3 | `2026-09-28-sw31-p2-temp1`, `sw33` |
| **Teacher-forced (same tokens every arm), 6 windows, means: 32K plain / 32K `--spec 1` / 245K `--spec 1`** | 106.7-107.2 / 112.6-112.9 / 87.4-88.8 | `2026-09-28-sw78-hc-comb` (after sw73-sw77); sw68 had 99.6 / 106.4 / 82.9 |
| **245K, temperature 1.0: plain / `--spec 1` / `--spec 2`, 6 windows, means** | 78.5 / 90.2 / 83.2 | same |
| **245K greedy `--spec 2`, fresh prefill with the head, 3 windows** | 83.7 / 94.8 / 92.3 | `2026-09-28-sw30-spec-245k` |
| Verify window KLD (windows of 3 with rewinds, hot set 512) | 0.009232 | `2026-09-28-sw33-p2-temp1-rerun` |
| **Prefill, automatic chunks, 32K / 64K (q8 KV)** | 6,216 / 6,239 tok/s | `2026-09-28-sw83-moeq2-pack` (sw71: 5,955 / 5,982) |
| **Prefill, automatic chunks, 245K (q8 KV / host KV + mirror)** | 5,309 / 5,170 tok/s | `2026-09-28-sw61-milestone` |
| **Engine: 32K prompt, MTP head, cache rebuild** | 7.9 s | `2026-09-28-sw61-milestone` |
| Prefill KLD (logits from chunks; fp16, q8; fast path after chunks) | 0.0082-0.0087 | sw47, sw49, sw50 |
| Prefill, reference path (CPU experts, 64-token batches) | 109-123 tok/s | |
| **Window 9's protocol** (the reference engines' prompts, 384 tokens, 3 runs), 1K / 32K / 134K / 250K: greedy `--spec 2` | 106.6 / 83.7 / 81.0 / 74.8 | `2026-09-29-sw91-depthbench` |
| same, temperature 1.0 `--spec 2` (sampled drafts) | 82.2 / 77.7 / 81.0 / 76.4 | same |
| same, no MTP, greedy | 94.7 / 81.7 / 77.4 / 73.0 | same |
| same, Strata greedy / temperature 1.0 (its build warns that its cache path changes outputs; Strata and llama.cpp reused prefixes, flashrt did not); llama.cpp | 87.0 / 96.0 / 85.0 / 80.4; 80.5 / 79.1 / 73.9 / 69.9; 37.4 / 37.2 / 32.8 / 30.7 | `2026-09-27-w9-validation` |

**Where the time goes at 245K with the hot set** (nsys `--cuda-graph-trace=node`, sw20):
- 14.1 ms of GPU kernel time per token;
- 3.6-4.1 ms of it is waiting on CPU misses, and the CPU runs at DRAM bandwidth (DDR5-3600, EXPO
  off);
- selection 0.85 ms, scoring 0.45 ms (at bandwidth), attention 0.43 ms, hot-set upkeep 0.56 ms.

**At 2K:**
- the hyper-connection kernels take about 2 ms;
- the miss wait is about 1.2 ms;
- the Q3R kernel takes 0.74 ms.

## Correctness machinery

- **`fr_kld`:** the P1 gate protocol. It reads llama-perplexity's `--kl-divergence-base` file.
  With `--fast`, the scored half runs one token at a time through the real decode path. The base
  is `$BENCH/kld/kl8k-f16.bin` on the box.
- **`fr_parity`:** block-level parity against llama.cpp eval-callback dumps
  (`$BENCH/parity/*.frd`). It loads weights with ggml Q3_K (no Q3R), so projections stay
  comparable.
- **Unit tests:**
  - `test_gemv`: every dense type, plus Q3R at 1 and 4 tokens;
  - `test_moe_hits`: the GPU hit kernels against a double-precision reference, including the pad
    entry;
  - `test_cpu_pool`: 20,000 runs with idle gaps (the Strata #29 race class);
  - `test_q2_0`, `test_moe_cpu`.
- **Determinism:** fr_bench runs are bit-reproducible. A GPU hit and a CPU miss are not
  bit-identical (Q8_1 against Q8 arithmetic), so tokens can differ between cache configurations.
  Compare with KLD, not tokens.

## Runbook (on the target box, as capped `systemd-run` units)

Paths in this runbook and in the `bench/results` scripts are placeholders:

| Placeholder | Directory |
|---|---|
| `$FLASHRT` | this checkout |
| `$MODELS` | the model's GGUF shards and the MTP draft head (`mtp-Flash-Next-Q8_0-noembd.gguf`) |
| `$BENCH` | a scratch directory: prompt ids, saved states, the KLD base, run folders and logs |
| `$DATA` | datasets, other model builds and routing traces |
| `$STRATA`, `$LLAMA_CPP` | the reference engines' trees (Strata; llama.cpp with the expert cache and sparse QSA) |
| `$EXT` | an external drive for backups |

The scripts run with `set -u`, so they stop if one is unset.

- **Model:** `M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`.
- **Prompt:** `I=$BENCH/p0c-20260927/wiki.prompt_ids.txt` (250K wikitext token ids, the P0
  prompt).
- **Decode speed:**
  `build/fr_bench $M --ids $I --n-prompt N --gen 128 --windows 3 [--kv q8] [--kv-hot 4096] [--swap-budget B] [--no-graphs] [--no-q3r] [--pcie-frac F] [--static-cache] [--count-half-life N] [--trace FILE]`
- **State snapshots** skip the prefill for depth tests (`--save-state` / `--load-state`, speed
  only). They live in `$BENCH/` on the box:
  - `state-245k.bin`: fp16 KV, 6.5 GB, full-prompt counts;
  - `state-245k-q8.bin`: q8 host KV, decayed counts;
  - `state-32k.bin`: fp16 KV.

  An fp16 state loads into a q8 cache. A state reflects the kernels that wrote it.
- **KLD gate:** `cd $BENCH/kld && build/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast [--kv q8 | --kv-hot 512] [--window W]`.
  It takes about 5 minutes. `--window W` scores verify windows with random rejected tails.
- **Speculative decoding:** add `--mtp $D --spec K --draft-vocab $BENCH/mtp-vocab/ranks.txt`
  (`D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf`; the ranking comes from
  `bench/mtp_vocab.py`). `[--mtp-bits 8|4|2]` sets the head's experts (default 4).
  Sampling: `--temp 1.0 --top-k 20 --top-p 0.95 [--seed S]`. `--dist-test N` runs the
  distribution test. States with the head: `$BENCH/state-32k-q8-mtp.bin` and
  `state-245k-q8-mtp.bin` (each with a `.mtp`), for `--kv-hot 4096`.
- **Engine:** `build/flashrt-engine $M [--mtp $D --spec K --draft-vocab RANKS] [--ctx N] [--cache-prior FILE]`,
  then JSON lines on stdin (`bench/engine_smoke.py` drives it). The prior:
  `$BENCH/cache-prior-calib32k.bin` (also in `bench/results/2026-09-28-sw35-server`).
  `engine_smoke.py --faults` checks that bad and failing requests leave it serving
  (`bench/results/2026-09-29-sw92-faults/sw92.sh`); run it after changes to `Session` or the
  forward's state.
- **Server:** `cargo build --release --manifest-path server/Cargo.toml`, then
  `server/target/release/flashrt-server --model $M --port 8090 --engine build/flashrt-engine --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 1 --engine-arg --draft-vocab --engine-arg RANKS --engine-arg --cache-prior --engine-arg PRIOR`
  (`--api-key KEY` to require one). Checks: `--check-tokenizer TEXT IDS`, `--render REQUEST.json`,
  and `bench/server_smoke.py --url ...` against a running server (`bench/results/2026-09-28-sw86-server/sw86.sh`
  runs the whole check as temporary units; `2026-09-29-sw93-server/sw93.sh` also kills the engine
  under the server and stops it gracefully). Only as a test unit: no service stays up on the box.
  Run it after any change to VRAM budgeting or the engine: sw86 caught a first-request
  out-of-memory that `fr_bench` cannot see. If the engine exits, the server answers 503 and exits
  3 s later, for a supervisor to restart; a service unit for it should set `KillMode=mixed`, so
  that a stop reaches the server, which then asks the engine to quit (sw93).
- **Kernel profile:**
  `/usr/local/cuda-12.9/bin/nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace=cuda build/fr_bench ... --gen 64`.
  Without `--cuda-graph-trace=node`, graphs appear as single launches.
- **Microbenchmarks:** `build/bench_gemv [--q3r]`, `build/bench_moe_cpu`.
- **Variance:** at depth, compare with at least 6 windows. The generated text alone moves the hit
  rate by 20 points from one window to the next.

## Known issues

- **Two unexplained aborts** ("unspecified launch failure", Xid 43, sw7 and sw11). They fit a
  false doorbell timeout from unsigned timer arithmetic, which is now fixed and reports instead
  of trapping. There has been none in the 30+ runs since. The root cause is not proven.
- **The chunk path holds about 0.45 MiB per token.** moe_q2 and the BF16 per-slot outputs
  removed about 120 KB per token (sw53, sw59), so 245K now fits chunks of 15-16K. The block
  scratch is still sized about 30% above the widest mixer's need.
- **Hot set:** `k_hot_select` is serial CLOCK in one thread (17 µs per layer), and the copy costs
  about 30 µs per layer at steady state. Both have room to improve.
- **`k_idx_select`** still spends about 70 µs per layer at 245K in 4 single-CTA histogram passes.
  A multi-CTA histogram would roughly halve it (estimate).
- **Failed requests** reset the session to an empty sequence, so the next request starts cold
  (sw92). A doorbell timeout or a sticky CUDA error ends the process (status 3).
- **Server limits:** text only (no images); tool_choice "required" or a named tool is not
  enforced (the model decides); Anthropic thinking blocks carry an empty signature; without a
  `thinking` field the model still reasons, and the reasoning is not returned.
- **Prefix reuse keeps one sequence and one checkpoint** (before the last prompt's last token).
  Two conversations interleaved on one server re-prefill each time.
- **The engine refills the expert cache from the prefill's routing counts** after the first
  prompt and after any prompt that adds 4,096 or more tokens. For short prompts the adaptive
  policy alone moves it.
- **The drafter's probability does not gate drafts well** (sw29). Draft length is fixed per run.
- **At depth, windows keep only token T - 1's selected KV blocks in the hot set;** the others
  are read from the host store when missing. Not measured separately.

## Next steps (priority order)

`docs/sweet-spots.md` has the tuned configurations, the knee of each track and the untested
paths ranked by expected value. Sampled drafts with speculative sampling, the top of that
list, are done (sw85: 32K `--spec 1` 119.0 → 143.1 tok/s, 245K 91.7 → 97.1).

1. **Cheaper verify windows:** the misses dominate. They are host-DRAM bound (sw79), so the
   levers are fewer misses and a better drafter:
   - a draft length chosen per round from the window's expected misses;
   - more cache slots;
   - drafter acceptance: 45-54% for one draft.
   GPU work inside the miss window does not pay (sw78, sw79). The grouped window hit kernels
   take 39 µs at T = 2 against 25 µs at T = 1 for the same 10 experts (`test_moe_hits`), which
   only matters in layers without misses.
2. **Prefill** (64K, 10.5 s). Done since sw69:
   - Q3_K as Q8_0;
   - the chunked GDN;
   - routing a warp per token (26x, sw81);
   - the q/k norm in the conv;
   - one BF16 conversion per input (sw82);
   - packed moe_q2 scales (sw83).

   sw80's profile of what is left: dense MMQ ~22%, hc ~21% (at bandwidth), routed experts ~19%,
   GDN ~9%, attention ~9%. In order:
   - GDN: the chunked form is in (sw71). Left there:
     - prep traffic: raw Q and K^T per key group, and V read by the state kernel with T in the
       state step, would cut about 70 → 35 MB per slab;
     - the state kernel's imbalance: 192 blocks on 84 SMs.
   - moe_q2's gate/up: about 175 TOPS against a 283-TOPS ceiling. The per-32 scale arithmetic
     is the floor, short of per-64 scales, which KLD rejected (sw57).
   - attention (L2-bound gathers);
   - the hc gated mean and norm, at bandwidth.
   Decode: the small mat-vecs that share an input, and the norms and SwiGLU in front of a
   mat-vec, are fused (sw73, sw74: plain +4.3%, `--spec 1` +1.9-2.3%). What is left there is
   mostly the hc kernels and the MoE combine.
3. **Tuning:** multi-CTA select, a parallel hot-set CLOCK, an adaptive swap budget (larger
   while the hit rate is low). Decide after phase 2 of `docs/improvement-plan.md`: the gap to
   Strata at 32K-250K on window 9's protocol is the cache warm-up (sw88), prefix reuse, or both.
4. **The MTP head's pass over the prompt:** prefill with the head runs 2,150-4,130 tok/s
   against 5,690-5,960 without it (sw87, sw91).
