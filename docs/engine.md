# flashrt engine as built (2026-09-28)

This describes what the code does today, why each piece is shaped the way it is, what was tried
and rejected, and how to measure it. `design.md` holds the plan and the phase gates. The
per-experiment evidence is in `bench/results/2026-09-2*`, each folder with a README.

**Status:**
- **P1 is complete.** The gates are met: 32K 94.9 / 96.2 / 99.2 tok/s, 250K 61-64 tok/s with
  fp16 KV, fast-path KLD 0.0087.
- **P4's KV work is done:** q8 KV and host-resident KV with a GPU hot set. With them, 245K runs at
  66-80 tok/s over 6 windows.
- **Next:** P2, speculative decoding with the MTP head.

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

- **Hyper-connection mix (4 streams, rank 320), two fused kernels:**
  - `k_hc_down`: RMS norm + BF16 down-projection + inject projection, with per-stream partial
    sums;
  - `k_hc_up_mix`: sum of the partials, scale+SiLU, BF16 up-projection, gated mean.
- **Mixer, one of two:**
  - **GDN** (36 layers): conv, L2 norm, a register-resident delta rule (`k_gdn_delta_reg`, the
    state read and written once per call), gated RMS norm.
  - **QSA** (12 layers, `il % 4 == 3`): Q/K/V, norm+rope, the KV write, the indexer (pooled keys,
    scores, select), then split-K flash-decode attention (see QSA below).
- **Hyper-connection combine,** then the FFN-side hyper-connection mix.
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

## Why it is shaped like this (with the evidence)

| Decision | Why | Evidence |
|---|---|---|
| **Doorbells instead of a host sync per layer** | The host sync serialised GPU work, CPU misses and launches. With doorbells the whole token is enqueued (now captured) at once. | sw3: 60.5 → 63.5 tok/s at 2K |
| **CUDA graphs, two per token** | About 1,600 launches per token left roughly 1.1 ms of gaps. The split around PLE keeps the SSD read overlapped. | sw17: 89.8 → 101.3 tok/s at 2K, same tokens; 32K 79 → 95-99 |
| **Per-token values in a device block** | Graph replay needs fixed launch arguments. The QSA kernels, doorbell kernels and embedding read position, seq and token from it. | sw17 |
| **Expert cache in the arena's planar layout** | Uploads are plain copies (no staging, no unrepack), and one layout serves the CPU and GPU kernels. | sw14 |
| **dp4a hit kernels** (int8 activations per 64 values) | ggml's grouped MMVQ ran the down-projection at about 300 GB/s and always covered 10 entries, pads included. | sw14: MoE hit time 1.6 → 0.97 ms per token; `test_moe_hits` 0.8-1.4% relative L2 |
| **Q3R for tall Q3_K matrices, converted in place** | ggml's Q3_K MMVQ reaches 364-388 GB/s on sm_120. The limit is decode arithmetic, not bytes; a planar dp4a layout reaches 549-611 GB/s. Only matrices with ≥ 4096 rows and K ≤ 4096 qualify: ggml wins on short or wide ones. | sw10 (v1 kept a duplicate copy: net loss), sw16 |
| **Adaptive decayed-LFU cache, swap budget 8** | A static cache from the prompt's routing falls to 66% hits on new text; the adaptive one holds about 89%. Budget 32 warms up faster at depth but churns the cache at 32K. | sw11-12, sw21-22 |
| **Prefill routing counts halve every 4,096 tokens** | At depth, the whole prompt's counts are a poor prior. | sw21: first-window hit rate at 245K 67-69% → 75-79% |
| **Split-K flash-decode attention** | The first kernel spent 7.1 ms per token on attention. | sw2: 7.1 → 0.23 ms |
| **Deterministic indexer selection** (block order, no float atomics anywhere) | Atomic slot order made the attention sum order, and so the output, vary from run to run. | sw5: runs bit-reproducible since |
| **Select: 8-bit radix, warp-aggregated histogram, segmented ordered output** | At 245K (61K blocks, one CTA per layer) selection took 1.2 ms per token. | sw20: 1.21 → 0.85 ms; selection unchanged (`fr_parity qsa`) |
| **fp16 KV storage** | The values were fp16-rounded already. Half the VRAM, same numerics. | sw12: parity identical line for line |
| **q8_0 KV** (`--kv q8`) | llama.cpp's own q8_0 cache is the noise floor (KLD 0.0078). It frees about 2,000 expert slots at 245K. | sw18: KLD 0.0092; 245K 61-64 → 63-71 tok/s |
| **Host KV + GPU hot set** (`--kv-hot 4096`) | QSA reads only 2,052 cells per layer per token. Promoting the selected blocks *before* attention keeps attention on GPU memory. | sw19-20 (v1 read misses zero-copy inside attention: slower), sw22 |
| **Token embedding in host memory** | Decode reads one 1.1 KB row per token, so its 260 MB of VRAM is better spent on the expert cache. | sw22 |
| **Graph-mode QSA always runs the selection path** | It keeps the graph valid at every position; the select kernel falls back to dense below the width. | sw17 |

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
- **Swap budget 32 as the default:** churns the cache at 32K. (sw22)
- **Q4 KV with a Hadamard rotation** (planned for P4): not done. With the hot set, KV VRAM is
  about 0.2 GB, so it matters little now.
- **Fewer experts per token** (from P0): rejected at 2-bit, because the KLD cost is too high.

## Current numbers (all exact within the KLD gate)

| Arm | tok/s | Folder |
|---|---|---|
| 2K, 256 tokens, 3 runs | 100.8 / 102.0 / 101.9 | `2026-09-28-sw22-defaults` |
| 32K fp16 KV, fresh prefill, 3 windows | 94.9 / 96.2 / 99.2 | `2026-09-28-sw17-graphs` |
| 245K fp16 KV, saved state, 3 windows | 61.3 / 62.0 / 64.2 | `2026-09-28-sw17-graphs` |
| 245K q8 KV, 3 windows | 70.2 / 63.5 / 71.5 | `2026-09-28-sw18-kv-q8` |
| 245K q8 + hot set 4096, fresh prefill, decayed prior, 6 windows | 68.7 / 70.5 / 75.9 / 67.3 / 80.2 / 76.3 | `2026-09-28-sw21-warmup` (budget 32) |
| Fast-path KLD (2 × 8K wikitext against the FP16-KV llama.cpp base) | 0.0087-0.0092 | sw17-sw22 |
| Prefill (reference path, CPU experts) | 111-123 tok/s | P3 targets ≥ 1,700 |

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

## Runbook (on the box, as capped `systemd-run` units; see `CLAUDE.local.md`)

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
- **KLD gate:** `cd $BENCH/kld && build/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast [--kv q8 | --kv-hot 512]`.
  It takes about 5 minutes.
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
- **Prefill** is the reference path (every expert on the CPU, batches of 64): 111-123 tok/s, 34
  minutes for 245K. That is P3.
- **Hot set:** `k_hot_select` is serial CLOCK in one thread (17 µs per layer), and the copy costs
  about 30 µs per layer at steady state. Both have room to improve.
- **`k_idx_select`** still spends about 70 µs per layer at 245K in 4 single-CTA histogram passes.
  A multi-CTA histogram would roughly halve it (estimate).
- **The engine binary and server** (`engine/main.cpp`, `server/`) predate this work. The tools
  (`fr_bench`, `fr_kld`) drive `ForwardRef` directly. Wiring the fast path into the engine
  protocol is outstanding.

## Next steps (priority order)

1. **P2: MTP speculative decoding.** Verifying a window of drafts reads each missed expert once
   per window, which attacks the 3.5-4 ms miss wait at depth. The multi-token CPU kernel exists
   (`moe_cpu` takes up to 4 tokens per miss), and `k_route`/`moe_hits` need windows > 1.
2. **Wire the fast path into `engine/`,** so the server can use it (graphs, doorbells, hot set).
3. **Tuning:** multi-CTA select, a parallel hot-set CLOCK, an adaptive swap budget (large while
   warming up, small afterwards).
4. **P3 prefill.**
