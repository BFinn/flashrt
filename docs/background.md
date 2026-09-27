# Background: what was measured and learned before flashrt

A digest of the 2026-09-26/27 investigation on the target box (RTX 5080 16 GB, Ryzen 9
7900X, DDR5 at 3600 MT/s), which led to flashrt. Read it before choosing what to build
next.

- **[M]** = measured on the box.
- **[Q]** = quoted from a source.
- **[E]** = estimate.

## Measured baselines [M]

Decode tok/s at 1K / 32K / 134K / 250K depth. Identical prompts and token ids, one growing
conversation, 384 generated tokens.

| Engine | 1K | 32K | 134K | 250K | Prefill at 32K / 134K / 250K |
|---|---:|---:|---:|---:|---|
| llama.cpp, stock (deployed) | 33.4 | 28.8 | 20.2 | 14.5 | 1133 / 758 / 478 |
| llama.cpp, patched: expert cache, sparse QSA gather, whole-block top-k, pooled-key cache; no MTP | 40.0 | 38.2 | 35.8 | 31.9 | 1117 / 725 / 444 |
| Strata 0.1.4, default flags, MTP | 62.7 | 67.7 | 70.7 | 52.2 | |
| Strata 0.1.4, tuned (`--vram-reserve-mib 1024 --pcie-frac 0.35 --pool-workers 8`), MTP | 79.1 | 83.3 | 76.8 | 74.9 | 1171 / 1098 / 989 |
| Strata, no drafts | 55.3 | 56.8 | 50.1 | 46.4 | |

Strata 0.1.6 (KV streaming) plus sampling (PR #19) was being validated in the background on
2026-09-27; the final table is in `bench/results/` once it is copied in:
- greedy: 77-84 tok/s at 250K;
- temperature 1.0 (top_p 0.95, top_k 20): 65-66 tok/s at 250K and about 75-79 at shorter
  depths, with 56-61% draft acceptance against 62-65% greedy.

## Where the time goes

- **Decode is memory traffic.**
  - Dense weights are about 3.5 GB, read on the GPU once per verify window: mixers,
    routers, the shared expert, hyper-connections (about 1.3 GB) and the head.
  - Routed experts are about 0.66 GB per token.
  - Misses come out of host DRAM whether the CPU computes them or PCIe copies them, so the
    wall is DRAM bandwidth (33.6 GB/s triad here) [Q/M].
- **Round time adds up.** Round ≈ mixer + max(CPU misses, GPU hits) + draft + overhead.
  The two halves wait for each other at every layer [Q].
- **MTP pays through shared dense reads, not expert reads.** The union of experts across
  draft tokens grows almost linearly [E from Strata's Table 5]. So MTP pays by amortising
  the dense reads and the per-layer syncs. That is why a llama.cpp MTP port only tied a
  larger cache.
- **The master lever is hit rate.** One GB of VRAM is about 700-725 Q2_0 expert blobs, for
  +1.5-2.5 points of hit rate and +3-5% tok/s near 5.5K slots [E].
- **Prefill is transfer-bound.** At a 2048-token chunk, (1 − 10/512)^2048 ≈ e^-40, so every
  non-resident expert streams on every chunk. Both engines reach 16-18.5 GB/s against a
  measured 20 GB/s host-to-device rate [M/E]. The lever is bigger chunks (4K-8K) on
  borrowed cache slots.
- **Ceilings [E]:**
  - about 140 tok/s with MTP and about 100 without, from the physics;
  - realistic: about 100-110 with MTP at 1K-32K and 85-95 at 250K;
  - prefill: 2,000-2,500 tok/s.

## Strata's ideas (from its paper, DETAILS.md and our analysis; clean-room: ideas only)

- **One graph per window.** A single CUDA graph per verify-window size covers all 48 layers
  plus the head. GPU→CPU handoff uses mapped-memory "doorbells" and in-graph spin-wait
  kernels, with no per-layer stream sync.
- **CPU misses run concurrently with GPU work.** The shared expert and cached experts run
  on the GPU while the CPU computes the misses. Each missed expert is split by rows across
  all cores, using an AVX-512 VBMI (unpack) and VNNI (dot) kernel over planar-repacked Q2_0
  blobs. A tuned share of misses goes over PCIe instead (0.35 is best on this box's slower
  RAM).
- **Adaptive cache.** Byte-sized slots, sized from free VRAM (about 5,000-5,800 here),
  pre-filled from a routing profile, then decayed-LFU swaps:
  - +1 per route, ×0.7 every 4 rounds;
  - admission threshold 2.0 and margin 1.5;
  - up to 96 swaps per round, asynchronous: evict now, admit when the copy completes.
- **Prefill borrows cache slots** for its buffers.
- **QSA indexer:** pooled block keys cached with an O(1) update per token; block-level
  top-k; split-K flash-decode reading the selected cells through the page table, with no
  gather.
- **MTP drafter:** fully in VRAM with a vocabulary trimmed to 40,525 tokens, about 2 ms per
  round. Multi-token CPU kernels read each missed expert once per window.
- **Exact sampling** with argmax drafts: sample each verify position independently, and
  accept while sample == draft.

## Tried and rejected on this box [M]

- **Fewer experts per token.** k=8 gave KLD 0.052 (PPL +3.4%) and k=7 gave KLD 0.097, for
  only +5-18% decode. It does not carry over from BF16 to 2-bit.
- **The CUDA graph cache-key fix** made no difference here: the expert-cache path disables
  CUDA graphs anyway.
- **In Strata:** `--pcie-frac` 0.1 or 0.5, `--spec 5`, min-p 0.3 and faster adaptation are
  all worse than the tuned flags.
- **From the literature, likely negative at batch 1:** speculative expert prefetch
  (ddvnguyen/llama.cpp#130, WiSP arXiv 2606.21868), KV offload (HiSparse and similar),
  FP4 for decode, and hot/cold-neuron methods (SwiGLU experts lack the ReLU sparsity they
  need).

## Promising but untested [E]

- **Cache-conditional routing** (arXiv 2412.00099): 30-50% fewer misses; needs a KLD gate.
- **Expert deferral** (KTransformers, SOSP 2025): overlaps CPU misses with the next layer;
  changes outputs.
- **Q4 KV with FWHT** (Strata PR #21): about 1,000 more slots.
- **A per-layer CPU/PCIe split** instead of a fixed fraction.
- **Huge pages** for the expert arena.
- **Expert-aware draft length** (EcoSpec, DraftExpert).
- **Prefill:** 8K chunks, a grouped int8 Q2_0 GEMM (Q2_0 values are exact in int8), and a
  tensor-core indexer.

## First experiments (P0), with the expected result

1. `membw` with 4 KiB vs 2 MiB pages, alone and with a concurrent host-to-device copy.
   Expect 38-45 GB/s alone and 31-38 concurrent.
2. `h2dbw`: is 20 GB/s really the host-to-device cap on Gen5? Expect 25-45 GB/s.
3. A routing-trace dump plus `tools/cache_sim.py`: hit rate vs slots and the gap to
   Belady.
4. Strata's own timing breakdown and a slot sweep, as a reference point.
5. llama.cpp `-ub 4096` and `-ub 8192` at 32K, to test the prefill-bound claim.

## Where the full material is

- **The research survey:** a separate private repository, branch `research/next-optimisations`,
  file `models/qwen3.8-flash-next-gsq-q2_0/next-optimisations.md`.
- **Vault notes** (Mac only, see `CLAUDE.local.md`): `Strata`, `PCIe Bandwidth as the MoE
  Offload Bottleneck`, `Flash-Next Offload Speedup Leads (2026-09)`, `Hierarchical KV
  Offload for Sparse Attention`, `Speculative Decoding with MoE`, `Training-to-Inference
  Expert Dropping`, `Predictive Expert Prefetching`.
- **Raw benchmark logs** on the box: `$BENCH` (windows 1-9).
