---
name: flashrt-cuda
description: >
  Writing, changing, testing or profiling CUDA kernels in flashrt (arch/qwen4exp/*.cu,
  kernels/cuda/*, tools/bench_*.cu) on the RTX 5080 (sm_120). Covers the hardware ceilings
  measured on this GPU, what CUDA-graph capture and determinism require of a kernel, the
  code idioms (toggles, ck, PDL launches, unit tests), where decode and prefill time can and
  cannot be won, and the gate sequence from build to a speed claim. Use before touching a
  kernel or proposing a kernel optimization in this repo.
---

# CUDA kernels in flashrt

Read `docs/engine.md` first for the kernel you are touching: its section says what the kernel
does, and the "Why" and "Tried and rejected" tables say what has been measured already. Do not
re-propose a rejected idea without new evidence that changes the reason it was rejected.

## The GPU (measured, not spec sheet, unless marked)

RTX 5080 16 GB, sm_120 (consumer Blackwell), 84 SMs, 64 MB L2, CUDA 12.9, driver 575.

| Ceiling | Value | Source |
|---|---|---|
| DRAM bandwidth | ~960 GB/s peak (spec); large mat-vecs reach 735-782 GB/s | p1-gemv README |
| int8 `mma.m16n8k32` | ~490 TOPS peak; ~283 TOPS with moe_q2's per-32 scale arithmetic | `bench_mma`, sweet-spots |
| fp16 / BF16 `mma` (fp32 accumulate) | ~122 TFLOPS | sw71 |
| TF32 | ~61 TFLOPS: no faster than the CUDA cores | sw70 |
| fp32 CUDA cores | ~56 TFLOPS | sw70 |
| Kernel launch inside a CUDA graph | ~1.4 µs per small kernel | sweet-spots |
| Programmatic dependent launch | saves ~0.25 µs per boundary | `bench_pdl` |
| Dynamic shared memory | the code opts in to 96 KB above 48 KB via `cudaFuncSetAttribute` | `blocks.cu` |

- sm_120 has warp-level `mma.sync` (int8 m16n8k32, fp16/bf16 m16n8k16). It does **not** have
  the datacenter Blackwell/Hopper paths (`wgmma`, `tcgen05`, tensor memory). Code or advice
  written for sm_90a/sm_100a does not apply.
- Sub-4-bit weights: decode arithmetic, not bytes, is the limit. Use int8 dp4a/mma with the
  activations quantized to int8; float activations are compute-bound on int-to-float
  conversion (sw10, sw14).
- **No hardware counters.** Performance counters are admin-only on the box, so Nsight Compute
  (`ncu`) does not work. Do not try to work around it (no sudo, no driver settings). Reason
  from a byte and op count against the ceilings above, time the kernel in isolation, and use
  `nsys` for the timeline.

## Where time can be won

**Decode is bound by host DRAM through the CPU misses, not by the GPU** (sw78, sw79).
- In a MoE layer with CPU misses, the GPU's hits and shared expert run while the host
  computes the misses; `k_moe_combine_db` then waits. A faster kernel inside that window only
  lengthens the wait. Moe-hits v2 and the combine fold gained nothing end to end.
- Wins come from work **outside** the miss window (hc mix, mixers, dense mat-vecs, LM head,
  attention) or from **fewer misses** (more cache slots: every MiB of VRAM is expert slots,
  about +0.3% speed per +1% of capacity).
- A decode step is hundreds of 2-6 µs kernels. **Fusing launches pays** (sw73-sw75: +4.3%,
  +2.7%); "PDL everywhere" does not.
- Freeing VRAM is a kernel win: Q3R v1 was faster per kernel and a net loss because its copy
  took expert-cache slots (sw10).

**Prefill is split between arithmetic and bandwidth** (sw80): dense MMQ ~22%, hc ~21% (at
bandwidth), routed experts ~19%, GDN ~9%, attention ~9%. `docs/sweet-spots.md` ranks the
untested paths; start there.

Before writing a kernel, write down its bytes and ops per call and the ceiling it would hit.
If the current kernel is already near that ceiling, the gain is not in the kernel.

## What a kernel must respect

- **CUDA graphs.** Decode runs as two captured graphs per token, verify windows as a graph
  pair per length. So on the decode and window paths:
  - launch arguments are fixed at capture: per-token values (token, position, seq) are read
    on the device from the pinned parameter block, never passed as arguments;
  - no allocation, host sync, `cudaMemcpy` to pageable memory or host branching on device data
    inside the captured region;
  - one-time setup (`cudaFuncSetAttribute`, lookup tables) happens before capture; the code
    uses a `static bool` guard at the launch site.
  - Grids must be valid at every position: e.g. graph-mode QSA always runs the selection path,
    and the select kernel falls back to dense below its width.
- **Determinism.** Runs are bit-reproducible since sw5. No float atomics anywhere: reduce in a
  fixed order (per-block partials, then an ordered sum). Integer atomics for counting are fine
  if the result's order does not reach float arithmetic.
- **Doorbell waits** (device-side spins on mapped host memory) are bounded with signed timer
  arithmetic and write an error word on timeout rather than trapping. Keep that pattern for
  any new device-side wait.
- **Rewinds.** Anything stateful in the forward (GDN state, conv history, PLE history) must be
  restorable by `ForwardRef::commit` after a partial accept. KV writes need nothing, because
  rejected positions are rewritten before being read.
- **Compile flags.** Only the vendored ggml target (`flashrt_gemv`) builds with
  `--use_fast_math`. flashrt's own kernels are precise; changing that is an output change and
  needs the KLD gate.

## Code idioms

- **Error checks:** each `.cu` has a file-local `ck(cudaError_t, const char* what)` that throws.
  After `<<<>>>` launches: `ck(cudaGetLastError(), "kernel name");`.
- **A/B toggles:** every new path gets a `FLASHRT_*` switch, default on once it wins, `0`
  restoring the old path, read once:
  ```cpp
  static const bool v2 = [] {   // FLASHRT_FOO=0: the old kernel
      const char* e = std::getenv("FLASHRT_FOO");
      return !(e && e[0] == '0');
  }();
  ```
  Keep the old path in the tree and add the toggle to the table in `docs/sweet-spots.md`.
- **PDL:** launch the dependent with `cudaLaunchKernelEx` and
  `cudaLaunchAttributeProgrammaticStreamSerialization`; the dependent calls
  `cudaGridDependencySynchronize()` before reading the producer's output (see `k_hc_up_mix2`
  in `blocks.cu`). Load independent data (weights) before the wait.
- **Comments** say what the kernel computes and why it is shaped that way, with the sw number
  that measured it. Match the density of the surrounding code.

## Unit test and microbenchmark

Each kernel family has `tests/test_<name>.cpp` (see `test_hc_decode.cpp`, `test_moe_hits.cpp`):
- random inputs at **qwen4exp's real shapes**, checked against a double-precision CPU
  reference, with a relative-L2 or max-error bound that is printed;
- the edge cases the engine hits (T = 1..4 for windows, the pad entry, tails not divisible by
  the tile);
- then a timing, with the weights **rotated through more than 256 MB of copies** so they come
  from DRAM and not the 64 MB L2, reported against the bytes the kernel must read (GB/s) or its
  ops (TOPS).

Register it in `CMakeLists.txt` with `add_test`. `test_gemm` and `test_moe_q2` need
`FLASHRT_TEST_MODEL=<gguf>`.

## From edit to a speed claim

Edit and commit on the Mac; build and measure on the GPU box (paths and box rules:
`CLAUDE.local.md`). In order:

1. **Build and ctest** on the box, outside any benchmark window (a build skews CPU-bound
   decode).
2. **KLD gate** if outputs can change (`fr_kld ... --fast`, about 5 minutes; add `--kv q8`,
   `--kv-hot 512` or `--window W` for the path touched). Measured bands: fast path 0.0087-0.0092,
   prefill 0.0082-0.0087, verify windows 0.0092. A shift outside the run-to-run spread is a
   failure even if small (sw57 rejected +0.0003).
   Tokens alone are not a check: a GPU hit and a CPU miss are not bit-identical.
3. **Kernel time in context:** `nsys profile --capture-range=cudaProfilerApi
   --cuda-graph-trace=node --trace=cuda build/fr_bench ... --gen 64`, then
   `nsys stats --report cuda_gpu_kern_sum --format csv`. Without `--cuda-graph-trace=node`,
   graphs show as single launches. Template: `bench/results/2026-09-28-sw76-profile/sw76.sh`.
4. **End-to-end A/B, same tokens:** `fr_bench --teacher`, toggle on against toggle off. At
   depth, 6+ windows (the generated text moves the hit rate by 20 points). Sampled drafts are
   the exception: measure them on sampled runs over 6+ windows.
5. **Server check** if VRAM budgeting or the engine changed: `bench/server_smoke.py` against a
   server on the current build (sw86 caught an out-of-memory `fr_bench` could not).
6. **Record:** `bench/results/<date>-swN-<topic>/` with the script (placeholders only: `$MODELS`,
   `$BENCH`, ...), logs and a README; run `bench/scrub.py` on it; update `docs/engine.md` (the
   section, the "Why" or "Rejected" table, current numbers) and `docs/sweet-spots.md`.

A kernel that is faster in its unit test and flat end to end is a result, not a failure: write
it into "Tried and rejected" with the reason, as sw78/sw79 did.

## Clean room

ggml, llama.cpp, ik_llama.cpp (MIT), vLLM, SGLang (Apache-2.0) and CUTLASS (BSD) may be read and
vendored with their notices (vendored ggml is in `third_party/ggml`). Strata's ideas are fine,
its source is not: never open Strata's `src/` or `tools/` while writing kernel code. See
`docs/clean-room.md`.
