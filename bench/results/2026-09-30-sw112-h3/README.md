# sw112: phase 4 (H-3, H-1, H-2, H-7), each proven output-neutral (2026-09-30)

Phase 4, H-3 of `docs/improvement-plan.md`. Every `FLASHRT_*` A/B switch whose comparison is
settled goes; the winning path of each stays, and the results folders stay as the record.

**Removed (19):** `GDN_CHUNK`, `GDN_COL`, `ROUTE_WARP`, `Q3_Q8`, `MOE_Q2MMA`, `MOE_YD16`,
`MOE_GU_STAGES`, `MOE_J`, `HC_GATE16`, `HC_DOWN2`, `HC_UP2`, `HC_COMB2`, `LINEAR_MULTI`,
`FUSE_EPI`, `DB_SKIP`, `ATTN_TC`, `IDX_TC`, `MTP_CHUNK`, `MTP_MIRROR`.
**Kept:** `MOE_AB64` (off: faster but fails KLD), `HC_Q8` (hc matrices as Q8P), `ARGMAX_DRAFTS`.

Code only a switch reached is deleted: the block-per-token routing kernel, the prefill expert
path that converted each streamed slice to ggml's layout for MMQ (with its converted-slice and
intermediate buffers), the float per-slot expert outputs, and the 3/4-stage
gate/up variants. Kernels that other shapes still reach (the GDN column kernel for 16-63 tokens,
the general-shape hc kernels, the plain mat-vecs) stay. 11 files, +69 / −348 lines.

## The check

Runs are bit-reproducible (sw107, sw109), so a neutral refactor must reproduce every value exactly.
`fingerprint.sh` records, for one build:
- KLD against the llama.cpp base (8K context, 2 chunks): the fast path, verify windows of 3 with
  the hot-set KV, and 1,024-token prefill chunks: mean, percentiles, same-top rate, swap count;
- `fr_bench --teacher` on window 9's 32K prompt, plain and with the head (`--spec 2`): hit rate,
  swaps, acceptance, and an md5 of the token line;
- a sampled run (t = 1.0, seed 7) and a greedy run on wikitext.

`before` ran on 88791c9, `after` on d8e8b51. Both are in this folder
(`sw112-before.txt`, `sw112-after.txt`, raw logs under `before/` and `after/`).

| Check | Result |
|---|---|
| Fingerprint, 7 runs | **identical** (`diff` empty): KLD fast 0.008996, win3 0.008913, chunk 0.008879; swaps 159,480 / 100,258; w9 teacher hit rate 82.09% (plain) and 72.19% (head), same tokens |
| ctest | 19 / 19 |
| `engine_smoke --reuse` with the head (N = 9,000, chunks 2,048) | 5 / 5 PASS, every restore KL 0.00000 |
| `engine_smoke --faults` with the head | 18 PASS, 0 failed |
| `server_smoke.py` on the new build | 15 / 15 ok |

No speed was measured: the paths that run are the same code as before.

## H-1: blocks.cu split (1e2eceb)

The same checks on phase 4's next step: `blocks.cu` (3,690 lines) split by block into `hc.cu`,
`gdn.cu`, `qsa.cu`, `moe_ref.cu` (batch routing and the reference MoE) and `ple.cu`, with
scratch, the linear layers, the embedding, argmax and the head left in `blocks.cu` and
`ck`, `block_sum` and `hist_rewind` in the internal `blocks_common.cuh`. The code moved as is;
PLE's three norms now call `rms_norm_rows` and its rewind `hist_rewind` (the same launches).

| Check | Result |
|---|---|
| Fingerprint, 7 runs | **identical** to `before` (`sw112-h1.txt`, raw logs under `h1/`) |
| ctest | 19 / 19 |
| `engine_smoke --reuse` / `--faults` with the head | 5 / 5 and 18 PASS |
| `server_smoke.py` | 15 / 15 |

Outputs of this second round are in `smoke-h1/`.

## H-2 and H-7: Forward, and graphs invalidated by a version (ff92e3d)

`ForwardRef` is now `Forward` (`forward.cu`, `forward.hpp`); `forward()` calls `begin_chunk`,
`fetch_ple_rows`, `run_graphs` or `run_eager`, and `end_step`, in the old order. A captured graph
was stale when any of six scratch addresses differed (`same_buffers`); now `BlockScratch::version`
takes a new value whenever `alloc_block_scratch` or `qsa_scratch_reserve` (re)allocates a buffer a
graph bakes in, and graphs (the target's and the MTP chain graph) compare versions.

| Check | Result |
|---|---|
| Fingerprint, 7 runs | **identical** to `before` (`sw112-h2.txt`, raw logs under `h2/`) |
| Graph captures per run | unchanged (1 in each fr_bench run) |
| ctest | 19 / 19 |
| `engine_smoke --reuse` / `--faults` with the head | 5 / 5 and 18 PASS |
| `server_smoke.py` | 15 / 15 |

Outputs in `smoke-h2/`.

## Files

- `fingerprint.sh LABEL`: the fingerprint (placeholders: `$MODELS`, `$BENCH`, `$FLASHRT`).
- `smoke.sh`: the engine and server checks.
- `smoke/`: their outputs.
