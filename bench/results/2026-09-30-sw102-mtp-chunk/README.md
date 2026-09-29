# sw102: the MTP head's prompt pass in chunk calls with grouped expert GEMMs (2026-09-30)

P-1 in `docs/improvement-plan.md`, step 2 (step 1 was the head's KV mirror, sw98). The head caught up
on a chunked prompt in batches of 64 rows, its MoE as 8-row mat-vec slices: for a 16K chunk that
was 2,048 slices, each reading up to 80 experts. Now, for the length of a chunked prefill, the head
has a second buffer set (about 450 MiB, out of the VRAM the expert cache lends to the chunks). It
catches up in calls of up to 1,024 rows, and its MoE runs as grouped expert GEMMs over each whole
call: `gemm::moe_prepare` / `moe_run`, the ggml MMQ path, over the head's Q2_0 experts in VRAM. The
decode set, on which the draft chain's graphs are captured, is untouched. `FLASHRT_MTP_CHUNK=0`
restores the batches.

## A/B at 131K (`sw102.sh`, as sw98)

A cold 131,072-token wikitext prompt with the head (`--spec 2`), then 256 tokens at temperature 1.0,
seed 1. Two runs per arm, alternating; each arm repeats exactly.

| Head's prompt pass | Prompt | Drafts accepted | Decode |
|---|---:|---:|---:|
| batches of 64, 8-row slices (sw98's path) | 25.8, 25.8 s | 146 of 218 (67.0%) | 106.0, 106.1 tok/s |
| **calls of 1,024, grouped GEMMs** | **23.6, 23.6 s** | 155 of 200 (77.5%) | 108.7, 108.5 tok/s |

The target alone prefills this prompt in about 22.5 s, so the head's share falls from about 3.3 s
to about 1.1 s. It was about 12.8 s before sw98.

## Checks

- `engine_smoke.py --reuse` with the head: every case bit-exact (KL 0.00000), the negative control
  detected (`reuse-mtp.txt`).
- `--faults` with the head: 0 checks failed. A failure mid-prefill frees the chunk buffers with the
  mirror (`faults-mtp.txt`).

## Window 9's protocol (MTP greedy, n = 3, against sw101; `sw102-summary.txt`)

| | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| prefill of the new tokens, sw101 → sw102 (tok/s) | 805 → 841 | 4,999 → **5,405** | 5,031 → **5,424** | 4,528 → **4,856** |
| prompt time, s | 1.31 → 1.25 | 6.56 → 6.07 | 20.13 → 18.67 | 25.77 → 24.03 |
| decode, tok/s | 112.1 → 109.2 | 87.0 → 87.3 | 83.7 → 83.1 | 74.6 → 81.8 |
| draft acceptance | 54.9 → 54.9% | 56.1 → 54.9% | 61.3 → 54.6% | 53.5 → 58.8% |
| expert-cache hit rate | 80.2 → 80.2% | 68.4 → 67.4% | 69.0 → 70.0% | 70.2 → 70.3% |

- **Prefill with the head: +7-8% at every depth.** Without the head the target prefills 5,662 /
  5,756 / 5,232 tok/s here (sw96), so the head now costs 5-7%. The first version cost 27-60%.
- **Decode moves with the drafts, not systematically.** The head's arithmetic in the prompt pass
  changed (grouped MMQ instead of mat-vec kernels), so it drafts other tokens. Over the four depths
  acceptance is unchanged: 56.5% before, 55.8% now. 250K's +9.6% comes from its higher acceptance
  there, and 134K's acceptance fell with its speed flat: both are one sample of different drafts.
  The hit rates are the same.
