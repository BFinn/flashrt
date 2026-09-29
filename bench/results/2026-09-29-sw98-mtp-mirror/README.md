# sw98: the MTP head's prompt pass on tensor cores (2026-09-29)

P-1 in `docs/improvement-plan.md`, first step. With the MTP head, prefill at depth ran at 2,150-
2,430 tok/s against 5,690-5,955 without it (sw91). The head catches up on the prompt in batches of
64 rows. In hot-set mode (the default `--kv-hot 4096`) its attention read the KV through the hot
set with the per-token kernel, because the tensor-core attention needs a VRAM copy of the KV. The
target's chunks build one (`qsa_mirror_begin`); the head had none, so its cost grew with depth.

**The change:** the head gets a mirror of its one QSA layer's KV for the length of a chunked prefill
(`MtpHead::prefill_begin/prefill_end`, about 270 MiB at 250K, out of the VRAM the expert cache
lends to the chunks). Its 64-row batches then take the tensor-core attention and indexer.
`FLASHRT_MTP_MIRROR=0` restores the old path.

**A/B** (`sw98.sh`, 2 runs per arm, alternating): a cold 131,072-token wikitext prompt with the head
(`--spec 2`), then 256 tokens at temperature 1.0, seed 1 (`engine_smoke.py`, request r1):

| Head's attention in the prompt pass | Prompt | Prefill | Drafts accepted | Decode |
|---|---|---|---|---|
| through the hot set (`FLASHRT_MTP_MIRROR=0`) | 35.3, 35.4 s | 3,710 tok/s | 143 of 224 (63.8%) | 98.2, 99.6 tok/s |
| tensor cores, from the mirror | **25.8, 25.8 s** | **5,080 tok/s** | 151 of 210 (71.9%) | 102.2, 101.4 tok/s |

- **The prompt time falls by 27%.** The target alone prefills about 5,700-5,950 tok/s, so the head
  now adds about 3.3 s at 131K where it added 12.8 s.
- **The head's outputs change** (another attention kernel), so its drafts do too. Acceptance did not
  fall: one run of 256 tokens per arm, and each arm repeats exactly (same seed), so this is one
  sample, not a measured gain. Each arm's first tokens are the same (sampled from the target).
- **The target is untouched:** the mirror is the head's alone.
- **Correctness:** `engine_smoke.py --faults` with the head passes with the mirror in place, and a
  failure mid-prefill frees it (sw95b, `faults-mtp-b.txt`). `--reuse` passes bit-exact (sw95f).

What remains of P-1: the head's MoE still runs as 8-token mat-vec slices, and its batches are
64 rows. A chunk-sized pass (its own chunk buffers, grouped expert GEMMs) is the next step.
