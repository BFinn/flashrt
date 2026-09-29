# sw95: host checkpoints during the prefill (2026-09-29)

Phase 2 of `docs/improvement-plan.md`, R-1. qwen4exp's recurrent state (GDN states and conv
histories, the PLE history, the indexer rings) cannot be rewound, so the engine could reuse a
previous prompt only up to its one checkpoint, taken at the end of that prompt. Window 9's
prompts keep a 19-token instruction after the grown text, so each deeper prompt diverged before
that checkpoint and flashrt re-prefilled everything, while Strata and llama.cpp reused the shared
prefix (sw91).

**The change** (`engine/session.cpp`):
- Up to `--ckpts` (8) checkpoints of the target's and the MTP head's recurrent state, in pinned
  host RAM, 112.7 MiB each. They are taken during a chunked prefill at chunk ends at least
  `--ckpt-interval` (4,096) tokens apart, and before the prompt's last `--ckpt-tail` (64) tokens,
  which then run as one batch. The device checkpoint at the prompt's end stays.
- A prompt restores the latest checkpoint, host or device, inside the prefix it shares with the
  previous sequence.
- When the ring is full, the entry whose removal leaves the smallest gap goes, so the positions
  thin out evenly over the prefix and the newest stays.
- Every request drops the checkpoints past the shared prefix. Before this change, a reset left
  the device checkpoint in place. After a prefill cancelled past it, a later prompt sharing that
  far could restore another text's state.

## Correctness: `engine_smoke.py --reuse` (with and without the MTP head)

Each case runs cold first (after an unrelated request) as its reference, then after the prompt it
should reuse from. The reused run must match the reference's state at the first generated
position: the same top token, KL over the reference's top 8 at most 0.001 (as sw92).
`--n 9000 --prefill-chunk 2048 --ckpt-interval 2048`, `--ctx 32768`.

| Case | Reused | Prompt time (cold) | KL, MTP arm / plain arm |
|---|---|---|---|
| tail: A with text inserted before its last 19 tokens (window 9's shape) | 8,935 of 11,250: the tail checkpoint | 2.31 s (5.49 s) | 0.00010 / 0.00005 |
| middle: A with a token changed at 4,500 | 4,096 | 3.14 s (4.68 s) | 0.00010 / 0.00000 |
| cancelled: A, then another text stopped at 9,016+, then 9,008 of it plus other tokens | 8,192: its own chunk-end checkpoint, not A's end checkpoint at 8,999 | 1.58 s (4.69 s) | 0.00012 / 0.00012 |
| negative control: another text | | | detected (a different top token) |

`engine_smoke.py --faults` (phase 1's checks) passes again in both arms, 17 of 17 each
(`faults-*.txt`).

In the default mode, `engine_smoke.py`'s request 4 (request 1's prompt again, after two requests
that extended it) now reuses 32,703 of 32,768 tokens through the tail checkpoint: 0.6 s instead of
a 7.8 s cold prefill (`tail64-*.txt`).

## The tail's cost

A cold 32,768-token prompt with the MTP head (`engine_smoke.py`, default mode, 3 runs per arm,
alternating):

| `--ckpt-tail` | Cold prompt (r1) | The same prompt after two extensions (r4) |
|---|---|---|
| 64 (the tail as a batch) | 7.8, 7.8, 7.8 s | reused 32,703: 0.6, 0.7, 0.6 s |
| 0 | 7.2, 7.2, 7.1 s | reused 16,384 (a chunk end): 4.0, 4.0, 4.0 s |

The 64-token batch costs 0.6 s: batches run the reference path, which computes every routed
expert on the CPU from host DRAM. That is 8% of every long prefill, too much to pay when the
pattern it serves is specific to some workloads. **So the tail is adaptive** (`sw95b.sh`, rerun
of the checks): the engine takes the tail checkpoint only after a prompt has diverged from the
previous one within the previous prompt's last `--ckpt-tail` tokens (a fixed tail after grown
text). It then uses that tail's length, rounded up to 8: window 9's 19-token instruction gives
a 24-token batch. The `--reuse` test's tail case first shows the engine that pattern.

## A sharper check, and what it found (sw95b-sw95f)

The first check allowed KL 0.001 at the first generated position. Making it tighter changed the
test and settled what a restore guarantees:

1. **sw95b-c:** with the adaptive tail, the tail case failed at KL 0.0021-0.0031. The cold
   references had run before the engine saw the fixed-tail pattern, so they had no tail batch;
   the test now sets the pattern up first. It still failed.
2. **The first token itself was noisy.** It ran on the decode fast path, where GPU hits and CPU
   misses differ in the last bits and depend on the cache's content. Under the test hook the
   prompt's last token now runs on the reference path (every expert on the CPU), so the
   logits depend on the state alone (sw95d). The unchanged-state follow-ups of `--faults` now
   measure exactly 0.
3. **Restores on the cold run's chunk grid are bit-exact** (sw95d, sw95e): KL 0.00000, relative
   logits 0.0000.
4. **Off the grid they differ by rounding, not state.** A restore at 8,960 moves the following
   chunks relative to a cold run's. The chunked GDN's 512-token slabs and the dense MMQ's split of
   the K sum then round differently. It measured KL 0.16 at one position (0.008 with
   `FLASHRT_GDN_CHUNK=0`); two cold runs with chunks of 2,048 and 1,792 differ by 0.0077 there
   (`segmentation.jsonl`).
5. **Batches and chunks round differently too** (sw95e): 96 new tokens after a restore ran as
   batches, while the cold run chunked them, and a near-tie flipped (the top two tokens 0.35 logits
   apart; chunked and batched cold runs differ by KL 0.008 there, `e2_paths.txt`).
6. **sw95f:** with restores on the grid and the same execution path, **every case is bit-exact in
   both arms**, and the negative control is detected.

So a restore reproduces the state bit for bit. What varies is the prefill's segmentation, which
any change of chunk length varies too.

## Files

`sw95.sh`..`sw95f.sh`; `reuse-*.txt`, `faults-*.txt` with their engine logs; `tail*.txt`; `first_top.py` and
`segmentation.jsonl` (the control); `e2_paths.py` and `e2_paths.txt` (chunks against batches); `mtp-convert.txt`.
