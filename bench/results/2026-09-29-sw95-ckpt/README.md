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

SW95_TAIL

## Files

`sw95.sh`; `reuse-{mtp,plain}.txt`, `faults-{mtp,plain}.txt` with their engine logs; `tail*.txt`.
