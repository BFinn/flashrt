# sw92: bad and failing requests leave the engine serving (2026-09-29)

Phase 1 of `docs/improvement-plan.md` (E-1, E-2). `bench/engine_smoke.py --faults` against the
real model (`sw92.sh`): the engine runs with `FLASHRT_TEST_HOOKS=1`, `--ctx 32768` and
`--prefill-chunk 512`. The reference request A is 2,048 wikitext tokens, greedy, 32 generated.
There are two arms:
- `mtp`: `--spec 2`, so the decode fault lands in a verify window;
- `plain`: no MTP head, so it lands in a plain decode step.

**Result: every check passes in both arms** (`mtp.txt`, `plain.txt`; `sw92.out`: rc 0 and 0).

| Case | The request | The next request (A again) |
|---|---|---|
| prompt of ctx + 1 tokens | error "prompt longer than the context" | reused 2,047: nothing changed |
| token id 10^9 in the prompt | error "token id out of range" | reused 2,047 |
| top_k 65 / temperature -1 / top_p 0 | error naming the limit | reused 2,047 |
| fault after the first prefill chunk | error, reported | cold (reused 0), succeeds |
| fault in the first verify window / decode step | error, reported | cold, succeeds |
| stop during the prefill | done, "cancelled" | cold, succeeds |
| another text (negative control) | | fails the state check (a different top token) |

Before the fix, a failure mid-prefill left the expert cache released and the sequence
half-advanced. A failure in a verify window left the window open, so the next
`forward_window` threw.

**How the state is compared.** Each follow-up must have the reference's top token at the first
generated position, and a KL divergence over the reference's top 8 of at most 0.001. The engine
reports those logits through a test-only field, `debug_first_top`. Across the 16 follow-ups, KL
was 0.00000-0.00017. One cold follow-up in the plain arm reproduced the reference exactly: KL 0,
relative logits 0.0000.

**Why not compare tokens.** The first attempt (`first/`) required the first 8 greedy tokens to
match the reference. Every follow-up failed that, including those after invalid requests, where
nothing had changed:
- the plain arm shared 4 tokens (14 after one cold restart);
- the MTP arm shared 1.

The expert cache adapts during every decode. A GPU hit and a CPU miss differ in the last bits
(`docs/engine.md`, Correctness machinery), so greedy tokens depend on the cache's content. The
logits show the same thing (`second/`, `third/`): the logits of the less likely candidates,
taken relative to the top one, moved by up to 0.8 between runs, while KL stayed below 0.0002.

**Limits of the check.** It catches a broken state, not a subtle one. This prompt's first
position is a confident prediction (top logit 22.8, the next 15.3). Changing one token 2,000
positions back moved KL by only 0.00003-0.00004 (`third/`, "sensitivity"), which is within the
noise. The negative control, a different text, changes the top token.

Also in this change, recorded in `docs/design.md` (Engine protocol):
- **Strict requests:** requests are checked before anything runs. `seed` must be a whole number
  up to 2^53. Non-numeric prompt ids are errors.
- **Bounded host wait:** the host-side doorbell wait gives up after 10 s (E-3).
- **Fatal failures:** after a failure the process cannot recover from (a doorbell timeout, a
  sticky CUDA error), the error says "(fatal: the engine exits)" and the engine exits with
  status 3.
