# sw91: flashrt on window 9's protocol with swap budget 32 (2026-09-29)

As sw87 (`sw91.sh`): window 9's token ids, one growing conversation, 384 tokens per depth without
stopping at end-of-sequence, a fresh engine per run, arms interleaved, 3 runs each. The one change
is the engine's new default expert-cache swap budget, 32 (sw89, sw90). There is no cache prior.

Decode tok/s, mean ± sd (`sw91-summary.txt`, from window 9's own `w9_summary.py`), window 9's
reference rows alongside:

| Arm | 1K | 32K | 134K | 250K |
|---|---|---|---|---|
| llama.cpp, expert cache, no MTP (w9 L, n=4) | 37.4 ± 0.6 | 37.2 ± 1.1 | 32.8 ± 1.6 | 30.7 ± 1.1 |
| **greedy** | | | | |
| Strata 0.1.6, MTP (w9 G, n=4) | 87.0 ± 0.7 | **96.0 ± 2.2** | **85.0 ± 2.7** | **80.4 ± 3.6** |
| flashrt, `--spec 2` (G, n=3) | **106.6 ± 1.5** | 83.7 ± 0.8 | 81.0 ± 0.4 | 74.8 ± 1.7 |
| flashrt, no MTP (P, n=3) | 94.7 ± 1.7 | 81.7 ± 1.0 | 77.4 ± 0.7 | 73.0 ± 1.2 |
| **temperature 1.0, top_p 0.95, top_k 20** | | | | |
| Strata 0.1.6, MTP (w9 S, n=4) | 80.5 ± 4.1 | 79.1 ± 1.3 | 73.9 ± 1.8 | 69.9 ± 7.2 |
| flashrt, `--spec 2`, sampled drafts (S, n=3) | **82.2 ± 4.2** | 77.7 ± 13.1 | **81.0 ± 0.9** | **76.4 ± 2.1** |

**Correction (2026-09-29, after an external review).** "One growing conversation" held for
the reference engines only. Strata checkpoints during prefill, and reused 32,768 tokens at 134K and 131,072 at 250K
(its logs in `2026-09-27-w9-validation`, "prompt ... tokens = N reused + M read"). llama.cpp
reused about 30,700 and 132,000. Only flashrt re-prefilled every depth: it keeps one checkpoint,
at the end of the previous prompt. So Strata's and llama.cpp's expert caches carried over from
the previous answer, while flashrt's was primed from the whole prompt. How much of the gap at
depth this explains is not measured yet (`docs/improvement-plan.md`, phase 2).

**Caveat on Strata's numbers (added 2026-09-29, after an external review).** With
`--expert-cache` on, Strata 0.1.6 logs a warning that its GPU hit path "is NOT CORRECT": its
tokens diverge from a cache-off run. Its timings are real, but its draft acceptance, and so its
MTP speed, come from outputs that differ from the model's.

**Summary:**
- **Against llama.cpp:** flashrt is 2.2-2.5x without MTP, 2.2-2.9x with it.
- **Against Strata, greedy:** flashrt is ahead at 1K (+23%) and behind at 32K-250K (-5% to -13%).
- **Against Strata at temperature 1.0:** flashrt is level at 1K and 32K (the 32K mean has one
  slow run), and ahead at 134K (+10%). At 250K (+9%) the gap is within Strata's spread (±7.2).
- **The swap budget** recovered 5-12 points over sw87.

The remaining gap at depth has two candidate causes, not yet separated: the expert-cache warm-up
on this prompt (sw88: the prompt's routing predicts the generation's poorly), and prefix reuse,
which Strata and llama.cpp had and flashrt did not (see sw87's correction). flashrt's measurements on wikitext continuation, where the
prompt predicts the generation well, run 30-45% faster (fr_bench).

Prefill:
- **Without the MTP head,** by the engine's own `prompt_ms`: 5,570 / 5,950 / 5,690 tok/s at
  32K / 134K / 250K.
- **With the head,** 4,000 / 2,430 / 2,150: the head runs over the whole prompt on a slower
  path.
- **By window 9's measure** (depth increase over the time to the first token): with the head,
  flashrt 3,990 / 1,833 / 1,000 against Strata 1,134 / 1,089 / 945; without it, 5,565 / 4,493 /
  2,647.
- **Making the head's prompt pass as fast as the target's** is a clear open item.
