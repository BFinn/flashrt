# sw87: flashrt on the reference runs' protocol (2026-09-29)

Until now flashrt's numbers came from `fr_bench`, and the reference engines' from their own drivers
(window 9: `bench/results/2026-09-27-w9-validation`). This run puts flashrt on window 9's protocol
(`bench/flashrt_depthbench.py`, `sw87.sh`):
- the same token ids (`strata-ids.json`: 1,055 / 32,793 / 134,029 / 250,712 tokens);
- one growing conversation; 384 generated tokens per depth, not stopping at end-of-sequence;
- a fresh engine per run, arms interleaved, 3 runs each;
- the same summarizer (`w9_summary.py`: decode by wall clock between the first and last token;
  prefill as the depth increase over the time to the first token).

**The prompt.** Window 9's prompt is synthetic: six sentences repeated in random order, then "Continue
the text above with a detailed, new paragraph about how these records are audited." The model
answers with a thinking block and an essay. None of the engines reused a prefix: each prompt ends
with that instruction, so a longer prompt is not a continuation of a shorter one. llama.cpp reused
part of the prompt at the two deepest depths.

**Correction (2026-09-29, after an external review).** The paragraph above is wrong about
Strata. Strata checkpoints during prefill, and reused 32,768 tokens at 134K and 131,072 at 250K
(its logs in `2026-09-27-w9-validation`, "prompt ... tokens = N reused + M read"). llama.cpp
reused about 30,700 and 132,000. Only flashrt re-prefilled every depth: it keeps one checkpoint,
at the end of the previous prompt. So Strata's and llama.cpp's expert caches carried over from
the previous answer, while flashrt's was primed from the whole prompt. How much of the gap at
depth this explains is not measured yet (`docs/improvement-plan.md`, phase 2).

**Caveat on Strata's numbers (added 2026-09-29, after an external review).** With
`--expert-cache` on, Strata 0.1.6 logs a warning that its GPU hit path "is NOT CORRECT": its
tokens diverge from a cache-off run. Its timings are real, but its draft acceptance, and so its
MTP speed, come from outputs that differ from the model's.

Decode tok/s, mean ± sd (`sw87-summary.txt`; window 9's rows for comparison):

| Arm | 1K | 32K | 134K | 250K |
|---|---|---|---|---|
| llama.cpp, expert cache, no MTP (w9 L, n=4) | 37.4 ± 0.6 | 37.2 ± 1.1 | 32.8 ± 1.6 | 30.7 ± 1.1 |
| Strata 0.1.6, MTP, greedy (w9 G, n=4) | 87.0 ± 0.7 | **96.0 ± 2.2** | **85.0 ± 2.7** | **80.4 ± 3.6** |
| Strata 0.1.6, MTP, temperature 1.0 (w9 S, n=4) | 80.5 ± 4.1 | 79.1 ± 1.3 | 73.9 ± 1.8 | 69.9 ± 7.2 |
| flashrt, no MTP, greedy (P, n=3) | **91.8 ± 1.5** | 73.0 ± 0.9 | 70.0 ± 0.5 | 67.1 ± 0.4 |
| flashrt, `--spec 2`, greedy (G, n=3) | **95.7 ± 0.7** | 70.3 ± 0.8 | 70.4 ± 0.0 | 63.8 ± 0.4 |
| flashrt, `--spec 2`, temperature 1.0, sampled drafts (S, n=3) | **91.5 ± 2.1** | 65.3 ± 8.6 | 69.4 ± 0.8 | 68.1 ± 4.9 |

Prefill of the full prompt (engine `prompt_ms`):
- flashrt without the head: 5,770 / 5,960 / 5,700 tok/s at 32K / 134K / 250K;
- flashrt with the MTP head: 4,130 / 2,430 / 2,150, because the head runs over the whole prompt
  too, on a slower path;
- Strata: 1,173 / 1,442 / 2,031 by the same measure.

**On this protocol flashrt is 2.2-2.5x llama.cpp, and ahead of Strata only at 1K.** At 32K-250K
it is 15-25% behind Strata greedy, and speculation gains nothing. That contradicts the
fr_bench numbers (32K plain ~107, `--spec 1` 143): sw88 finds why, and sw89 recovers part of it.
