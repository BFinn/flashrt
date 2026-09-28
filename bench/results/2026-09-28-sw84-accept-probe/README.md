# sw84: argmax drafts against speculative sampling, measured (2026-09-28)

**How drafts work now.** The head proposes its argmax, and the verifier keeps it when its
sampled token equals it. At temperature 1.0 a draft is therefore accepted with p(argmax q):
p is the target's sampling distribution (top-k 20, top-p 0.95), q the head's.

**The alternative.** Speculative sampling (Leviathan et al., Chen et al., 2023) draws the draft
from q and accepts it with min(1, p/q), else resamples from max(0, p - q). It is equally exact in
distribution and accepts with probability sum_x min(p(x), q(x)).

**The probe.** `fr_bench --accept-probe` computes both for every round's first draft from the real
logits (the head's through the same sampler chain, or its plain softmax). Setup: `--spec 1` at
P2 conditions (temperature 1.0, top-k 20, top-p 0.95, seed 1), from the saved states, 2 windows
of 128 tokens, sampled text.

| Arm | rounds | argmax draft (now) | speculative sampling, head softmax | **with the head through the sampler chain** | target's own top-token p |
|---|---|---|---|---|---|
| 32K, head over 32,768 ranked + prompt tokens | 181 | 0.475 | 0.626 | **0.657** | 0.572 |
| 245K, same head | 163 | 0.577 | 0.676 | **0.712** | 0.667 |
| 32K, full-vocabulary head | 171 | 0.519 | 0.657 | **0.691** | 0.617 |

- The measured acceptance of these runs (42.0%, 57.7%, 50.3%) matches the argmax column within
  sampling noise.
- **Speculative sampling would accept 13-18 points more** (a third more at 32K).
- It would even beat a perfect argmax drafter, which is bounded by the target's own top-token
  probability.
- **Passing the head's logits through the sampler's top-k/top-p** matches p better than its raw
  softmax (+3 points).
- **The trimmed vocabulary** costs little here: the runs sample different text, so the 32K rows
  differ by trajectory as well.

**Rough throughput estimate** (not measured). From sw78's teacher-forced round (draft ~1.0,
verify ~11.6, commit 0.26 ms) and plain step (9.4 ms):
- **32K:** 1.48 → 1.66 tokens per round, about 114 → 128 tok/s (+12%).
- **245K:** about +15%.
- **A second draft** (1 + a + a^2 ≈ 2.1 tokens for a wider, costlier window) still does not pay
  at 32K.

The speeds in these logs are not comparable: the probe copies both logits rows to the host every
round.
