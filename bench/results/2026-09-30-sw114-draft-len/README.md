# sw114: a per-round draft length (P-3), measured and simulated: no rule worth adopting (2026-09-30)

**Question.** `--spec K` is fixed per run (2 by default). Could each round choose its K from the
head's confidence, or from how the last round went, and gain speed? The plan's P-3; the vault's
MoE speculation notes report a verify cost from 2.4x down to 1.5x of a step with confidence
gating on offloaded MoEs (DraftExpert).

**Data** (`sw114.sh`). `fr_bench --round-log` (new) writes every speculative round: drafts
verified and kept, draft and verify ms, and each draft's probability under the head (the top
token's softmax for argmax drafts; q of the drawn draft for sampled ones; `MtpHead::draft_probs`).
`--spec 1, 2, 3` and a plain run, greedy and temperature 1.0 (sampled drafts), 6 windows of 128
tokens each, from the saved 32K and 245K wikitext states and from window 9's 32K prompt. Logs in
`logs/`.

**Costs** (the model `simulate.py` fits). The draft phase is about 0.2 + 0.9-1.2 ms per draft. The
verify phase per window length T:

| verify ms, T = 1 / 2 / 3 / 4 | greedy | temperature 1.0 |
|---|---|---|
| 32K wikitext | 9.9 / 11.9 / 13.7 / 15.9 | 9.4 / 11.3 / 13.8 / 15.1 |
| 245K wikitext | 12.2 / 17.3 / 21.4 / 26.5 | 12.0 / 15.4 / 21.0 / 23.9 |
| window 9, 32K | 9.1 / 13.0 / 15.8 / 19.7 | 9.4 / 14.2 / 15.9 / 19.8 |

The first extra verified token costs about 20% of a one-token step at 32K wikitext and 40-45% at
245K and on window 9 (the union of the window's CPU misses).

**Calibration** (`simulate.out`). For argmax drafts the head's top probability is well calibrated:
at 32K a first draft is kept 27% of the time below 0.2 and 96% above 0.9, and the same holds for
drafts 2 and 3. For sampled drafts q(d) predicts little (49% kept below 0.2 at 32K), as expected:
speculative sampling keeps a draft with probability min(1, p/q), so a low-q draft is not a bad one.

**Simulation.** The spec 3 rounds replayed under a rule: a round verifies a prefix of its drafts
(acceptance of a prefix does not depend on later drafts), tokens = kept + 1, time = drafts made
plus the verify phase for that length. Against the best fixed K per context:

| tok/s (draft + verify) | best fixed K | best rule for that context | oracle |
|---|---|---|---|
| 32K greedy | 130.6 (K=2) | 134.7 (stop below 0.3, K_max 3) | 158.5 |
| 32K t = 1 | 136.8 (K=1) | 135.1 | 165.4 |
| 245K greedy | 120.0 (K=3) | 123.6 (stop below 0.5, K_max 3) | 128.2 |
| 245K t = 1 | 105.6 (K=1) | 106.0 | 124.5 |
| window 9 greedy | 141.0 (K=2) | 141.8 (stop below 0.3, K_max 2) | 160.5 |
| window 9 t = 1 | 122.8 (K=2) | 128.3 (stop below 0.6, K_max 3) | 148.6 |

The oracle knows which drafts will be kept: 7-21% over the best fixed K. The t = 1 rows' rules
threshold q(d), which is not exact for sampled drafts (below); they are shown as an upper bound.

**One rule for all contexts** (`rules.out`, relative to fixed K = 2):
- the best probability threshold (stop below 0.3, K_max 2): +1.6% geometric mean, −0.8% to +4.0%;
- K from the previous round's kept count (acceptance comes in streaks: after a full accept the
  next round keeps 1.4-2.8 drafts on average, after none 1.1-1.6), best of 2,187 maps plus the
  threshold: +2.1%, −4.0% to +6.9%;
- next K = kept + 1 without a threshold (the only kind allowed for sampled drafts, below): +0.6%,
  −2.1% to +2.7%.

These are maxima over many candidates, and the model misses the measured runs by up to 8% (each
run samples its own text: spec 3's rounds replayed at K = 2 give 2.06 tokens per round, the spec 2
run itself 2.24).

**Exactness constrains the sampled case.** With sampled drafts, choosing whether to verify draft
j from its own value (q(d_j)) biases the output: speculative sampling reproduces p only when summed
over every draft q can produce. A sampled-draft rule may use only what is fixed before the draft is
drawn (q's shape, earlier rounds). Argmax drafts have no such constraint.

**Conclusion.** Not adopted: at most about 2% in simulation, not consistent across contexts, and
below the simulator's error. Fixed K is near the best that the head's probability or the last
round can predict; the oracle's headroom needs a better predictor (target-side signals, or q's
shape for sampled drafts), which these logs cannot test. Notable on the way: at 245K greedy on
repetitive text K = 3 beats K = 2 (measured 120.0 against 111.8 tok/s, draft + verify).

**Caveat (found in sw115).** `fr_bench` does not stop greedy decoding at the end of the answer, and
greedy text then loops or repeats an end token. Window 9's greedy answer ends at token 524 of its
768 here (sw115, the same prompt), so the w9 greedy rows include rounds on a repeated token, which
flatters long K; the 245K greedy text also turns repetitive late. The note above that K = 3 beats
K = 2 at 245K greedy should be read with that in mind. The sampled rows are not affected.

The instrumentation stays: `--round-log`, and the chain records the argmax drafts' probabilities
(one small reduction per draft step).

## Files

- `sw114.sh`: the runs; `sw114.out` their summary lines; `logs/`: fr_bench outputs and round logs.
- `simulate.py LOGDIR`: costs, calibration, per-context rules and the oracle (`simulate.out`).
- `rules.py LOGDIR`: single rules across contexts (`rules.out`).
