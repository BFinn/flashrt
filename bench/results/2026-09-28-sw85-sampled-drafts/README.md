# sw85: sampled drafts with speculative sampling (2026-09-28)

**Before:** the MTP head proposed its argmax, and the verifier kept a draft when its sampled token
equalled it. At temperature 1.0 that accepts with p(argmax q). sw84's probe measured
sum min(p, q) at 0.66-0.71 against 0.48-0.58.

**Now (default for `--spec` at temperature > 0):**
- **Drafting** (`sample::draft_row`): the head's logits go through the target's own sampler
  chain (top-k, top-p, min-p, temperature). The draft is drawn from that q with a draw of its
  own, keyed by (seed ^ salt, position). q (token ids, probabilities) stays on the GPU per chain
  step.
- **Verifying** (`sample::spec_verify`): draft j is accepted when u * q(d) < p(d), with a third
  draw. Otherwise the token is drawn from max(0, p - q) normalised, with the plain sampler's
  draw. The window's last row is a plain sample.
- **The draft chain** reads its sampling parameters and positions from device memory, so its
  captured graph serves any request (a second graph for argmax drafts).
- **Wired into** `fr_bench` and `flashrt-engine`. `--argmax-drafts` /
  `FLASHRT_ARGMAX_DRAFTS=1` restores argmax drafts. Greedy decoding and `--teacher` runs keep
  argmax drafts: teacher forcing defines acceptance as "equals the file's token".

**Exactness.** The output distribution equals plain sampling's, but tokens no longer match plain
sampling one for one.
- **`test_spec_sample`** (synthetic rows, 200,000 seeds):
  - the emitted token at the draft's position has TV 0.0033 from the target's chain
    distribution (noise bound 0.0104);
  - the last row has TV 0.0029;
  - acceptance is 0.7690 against sum min(p, q) = 0.7682 (argmax drafts would accept 0.342
    there).
- **Distribution tests on the model** (`dist_spec*_32k.txt`, 400 seeds, 32K): histogram TV 0.050
  against a noise level of ~0.098 for two 400-sample histograms of 12 tokens. The first token
  equals plain sampling's in 36.75% of seeds: no token parity, as expected.

**Decode**, P2 conditions (temperature 1.0, top-k 20, top-p 0.95, seed 1; q8 host KV + hot set
4,096; head Q2_0 over 32,768 ranked + prompt tokens), from the saved states, 6 windows of 128
tokens. The text is sampled, so each arm decodes different tokens:

| Arm | tok/s per window | mean | tokens / round | verify per round |
|---|---|---|---|---|
| 32K `--spec 1`, argmax drafts | 109.8 117.8 128.2 115.7 118.9 123.7 | 119.0 | 1.508 | 11.46 ms |
| 32K `--spec 1`, **sampled** | 136.3 149.8 134.7 150.1 140.0 147.6 | **143.1 (+20%)** | 1.760 | 11.20 ms |
| 32K `--spec 2`, sampled | 149.8 142.3 142.1 144.8 148.3 153.6 | **146.8 (+23%)** | 2.186 | 12.74 ms |
| 245K `--spec 1`, argmax drafts | 85.4 83.9 97.4 95.3 92.8 95.3 | 91.7 | 1.601 | 16.07 ms |
| 245K `--spec 1`, **sampled** | 90.8 98.9 94.1 93.3 103.3 102.3 | **97.1 (+6%)** | 1.734 | 16.50 ms |
| 245K `--spec 2`, sampled | 83.8 93.9 96.8 102.7 98.3 110.2 | **97.6** | 2.180 | 19.87 ms |

- **At 32K** every sampled window is faster than every argmax window. First-draft acceptance
  rose from 50.8% to 76.0%, above sw84's 0.66 estimate (a different text).
- **At 245K** the gain is smaller (60.1% → 73.4% acceptance), and verify is 1.4x as costly
  relative to a draft.
- **With sampled drafts a second draft is now break-even or slightly ahead** (32K +2.6%, 245K
  +0.5%, both within window-to-window spread). Under argmax drafts it lost (sw33).
