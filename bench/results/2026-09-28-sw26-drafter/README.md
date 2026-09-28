# sw26: a cheaper drafter: Q4_0 experts, trimmed LM head (2026-09-28)

**Changes:**
- **Head experts requantized Q8_0 → Q4_0 at load:** 2,649 → 1,449 MiB of VRAM.
- **Trimmed head:** the top N tokens of a frequency ranking (`bench/mtp_vocab.py` over wiki.train
  and the five calibration domains; the bench prompt is wikitext test, which is disjoint) plus
  the prompt's distinct tokens, gathered into a compact copy of the target's LM head rows.

**Acceptance probe, 2K, 3 drafts:**

| Head | draft step | tokens/round k=1 / 2 / 3 |
|---|---|---|
| Q8_0, full vocab (sw23) | 0.94 ms | 1.711 / 2.115 / 2.348 |
| Q4_0, full vocab | 0.91 ms | 1.701 / 2.071 / 2.272 |
| Q4_0, 32,862 tokens | 0.42 ms | 1.689 / 2.071 / 2.276 |
| Q4_0, 16,540 tokens | 0.39 ms | 1.693 / 2.091 / 2.362 |

Neither costs measurable acceptance (differences are within one 256-token run's noise); the
hit rate recovers from 85.9% to 88.2-88.6%.

**Speculative decode, Q4_0 + 32K vocab, still eager windows:**

| Arm | tok/s | tokens/round | draft ms | verify ms |
|---|---|---|---|---|
| `--spec 1` | 105.86 | 1.691 | 0.47 | 15.36 |
| `--spec 2` | 98.70 | 1.932 | 0.90 | 18.37 |
| `--spec 3` | 90.43 | 2.151 | 1.33 | 22.07 |

First arm above the 101-102 tok/s baseline.
