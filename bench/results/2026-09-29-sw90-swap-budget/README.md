# sw90: swap budget 8 against 32 on the wikitext runs (2026-09-29)

Teacher-forced, P2 conditions (temperature 1.0, top-k 20, top-p 0.95; q8 host KV + hot set 4,096),
from the saved states, 6 windows of 128 tokens:

| Arm | swap budget 8 | swap budget 32 |
|---|---|---|
| 32K plain | 108.0 112.3 111.0 109.3 109.5 100.4, mean **108.4** | 107.6 104.5 108.7 109.1 109.0 102.7, mean **107.0** |
| 245K `--spec 1` | mean **89.1** (verify 15.25 ms) | mean **87.9** (verify 15.40 ms) |

Hit rates are unchanged (94.4% and 88.2% against 94.4% and 87.9%). The extra uploads cost
about 1%.

**Decision: 32 is the default** in the engine and `fr_bench`:
- it is +9-12% when the generation routes unlike its prompt (sw89), which chat and document
  prompts usually do;
- it is -1% when it continues the prompt's own text.

An adaptive budget, larger while the hit rate is low, would take both. It is listed as an
untested path in `docs/sweet-spots.md`.
