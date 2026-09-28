# sw64: expert-cache capacity at decode (VRAM reserve 1,024 against 256 MiB) (2026-09-28)

`--spec 1` at P2 conditions (temperature 1.0, top-k 20, top-p 0.95; host KV + hot set 4,096),
from the saved states, 6 windows of 128 tokens.

**First with sampled text** (`sampled/`), which proved inconclusive. At 245K the hit rate rose
75.6% → 83.6% while tok/s fell 83.9 → 81.9. A different slot count changes which experts are
GPU hits and which are CPU misses (same math, other rounding), so each arm samples other text,
and the text moves the hit rate by up to 20 points window to window.

**Then with `fr_bench --teacher`** (new): decode feeds the ids file's own continuation, so every
arm routes the same tokens (identical round counts and acceptance):

| Context | reserve | slots | hit rate | CPU misses | verify per round | tok/s per window | mean |
|---|---|---|---|---|---|---|---|
| 32K | 1,024 | 6,817 | 90.37% | 49,577 | 12.88 ms | 103.7 / 94.4 / 105.5 / 96.2 / 106.7 / 101.8 | 101.4 |
| 32K | 256 | 7,399 | 91.89% | 41,729 | 12.52 ms | 108.4 / 96.9 / 107.9 / 98.8 / 109.8 / 102.9 | **104.1** |
| 245K | 1,024 | 6,547 | 84.18% | 78,957 | 17.42 ms | 70.8 / 86.8 / 83.5 / 72.3 / 79.6 / 79.4 | 78.7 |
| 245K | 256 | 7,129 | 86.05% | 69,380 | 17.03 ms | 70.3 / 86.3 / 82.3 / 76.8 / 84.6 / 83.7 | **80.7** |

The +580 slots (+8.8%) give +2.5-2.7%, about 0.3% speed per 1% capacity. After the cache fill,
decode allocates only small things (graph instantiation, a few MB of window scratch; the head's
vocabulary is pre-reserved), so **the default reserve is now 256 MiB** in the engine and
`fr_bench`.
