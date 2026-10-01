# sw119: window 9 and the agent session after the expert-cache fix (2026-09-30)

sw118 fixed a slot leak in the expert cache under speculation. This re-measures the headline
numbers on the fixed build (`sw119.sh`; `sw119-summary.txt` is `bench/depthsum.py` over sw119 and
sw110).

**Window 9's protocol** (sw110's arms; sw110 ran the build before the fix; decode tok/s, n = 5,
P n = 2 as a control):

| arm | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| G, head, greedy: sw110 → sw119 | 131.9 → 125.7 | 107.1 → 106.0 | 99.1 → 99.5 | 87.0 → **92.3** |
| S, head, temperature 1.0 | 119.5 → 118.3 | 108.3 → 106.1 | 99.8 → 99.5 | 91.4 → **96.8** |
| P, no head | 102.2 → 102.0 | 94.8 → 94.0 | 87.5 → 87.7 | 82.8 → 82.3 |
| G hit rate, % | 84.6 → 85.9 | 81.1 → 81.2 | 82.1 → 83.1 | 81.8 → 81.9 |
| G draft acceptance, % | 64.3 → 55.8 | 55.2 → 54.6 | 57.0 → 55.5 | 49.7 → 55.8 |

- Hit rates are equal or up to 1.3 points higher. The speed changes at 1K and 250K follow the
  draft acceptance: the fix changes which experts are cached, GPU-hit and CPU-miss arithmetic
  differ in the last bits, greedy decoding then takes another near-tie, and the generated text
  (384 tokens per depth, not stopped at end-of-turn) drafts better or worse.
- Each prompt here adds 4,096+ new tokens, so the cache is refilled per depth and a 384-token
  decode leaks little: no systematic effect was expected, and none shows.
- Still ahead of Strata 0.1.6 greedy (87.0 / 96.0 / 85.0 / 80.4) at every depth: +44%, +10%, +17%,
  +15%; at temperature 1.0 by 34-47%.
- Prefill, reuse and time to the first token are unchanged.

**The agent session** (`agent.out`, `bench/agent_trace.py`, greedy, 2 runs per arm, the engine
before the fix built from 749cb02^): before 126.3 / 125.9 tok/s at 83.7% hits, after 120.1 / 121.6
at 81.4%. Each arm repeats itself exactly, and the two arms part ways at turn 3 (different tool
calls, 3,532 against 3,240 tokens), so the sessions differ in content. Over the first three turns,
which are the same requests, the arms are level (118.7 / 98.6 / 126.5 against 125.6 / 95.6 / 123.2
tok/s). 7-8 of the 12 turns add 4,096+ tokens and refill the cache. No effect is measurable here.

**Where the fix matters:** many requests that each add fewer than 4,096 tokens (chat through the
server): GSM8K over 200 requests, 61 → 143-148 tok/s at steady state (sw118).

## Files

- `sw119.sh`; `sw119.log` (the whole run), `sw119.out` (the depth rows), `agent.out`.
