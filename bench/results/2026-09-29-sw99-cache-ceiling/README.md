# sw99: the expert cache's ceiling on window 9's generation (2026-09-29)

sw96 left decode at depth where it was: the expert cache hits 66-68% (with the MTP head) on window
9's generation against 93% on wikitext. How much can any cache policy recover? `sw99.sh` records
routing traces with llama.cpp (`tools/route_trace`, experts on the CPU, greedy): the prompt's and
384 generated tokens'. It does this for window 9's 32K prompt and for 32K of wikitext.
`tools/cache_sim.py` then replays them at the engine's cache sizes: 7,780 slots with the MTP head,
8,634 without. It now has an `engine` policy: primed from the prompt's routing counts (halved every
4,096 tokens, as the engine's are), decayed LFU with the engine's admission rule, at most `--budget`
uploads per token, each resident from the next token on. The traces (.npy, 63 MB each for the
prompts) stay on the box.

**The model is credible:** it predicts 73.0% hits for window 9 at budget 32 and 7,780 slots. The
engine measured 75.0% on that prompt without the head (sw96, P arm, 8,634 slots; the model gives
75.4% there).

## Hits at 7,780 slots (window 9 / wikitext)

| Policy | Window 9 | 64-token windows, window 9 | Wikitext |
|---|---:|---|---:|
| engine, budget 8 | 63.3% | 43 → 60 → 64 → 66 → 69 → 78 | 85.2% |
| engine, budget 32 (the default) | 73.0% | 46 → 69 → 73 → 79 → 84 → 87 | 87.9% |
| engine, budget 128 | 74.0% | 47 → 70 → 74 → 81 → 85 → 87 | 88.6% |
| static: the generation's own top pairs (an oracle fill) | 89.6% | | 92.4% |
| Bélády (the optimum) | 90.5% | | 92.1% |

**The loss is the warm-up, not the steady state.** By its last window the policy reaches 87%,
close to the ceiling. The prompt-primed cache hits 46% over the first 64 tokens and 69% over the
next. A larger budget barely helps: the admission rule, not the number of uploads, limits how fast
the cache follows the generation.

## What would fix the warm-up (`primes.py`, `admission.py`; window 9 / wikitext, budget 32)

| Change | Window 9 | Wikitext | Uploads per token (w9 / wiki) |
|---|---:|---:|---:|
| none (admit 2, margin 1.5) | 73.0% | 87.9% | 14.3 / 7.4 |
| prime from the prompt's last 64 tokens only | 79.2% | 85.7% | |
| prime from the prompt plus another text's generation (weight 1) | 78.0% | 85.6% | |
| admit 1.5 | 75.5% | 88.8% | 17.9 / 9.3 |
| **admit 1, margin 1.2** | **78.0%** | **89.3%** | 27.7 / 21.5 |
| admit 1, margin 1.2, budget 64 | 80.5% | 90.9% | 44.3 / 29.1 |

- **Primes that favour the prompt's tail, or a generation prior, trade one text against the
  other.**
- **Faster admission helps both.** Window 9's steady state rises too (91.8% in the last window).
  It costs uploads, and each upload reads a whole expert from host DRAM, which the CPU misses need.
  The simulation does not charge for that, so the engine decides: sw100.
