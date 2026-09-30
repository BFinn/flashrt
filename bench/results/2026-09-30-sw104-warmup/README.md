# sw104: the expert cache's warm-up (2026-09-30)

P-2 in `docs/improvement-plan.md`. After a prompt fills the cache, the answer's first ~128 tokens hit
poorly (sw99: 46% then 69% on window 9). sw99's simulator showed why. The policy's counts started
as the prompt's routing counts: the weakest resident starts at 56 on window 9. At a decay of 0.7
every 4 tokens, an expert the answer needs cannot beat it for about 70 tokens. And once admission
is quick, the upload budget limits how fast the cache follows.

**The simulator** (`tools/cache_sim.py --policies engine`, now with `seed_scale` and an early
budget; window 9 / wikitext at 7,780 slots, admit 1, margin 1.2):

| Setting | Window 9 hits | first 64 tokens | uploads/token | Wikitext hits |
|---|---:|---:|---:|---:|
| counts seeded at 1x (the default) | 78.0% | 48% | 27.7 | 89.3% |
| seeded at 0.03x | 80.6% | 59% | 30.0 | 89.3% |
| 0.03x, budget 64 | 84.1% | 67% | 49.5 | 91.0% |
| 0.03x, budget 128 for the first 64 tokens, then 32 | 84.1% | 73% | 41.0 | 89.8% |
| 0.03x, budget 128 | 85.5% | 73% | 64.0 | 92.0% |

**New engine settings** (`CachePolicyConfig`; flags `--cache-seed-scale`, `--swap-budget`, and an
early budget for this A/B). The per-token table update now holds 256 entries, not 64: at 64 it
capped uploads near 31 per token whatever the budget.

**A/B** (`sw104.sh`, `sw104b.sh`): as sw100, `fr_bench` with a fresh 32K prefill, 320 tokens
teacher-forced on the model's own greedy generation. Plain decode and, for window 9, `--spec 2`
with the MTP head (argmax drafts, as teacher forcing needs). Arms rotate. Decode tok/s, mean ± sd.

| Arm | Window 9, plain (n = 5) | Wikitext, plain (n = 5) | Window 9, with the head (n = 3) |
|---|---:|---:|---:|
| the defaults (seed 1, budget 32) | 79.7 ± 1.4 (74.4% hits) | 102.5 ± 1.1 (93.4%) | 87.7 ± 0.8 (59.1%) |
| seed 0.03 | 84.1 ± 0.8, +5.5% (78.0%) | 107.9 ± 1.1, +5.2% | 94.1 ± 1.7, +7.4% (66.4%) |
| seed 0.03, budget 128 for 64 tokens | 86.1 ± 1.7, +7.9% (80.7%) | 106.4 ± 2.2, +3.7% | 104.1 ± 2.4, +18.8% (70.8%) |
| seed 0.03, budget 64 for 64 tokens (n = 3) | 86.8 ± 0.4, +8.8% (79.2%) | 108.0 ± 1.0, +5.3% | 100.9 ± 1.4, +15.1% (68.7%) |
| **seed 0.03, budget 64** | **85.9 ± 0.7, +7.7% (80.5%)** | **107.1 ± 0.8, +4.4%** | **108.5 ± 1.0, +23.8% (72.5%)** |

- **The new defaults are seed scale 0.03 and budget 64.** The engine's usual configuration has
  the head, and there the flat budget wins clearly. A verify step routes three tokens, so a
  per-step budget of 32 left it the least room. The early budget lost there, and is removed.
- **Uploads per run rise** (window 9 plain: 6.5K → 10.3K swaps), and the speed still rises: the
  misses they remove cost more than the uploads.
- **Seeding alone speeds wikitext by 5% with its hit rate unchanged** (93.4% / 93.3%). Its misses
  are cheaper: 45 µs against 56 µs for a layer with one miss, 1.27 against 1.48 ms of host miss time
  per token, with the same number of misses per layer. The seed changes which experts miss. Why
  those are cheaper is not measured.

sw105 checks the KLD gate, window 9's protocol and the agentic session with the new defaults.
