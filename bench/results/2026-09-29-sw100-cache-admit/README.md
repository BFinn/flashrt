# sw100: the expert cache's admission, in the engine (2026-09-29)

P-2 in `docs/improvement-plan.md`. sw99's simulation put window 9's decode loss in the cache's
warm-up, and found one change that lifted both texts: admit a missed expert at count 1 instead
of 2, and at 1.2 times the weakest resident's count instead of 1.5. It doubles the uploads, which
read host DRAM as the CPU misses do, so the engine decides.

**Setup** (`sw100.sh`): `fr_bench`, a fresh 32K prefill, 320 decoded tokens teacher-forced on the
model's own greedy generation for that prompt (sw99's route traces). Every arm therefore routes the
same tokens. Window 9's 32K prompt, and 32K of wikitext. Plain decode, no MTP head (8,634 cache
slots). New flags `--cache-admit` and `--cache-margin` (engine and `fr_bench`). Two runs per arm,
alternating.

| Admit / margin / budget | Window 9 decode | hits | swaps | Wikitext decode | hits |
|---|---:|---:|---:|---:|---:|
| 2 / 1.5 / 32 (the old default) | 76.8, 77.4 tok/s | 69.5% | | 103.5, 101.7 tok/s | 92.2% |
| 1.5 / 1.5 / 32 | 78.4, 78.4 (+1.7%) | 72.1% | 4,824-4,846 | 106.5, 102.8 (+2.0%) | 92.6% |
| **1 / 1.2 / 32** | **79.7, 79.2 (+3.0%)** | **74.5%** | 6,458-6,836 | **103.6, 104.9 (+1.6%)** | **93.4%** |
| 1 / 1.2 / 64 | 79.5, 79.9 (+3.4%) | 75.5% | | 103.8, 104.5 (+1.5%) | 93.4% |

- **The hit rates move as simulated:** window 9 +5.0 points at admit 1 (sw99 predicted +5.0).
- **Decode gains less than the hits:** +3.0% on window 9 for 5 points fewer misses. The extra
  uploads read host DRAM too.
- **Admit 1 / margin 1.2 / budget 32 is the new default** (`CachePolicyConfig`, so the engine,
  `fr_bench` and `fr_kld` agree). It gains on both texts, most on the one that routes unlike its
  prompt. Budget 64 adds 0.4% on window 9 and nothing on wikitext.
- Window 9's runs repeat within 0.6 tok/s. Wikitext's spread is about 1.8 tok/s, so its +1.6% is
  about one spread. It is not worse.

sw101 checks the KLD gate, the MTP arms and window 9's protocol with the new default.
