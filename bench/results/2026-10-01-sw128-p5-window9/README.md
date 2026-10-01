# sw128: the server check and window 9's protocol after P-5 (2026-10-01)

The build has all of P-5:
- the selection on an 8-CTA cluster for decode (sw121), and by depth for prefill (sw127);
- the parallel CLOCK (sw122);
- the argmax on a cluster and fp16 pooled indexer keys (sw126).

**Server check** (the pooled keys changed VRAM). flashrt-server with flashrt-engine (`--spec 2`, 64K
context, cache prior) ran as a temporary unit, with `bench/server_smoke.py` against it:
**all checks passed** (`smoke.txt`).

**Window 9's protocol** (sw110, sw124: the reference engines' prompts, 384 tokens, one growing
conversation, 5 runs per arm, interleaved; `sw128.out`, `*.log`). `sw128-summary.txt` is
`bench/depthsum.py` over sw128, sw124 and sw119.

| decode tok/s | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| P, no head, greedy | 103.5 ± 2.1 | 95.5 ± 0.8 | 93.9 ± 0.9 | **91.9 ± 1.7** |
| G, head, greedy | 127.9 ± 4.4 | 111.3 ± 4.1 | 103.7 ± 2.3 | **104.9 ± 2.7** |
| S, head, temperature 1.0 | 118.8 ± 16.5 | 95.5 ± 17.0 | 106.7 ± 2.1 | 106.4 ± 2.9 |
| sw124 (P / G / S) | 101.3 / 125.8 / 124.6 | 94.9 / 107.8 / 106.7 | 93.5 / 104.2 / 102.3 | 90.3 / 99.1 / 102.3 |
| sw119, before P-5 (P / G / S) | 102.0 / 125.7 / 118.3 | 94.0 / 106.0 / 106.1 | 87.7 / 99.5 / 99.5 | 82.3 / 92.3 / 96.8 |

| prefill tok/s of the new tokens, P | 32K | 134K | 250K |
|---|---:|---:|---:|
| sw128 | **5,635** | **5,683** | **5,254** |
| sw124 (sw121's selection in prefill) | 5,060 | 5,064 | 4,734 |
| sw119 | 5,665 | 5,756 | 5,233 |

- **Against sw119, before P-5,** at 250K: P +11.7%, G +13.7%, S +9.9%; at 134K P +7.1%, G +4.2%.
  Prefill is back at sw119's level (sw127), and the 250K prompt takes 22.2 s without the head
  (sw124: 24.7 s).
- **The greedy text changed** with the fp16 keys (KLD within the band, sw126). The G arm's hit
  rates and draft acceptance therefore moved too: acceptance 54.6 → 58.2% at 32K and 55.8 →
  58.8% at 250K. Part of G's gain over sw124 is that text, and sw126's teacher-forced A/B is the
  controlled measure.
- **The S arm's spread at 1K and 32K is its text.** Each run samples its own answer. At 32K two of
  five answers kept 35-37% of their drafts (162 / 442 and 158 / 452), against 55-62% in the
  others, and ran at 78-80 tok/s with normal hit rates (75-81%).
- **Against Strata 0.1.6** (w9-validation), greedy: G ahead by 47%, 16%, 22% and 30%; P ahead at 1K,
  134K and 250K by 10-19%, and level at 32K. At temperature 1.0, S is ahead by 21-52%.
- **Against llama.cpp** greedy: P 2.6-3.0x, G 3.0-3.4x.
