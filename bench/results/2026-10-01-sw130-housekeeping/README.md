# sw130: housekeeping: the P-5 toggles removed; fr_bench against the engine explained (2026-10-01)

**1. Toggles removed.** `FLASHRT_ARGMAX_CLUSTER` (and the one-CTA argmax) and `FLASHRT_SELECT_CL1`
are gone; both paths had won exactly (sw126, sw127). sw112's fingerprint of this build equals
sw127's: **identical** (`fp-new.txt`).

**2. Why fr_bench and the engine differed** on window 9's 32K prompt with `--spec 2`. sw129 gave
97.4 tok/s at 75.4% hits in `fr_bench` (256 tokens) against 111.4 at 81.7% in the engine (window 9,
384 tokens). Here both tools ran that prompt alone, fresh, greedy, at both lengths (`runs/`):

| decode tok/s, hits | fr_bench | engine |
|---|---|---|
| plain, 256 tokens | 94.4, 85.3% | 90.1, 85.0% |
| `--spec 2`, 256 tokens | 98.2, 75.4% | 94.1, 74.7% |
| plain, 384 tokens | 99.5, 88.9% | 96.2, 88.1% |
| `--spec 2`, 384 tokens | 105.3, 82.1% | 106.0, 81.0% |

- **At equal length the two tools agree** to within 4%. The difference was the run length. The
  expert cache adapts during the answer (its warm-up, sw99), so a longer run averages higher hit
  rates.
- **Window 9's 32K turn is warmer still:** it follows the 1K turn, whose 384 tokens had already
  adapted the cache. That gives 111.4 there against 106.0 fresh.
- **The engine runs 3-4% slower without speculation.** It has fewer slots: 8,517 against
  fr_bench's 8,913 here, because it reserves 512 MiB and allocates its checkpoints up front (sw86).
  It also times on the wall clock between streamed tokens.
- **Consequence:** `bench/run.sh`'s decode step quotes fr_bench's own reference values (sw129). A/B
  comparisons stay within one tool and one length, as they always have.
