# sw109: uploads commit at the next step (2026-09-30)

sw107 made the expert cache's upload commits deterministic, two steps after issue. sw108's agent
turns 0 and 1 have the same prompts and lengths in every run, and there they hit 1-2 points below
the timing-dependent build: that build often committed a step sooner. Now an upload commits at the
next step, waiting for it there. The wait overlaps the token just enqueued: the thread that runs
the policy synchronises on that token next anyway, and CPU misses run on the miss server's thread.

**Deterministic:** `--fast` KLD 0.008996 in both runs (159,480 swaps each), inside the band. Window
3, hot set 512: 0.008913, 100,258 swaps. The timing-dependent build measured exactly these two
values in sw105, and the head run below matches its hit rate and swaps exactly: in those runs it
had committed at the next step.

**Teacher-forced, lag 1 against lag 2** (as sw104, alternating, 3 runs each, `sw109.out`):

| | Lag 2 | Lag 1 |
|---|---:|---:|
| window 9, plain | 86.7 ± 0.4 tok/s (81.4% hits) | 86.9 ± 0.5 (82.1%) |
| wikitext | 107.7 ± 0.7 (93.4%) | 108.1 ± 1.3 (93.6%) |
| window 9 with the head | 106.1 ± 1.5 (71.8%) | 106.4 ± 4.4 (72.5%) |

**The agentic session** (`agent.out`, greedy, 2 runs): **120.4 and 122.6 tok/s at 82.1% hits**. Both
runs generated identical tokens. The same session measured 114.3 at 78.7% with lag 2 (sw108) and
108.3 at 76.1% with the old settings (sw105): **+12%**. Turns 0 and 1 hit 74.0% / 56.7%, as the
timing-dependent build did.

**Lag 1 is fixed** (its toggle is removed). **Server smoke** on that build: all checks pass
(`sw109b.out`).
