# sw106: why the fast-path KLD rose with sw104's cache settings (2026-09-30)

sw105 measured `--fast` KLD 0.009204 against sw101's 0.008602, at the same hit rate (91%) with three
times the swaps. `fr_kld` now takes `--swap-budget` and `--cache-seed-scale`. Before, it used
`CachePolicyConfig`'s defaults: seed 1 and budget 8.

| Settings | KLD | Swaps |
|---|---:|---:|
| new (seed 0.03, budget 64), run a | 0.008975 | 159,553 |
| new, run b | 0.008762 | 159,108 |
| (new, sw105) | 0.009204 | 159,314 |
| old (seed 1, budget 8) | 0.008602 | 55,410 (sw101's values, exactly) |
| seed 1, budget 64 | 0.008799 | 158,524 |
| seed 0.03, budget 8 | 0.008385 | 55,463 |

**The runs with budget 64 do not repeat.** The policy committed an upload when a
`cudaEventQuery` found it finished. With 8 uploads in flight they were always done by the next
step. With 64 (about 77 MB per step) whether one has landed depends on timing, so the cache's
content differs between runs. The fast path's arithmetic depends on that content: a GPU hit and a
CPU miss round differently. No slot is read while it is written: an entry goes live only after its
copy completes. But flashrt's runs were bit-reproducible, and teacher-forced A/Bs depend on it. The
spread here, 0.00876-0.00920, is the size of the cache-content effect sw94 measured, not a
precision loss. sw107 makes the commit deterministic.
