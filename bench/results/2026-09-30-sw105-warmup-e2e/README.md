# sw105: the expert cache's warm-up settings end to end (2026-09-30)

sw104's new defaults (seed scale 0.03, swap budget 64) on the engine (`sw105.sh`). **This build
committed uploads whenever a query found them done.** With 64 in flight that made runs differ
(sw106), so these numbers come from a nondeterministic build. The commit is deterministic from
sw107 on, and the end-to-end numbers are measured again on that build (sw108).

**Window 9's protocol**, MTP greedy, n = 3 (`sw105-summary.txt`), against sw102 (same build, old
settings):

| | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| sw102, decode tok/s | 109.2 | 87.3 | 83.1 | 81.8 |
| sw105 | **126.2** | **107.5** | **101.0** | **89.3** |
| hit rate, sw102 → sw105 | 80.2 → 84.1% | 67.4 → 80.8% | 70.0 → 82.5% | 70.3 → 81.7% |

**The agentic session** (`agent_trace.py`, greedy, 2 runs each, interleaved): old settings (seed 1,
budget 32) 108.3 / 108.3 tok/s at 76.1% hits; new 126.5 / 124.4 at 83.1-83.9% (+16%). Time to the
first token is unchanged (17 s over the 12 turns).

**KLD:** `--fast` 0.009204 (sw101: 0.008602), window 3 0.008913 (0.008763). sw106 found the runs no
longer repeat.
