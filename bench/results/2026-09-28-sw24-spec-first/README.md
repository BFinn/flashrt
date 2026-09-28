# sw24: first speculative decode, greedy, 2K (2026-09-28)

**What:** `ForwardRef::forward_window` / `commit` (verify windows on the fast path: GDN state
backup and replay, conv and PLE history rewind, MoE routing per token with misses grouped per
expert on the CPU) and `fr_bench --spec K`. Everything eager (no graphs for windows yet), Q8_0
head experts, full LM head.

| Arm | tok/s | tokens/round | draft ms | verify ms/round |
|---|---|---|---|---|
| `--spec 2` | 91.74 | 1.969 | 1.95 | 19.20 |
| `--spec 3` | 74.84 | 2.065 | 2.89 | 24.28 |

Baseline (sw22, no head loaded): 100.8-102.0 tok/s.

- Speculation works end to end but loses: a verify of 3 tokens costs twice a token.
- Per round, 10.8 ms of GPU work before each layer's routing (7.4 ms per token plain, with
  graphs) and 7.5 ms of CPU misses: the window's union of missed experts, with 1,700 fewer
  cache slots because of the head's 2.6 GB.
