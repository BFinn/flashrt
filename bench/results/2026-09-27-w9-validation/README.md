# Window 9: reference engines, final validation (2026-09-27, 11:34-13:02)

The baseline flashrt is measured against. RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600.
One growing conversation (prefix reuse) at 1K / 32K / 134K / 250K tokens, the same token ids
for every arm, 384 generated tokens, arms interleaved (`w9.sh`, then `w9b.sh` after an
interruption). Summaries: `w9-summary.txt` (from `w9_summary.py` over `w9-all.out`).

| Arm | Build | Runs |
|---|---|---|
| L | llama.cpp dev tree (branch `mtp`, e4c893841 + uncommitted QSA block selection and pooled-key cache), 48-slot expert cache, no MTP | 4 |
| G | Strata engine 0.1.6 (KV streaming) + PR #19 sampling (5ffa807), tuned: `--vram-reserve-mib 1024 --pcie-frac 0.35 --pool-workers 8`, MTP, greedy | 4 |
| S | as G, temperature 1.0, top_p 0.95, top_k 20 | 4 |
| S06 | as G, temperature 0.6, top_p 0.95, top_k 20 | 2 |

Decode tok/s, mean ± sd:

| Arm | 1K | 32K | 134K | 250K |
|---|---:|---:|---:|---:|
| L | 37.4 ± 0.6 | 37.2 ± 1.1 | 32.8 ± 1.6 | 30.7 ± 1.1 |
| G | 87.0 ± 0.7 | 96.0 ± 2.2 | 85.0 ± 2.7 | 80.4 ± 3.6 |
| S | 80.5 ± 4.1 | 79.1 ± 1.3 | 73.9 ± 1.8 | 69.9 ± 7.2 (n=3, one EOS) |
| S06 | 75.7 ± 1.1 | 80.3 ± 4.8 | 74.6 ± 3.6 | 72.2 ± 0.9 |

Prefill tok/s of the new tokens: L 616 / 1107 / 723 / 442; Strata (all arms) 677 / 1134 /
1089 / 945.

Notes:
- Strata 0.1.6 greedy is 10-15% faster than the 0.1.4 numbers in `docs/background.md`
  (79 / 83 / 77 / 75), mostly from KV streaming at depth.
- Sampling at t=1.0 costs 7-18% against greedy.
- The P2 gate in `docs/design.md` (≥80 / ≥72 at t=1.0) sits right at Strata 0.1.6's
  sampled numbers (79 / 70), so matching Strata meets the gate, and beating it needs more.
