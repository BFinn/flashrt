# sw27: verify windows in CUDA graphs (2026-09-28)

One pair of graphs per window length (and window mode); embedding and QSA graph mode for T > 1
tokens. 2K, Q4_0 head, 32K vocab, greedy.

| Arm | tok/s | tokens/round | verify ms/round (eager, sw26) |
|---|---|---|---|
| `--spec 1` | **116.24** | 1.691 | 13.93 (15.36) |
| `--spec 2` | 106.72 | 1.932 | 16.90 (18.37) |
| `--spec 3` | 95.26 | 2.151 | 20.86 (22.07) |

Outputs are identical to the eager runs (same acceptance histograms).
