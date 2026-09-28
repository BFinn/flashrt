# sw31: P2 conditions (temperature 1.0), distribution test, Q2_0 head (2026-09-28)

Sampling at the P2 gate's settings (`--temp 1.0 --top-k 20 --top-p 0.95`, seed 1), q8 host KV
with a GPU hot set of 4,096 blocks, Q4_0 head experts unless noted, head vocabulary 32,768 ranked
+ the prompt's tokens, draft chain in a graph, GPU hits grouped by expert. States:
`state-32k-q8-mtp.bin` (saved by `spec2_t1_32k_fresh`) and `state-245k-q8-mtp.bin` (sw30), each
with the head's `.mtp`. 6 windows of 128 tokens unless noted.

| Arm | tok/s min / max / mean | tokens/round | hit rate |
|---|---|---|---|
| 32K plain (no head) | 95.9 / 99.3 / 97.9 | 1 | 94.0% |
| 32K `--spec 1` | 107.8 / 119.6 / 114.2 | 1.560 | 90.9% |
| 32K `--spec 2`, fresh prefill, 3 windows | 100.5 / 107.1 / 103.6 | 1.809 | 90.3% |
| 245K plain (no head) | 77.2 / 80.9 / 78.5 | 1 | 88.0% |
| 245K `--spec 1` | 81.5 / 101.3 / 90.2 | 1.614 | 85.1% |
| 245K greedy `--spec 2` | 87.5 / 100.1 / 92.1 | 2.771 | 67.0% |

2K, greedy, one run each: `--spec 2` 119.3 (the graphed draft chain); with the Q2_0 head 126.8
(7,453 cache slots against 6,940, hit rate 89.6% against 87.7%, 1.969 tokens per round against
2.032); with 11 CPU workers 124.6 (same tokens as 8 workers).

**Distribution test** (`--dist-test`, `--spec 2`): at 2K, 400 seeds: the first token equal to
the plain step's in 400 (TV 0), the second token (emitted in 82) equal in 82 of 82. At 245K
(from `dist_245k.txt`, before the abort below): 300 of 300, and 59 of 59.

**Aborted:** `spec2_t1_32k`, `spec2_t1_245k`, `dist_245k` (after its test), `spec2_t1_w11_245k`:
"draft chain: an illegal memory access". The captured draft-chain graph baked in the head's QSA
scratch buffers; a larger eager catch-up then reallocated them. Runs from a fresh prefill had
grown the scratch in the prefill already. Fixed (graphs recapture when their buffers moved) and
rerun in sw33.
