# Speed work 17: CUDA graphs for decode (2026-09-28)

Each decode token now replays two captured CUDA graphs instead of launching about 1,600
kernels. The graphs split around the PLE read:
- **Graph A:** the parameter copy, the embedding and layer 0.
- **Between the graphs:** the host waits for the PLE n-gram rows, which are read from the SSD
  on a helper thread.
- **Graph B:** the PLE upload, layers 1-47 and the head.

Per-token values are read from a 12-byte device block (token, position, doorbell seq) instead
of launch arguments:
- the QSA kernels, including the K/V write offset;
- the doorbell kernels;
- the embedding, which gathers the Q3_K row by the device-side token id.

In graph mode the QSA path always runs the indexer selection, which falls back to dense attention
below its width, and its scratch is reserved for the full KV capacity before capture. Any other
`forward()` drops the graphs, and the next decode token captures them again (97 µs to
instantiate). The adaptive cache's table updates are enqueued after graph B, outside the graphs.

## Correctness

- **Same session, same state (`graph_2k_try.txt` / `nograph_2k_try.txt`):** graphs on and off give
  identical tokens and identical hit/miss counts.
- **Fast-path KLD:** 0.008744, median 0.00114, same top-1 96.73%. Bit-identical to sw16.

## Speed

| Arm | Runs | Before (sw16) | With graphs |
|---|---:|---|---|
| 2K, 256 tokens | 3 | 89.82 / 88.50 / 89.67 | **98.70 / 99.88 / 100.76** |
| 32K fresh prefill (state saved), 3 windows of 128 | 1 | 78.5 / 77.9 / 79.5 (sw12) | **94.88 / 96.18 / 99.22** |
| 245,760 from the sw15 state, 3 windows of 128 | 1 | 57.94 / 57.17 / 58.28 | **61.27 / 62.01 / 64.23** |
| Same session, 2K (`*_try.txt`) | 1 each | 89.80 (off) | 101.32 (on) |

- **The gain is the gaps between kernels.** At 2K, about 1.1 ms per token came off the
  roughly 11.2 ms token.
- **At 245K the ceiling is still the expert cache:** 3,499 slots, with hit rates of 55-58%.
- **The nsys summary (`decode_2k_*` not kept) counts graph launches, not the kernels inside
  them.** A kernel-level profile needs `--cuda-graph-trace=node`.
