# P1: CPU miss path for one MoE layer (`moe_cpu` on `CpuPool`, 2026-09-27, 1 run per cell)

Ryzen 9 7900X, DDR5-3600, nothing else running. `tools/bench_moe_cpu`: each call computes k
random experts from a 3.3 GB arena of random Q2_0 experts (1.38 MB each; far above the L3)
for 1 or 3 tokens, with each expert's rows split across the pool (the caller is worker 0).
About 1.5 s per cell. Raw output: `bench_moe_cpu_t1.txt`, `bench_moe_cpu_t3.txt`.

Mean latency per call, µs (weight read rate):

| Workers | 1 miss | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 4, 1 token | 33.0 (41.9 GB/s) | 64.9 | 125.7 | 245.3 | 517.3 |
| **6, 1 token** | **31.7 (43.6)** | **59.0 (46.9)** | **115.7 (47.8)** | **224.6 (49.2)** | **447.3 (49.4)** |
| 8, 1 token | 31.6 | 60.4 | 121.8 | 250.5 | 502.5 |
| 11, 1 token | 32.4 | 62.3 | 124.7 | 245.6 | 483.6 |
| 6, 3 tokens | | 79.2 | 156.5 | 304.1 | 607.0 |
| 8, 3 tokens | | 74.8 | 138.1 | 268.6 | 524.2 |
| 11, 3 tokens | | 66.0 (41.9) | 128.6 (43.0) | 255.8 (43.2) | 515.5 |

- **Fixed overhead is close to zero.** A single miss takes 32 µs, which is its bytes at
  about 43 GB/s. p99 stays within about 15% of the mean.
- **Six workers are the sweet spot for one token,** at the DRAM ceiling measured by
  `membw` (48.6 GB/s). More workers only contend.
- **Three-token windows (MTP verify) are compute-bound on 6 workers,** and want 11.
- **For scale (estimate, not measured end to end):**
  - Greedy decode at an 86.5% hit rate misses about 1.35 experts per layer, around
    40-45 µs per layer, or about 2 ms per token over 48 layers.
  - Strata's pool reported 5.2 ms per round at 32K, for about 2.6 distinct CPU experts
    per layer at 1.66 tokens per round. That figure includes its doorbell waits.
