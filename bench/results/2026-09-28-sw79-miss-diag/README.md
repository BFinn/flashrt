# sw79: what a CPU miss costs inside decode (2026-09-28)

**The question.** The sw76 profile's miss-server log showed ~97 µs per single-miss layer, against
32 µs for one miss in `bench_moe_cpu` (p1-moe-cpu).

**The run.** Plain 32K, teacher-forced, P2 conditions from the saved state, 2 windows of 128
tokens per arm, no profiler:

| Arm | tok/s | host miss time per token | 1 miss per layer | 2 | 3 |
|---|---|---|---|---|---|
| default (8 workers, adaptive cache; ran first) | 106.4 / 108.1 | 1.13 ms | 60.1 µs | 83.4 | 107.0 |
| `--static-cache` (no expert swaps) | 106.7 / 111.2 | 1.01 ms | 41.1 | 69.6 | 101.1 |
| `--workers 6` | 108.4 / 111.0 | **0.86 ms** | 43.1 | 67.9 | 97.4 |
| `--workers 11` | 106.4 / 112.5 | 0.93 ms | 45.6 | 75.0 | 105.0 |
| `--spin-us 100000` (workers never sleep) | 106.8 / 110.4 | 0.92 ms | 43.9 | 75.7 | 109.4 |

`bench_moe_cpu` in the same state (`bench_moe_cpu.txt`): one miss 30-34 µs (4-8 workers).

- **The 97 µs came from running under nsys.** Without it, a single-miss layer costs 41-46 µs
  (the first arm's 60 µs looks like a first-run effect: the others share its configuration
  except for one flag).
- **About 10 µs over the standalone figure.** The host part includes quantizing x and the
  mailbox writes.
- **The CPU runs misses for about 0.9-1.1 ms of a ~9.3 ms token,** partly overlapped with GPU
  work. At ~43 GB/s per miss it is near the host-DRAM ceiling (STREAM 33.6, `membw` 48.6
  GB/s). There is no large lever left here without faster memory.
- **Six workers are marginally best for one token,** as p1-moe-cpu found, but within noise.
  The default stays at 8 (verify windows want more).

**A consequence for GPU tuning.** In a layer with misses, the GPU work between `k_route` and
`k_moe_combine_db` (hits, shared expert) overlaps the CPU's miss time, so speeding it up mostly
lengthens the wait. This is why sw78 and the moe-hits v2 attempt (reverted: `test_moe_hits`
25.1 against 25.0 µs) showed nothing end to end. Decode gains now have to come from work
outside that window (hc, attention, GDN, dense mat-vecs), or from fewer misses.
