# sw125: the hc mix's weights loaded during the miss wait (P-5): rejected (2026-10-01)

**Budget first.** In context a one-token mix is `k_hc_down2` 7.8 µs plus `k_hc_up_mix2` 15.3 µs,
PDL-overlapped (sw120), 97 mixes per token, about 16% of a 32K token. A mix reads 10.1 MB, so
its floor at ~760 GB/s is 13.3 µs. Even a perfect mix would gain about 3 µs × 97 ≈ 0.3 ms per
token, **≈3% at most**. The round was time-boxed and was to stop if it gained under 3%.

**Idea.** The mix after the MoE waits for `k_moe_combine_db`, which spends ~11.5 µs per layer
waiting for the CPU misses on 10 blocks, with one thread spinning. sw123 filled that wait with an L2
prefetch kernel, which held the stream. This round moved the loads into the mix kernels instead:
- `k_moe_combine_db` calls `cudaTriggerProgrammaticLaunchCompletion()` at its start;
- `k_hc_down2` is launched as its programmatic dependent, loads its weights into registers, then
  calls `cudaGridDependencySynchronize()`;
- `k_hc_up_mix2` already chains off down2's trigger (sw75).

So both kernels' weights could be in flight during the wait, with no extra kernel on the stream.
Toggle `FLASHRT_HC_EARLY` (3799139); outputs unchanged by construction.

**Unit test** (`test_hc_decode`, mixes back to back, weights rotated through > 256 MB): correct at
T = 1..4. Per mix, off / on: T 1 16.9 / 15.5 µs, T 3 22.4 / 20.2 µs, T 1 with the combine
18.4 / 16.0 µs. The next mix's down kernel overlaps the previous mix's up kernel.

**End to end** (same binary, toggle rotated over 4 rounds, teacher-forced from the saved states,
256 tokens; `sw125.out`, `runs/`), tok/s, means of 4:

| Arm | off | on | Δ | off runs | on runs |
|---|---:|---:|---:|---|---|
| 245K plain | 93.71 | 93.78 | +0.1% | 94.31 92.65 93.96 93.90 | 92.24 92.78 94.99 95.12 |
| 32K plain | 113.31 | 110.79 | −2.2% | 112.78 110.89 114.93 114.66 | 110.24 107.02 114.85 111.06 |
| 32K `--spec 2` | 103.63 | 106.50 | +2.8% | 102.98 106.52 105.72 99.29 | 105.81 107.08 106.02 107.08 |
| 245K `--spec 2` | 90.70 | 88.51 | −2.4% | 88.52 91.39 91.37 91.52 | 86.48 91.28 84.89 91.40 |

No arm reaches +3%, and the signs disagree across arms.

**Why** (nsys at 32K plain, 256 tokens, `p_e*_cuda_gpu_kern_sum.csv`; 102.0 → 98.5 tok/s under the
profiler), average µs per call, off / on:
- `k_moe_combine_db`: 13.3 / **18.5**;
- `k_hc_down2`: 7.77 / 7.93;
- `k_hc_up_mix2`: 15.29 / 15.39.

The mix kernels did not get shorter in context, and the combine's wait grew by 5.2 µs per layer.
sw123's per-line prefetch hints showed the same (the wait 11.5 → 22.5 µs). Two ways of moving the
mix's DRAM traffic into the miss window have now both lengthened it. The mechanism is not
isolated: GPU DRAM traffic should not slow the CPU's misses, and without hardware counters the
mapped-memory polling path cannot be observed directly. Nor is it known why down2's measured
duration did not grow if it started early.

**Conclusion.** Not adopted; the code is removed (7435a51). The fingerprint with the toggle on
equals sw122's (`fp-new.txt`). This closes the hc mix track in P-5: its ceiling is ~3%, and the
wait it would have to hide under does not accept extra traffic.
