# sw28: where a verify window's time goes (nsys, 2026-09-28)

`nsys --cuda-graph-trace=node`, 2K, `--spec 2` (windows of 3) against plain decode, 128 tokens
each. `kcmp.py` compares the kernel sums (per round vs per token; about 66 rounds: nsys shuts
the app down at the end of the capture, so the round count is taken from sw27).

| Kernel group | spec2 ms/round | plain ms/token |
|---|---|---|
| `k_moe_combine_db` (waiting for the CPU misses) | 6.60 | 1.07 |
| hyper-connection mix, generic path (MMVF BF16 x2, norms) | ~4.6 | ~2.0 (fused, T = 1 only) |
| MoE hits (`k_moe_gate_up` + `k_moe_down`) | 1.79 | 0.98 |
| Q3R | 1.00 | 0.73 |
| GDN delta rule (with the window backup) | 0.51 | 0.20 |
| total GPU | 19.8 | 9.3 |

- **The CPU misses dominate:** the window's union of missed experts, read from DRAM.
- **The hc mix did not amortise:** the fused kernels only took one token (fixed in sw29).
