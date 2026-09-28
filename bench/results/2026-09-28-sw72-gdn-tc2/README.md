# sw72: chunked GDN follow-ups (2026-09-28): three state blocks per SM kept, U~ in fp16 reverted

Two changes after sw71, measured together:
- **The state kernel at three blocks per SM** (`__launch_bounds__(256, 3)`, 80 registers).
  The 192 blocks then run in one wave.
  - `test_gdn` at T = 3,000: 0.70 → 0.67 ms.
  - nsys: state 53 → 46 µs per 512-token slab.
  - It is mma-bound on the SMs that hold three blocks: about 424 mma per block and chunk.
- **U~ stored in fp16** instead of fp32 (half its workspace):
  - `test_gdn` is unchanged (0.67 ms; relative error 5.0e-4 either way).

The run (`sw72.sh`, the defaults with both changes; one run each):

| | sw71 (fp32 U~, two blocks per SM) | sw72 |
|---|---|---|
| KLD mean (`kld-chunk1.log`) | 0.008396 (96.996%) | 0.008744 (96.740%) |
| prefill 32K | 5,954.9 tok/s | 5,977.7 |
| prefill 64K | 5,982.0 | 6,011.5 |

- **Prefill** moved by +0.4-0.5%, within one-run noise.
- **KLD** is still inside the 0.0082-0.0090 band of earlier runs, but higher than sw71.
- **Decision:** U~ goes back to fp32, since it bought no speed. The three-block launch bound
  stays; it does not change outputs.

Also tried (test_gdn, T = 3,000; not kept):

| variant | chunked |
|---|---|
| slab of 4 chunks | 0.78 ms |
| slab of 16 chunks | 0.69 ms |
| 64 columns x 8 warps (96 blocks) | 0.75 ms |
| 64 or 128 columns x 16 warps | 0.77 ms |
| **32 columns x 8 warps, 3 per SM, slab 8 (default)** | **0.67 ms** |
