# sw50: GDN column kernel v3, and prefill at 32K / 64K / 245K (2026-09-28)

**GDN v3** (`k_gdn_delta_col`): four lanes share two state columns (32 rows each). Each lane
reads the token's k and q rows once per token, for both columns, from a layout where a warp's
four addresses are adjacent. Tokens arrive in tiles of 8 with three in flight (`cp.async`), and
each block has 2 warps, so the 192 blocks spread evenly over the SMs.

v2 was bound by shared-memory bandwidth. Each warp-wide 16-byte read over 4 distinct addresses
costs about 4 cycles, 24 of them per token. Add the uneven blocks per SM, and the model predicts
0.59 µs per token-layer against 0.54 measured.

**The expert arena is now registered at load.** `cudaHostRegister` takes 1.8 s, and before this
change it fell inside the first prefill chunk. The engine and `fr_bench` now do it at startup,
and `fr_bench` prints it as `expert arena registered in 1.8 s`. The prefill times below exclude
it; sw46-sw49 include it.

All runs: q8 KV, chunks of 8,192, one run each.

| Prompt | Arm | prefill |
|---|---|---|
| 32,768 | old GDN block kernel (`FLASHRT_GDN_COL=0`) | 3,216.4 tok/s |
| 32,768 | GDN v3 | **3,530.8 tok/s** (+9.8%) |
| 65,536 | all | **3,505.7 tok/s** |
| 245,760 | q8 in VRAM | **3,361.7 tok/s** (73.1 s; P3 sw42: 1,959) |
| 245,760 | host KV + hot set 4,096 (VRAM mirror) | **3,306.5 tok/s** (74.3 s; P3 sw43: 1,940) |

At 245K the chunks run at 3,585 tok/s at the start and 3,241 near the end. Decode after
prefill: 66-72 tok/s over 16 tokens (normal).

**KLD gate** (`fr_kld --ctx 8192 --chunks 2`):

| Run | KLD mean | same top-1 |
|---|---|---|
| fp16 KV, `--prefill-chunk 1024` | 0.008466 | 96.74% |
| q8 KV, `--prefill-chunk 1024` | 0.008724 | 96.89% |
| `--fast --prefill-chunk 2048 --kv-hot 512` | 0.008503 | 96.75% |
