# sw53: VRAM held by the chunk path, by chunk length (2026-09-28)

`fr_bench` now prints free VRAM before prefill, and used/free VRAM plus the chunk buffers after
it. 32,768 tokens, q8 KV, one run each:

| Chunk | chunk buffers | VRAM used above the pre-prefill level | prefill |
|---|---|---|---|
| 8,192 | 6,176 MiB | 6,336 MiB | 3,690.6 tok/s |
| 12,288 | 8,246 MiB | 8,450 MiB | 3,843.4 tok/s |
| 16,384 | 10,316 MiB | 10,566 MiB | 4,015.2 tok/s |

The chunk path needs about 2,036 MiB fixed (two planar expert slices and the converted one),
plus 0.505 MiB per token, plus 2-3% grown later (GEMM workspaces, QSA scratch). Per token:
- the expert stream, about 220 KB (the per-slot down outputs alone are 100 KB);
- the block scratch, 205 KB, about 30% more than the widest mixer needs;
- the layer buffers, 102 KB.

**245,760 tokens, q8 in VRAM, chunks of 12,288:** 8,521 MiB free before prefill; afterwards
15,800 MiB used and 39 MiB free. It fits and runs at **3,662.2 tok/s** (67.1 s; chunks of
8,192: 3,361.7).
