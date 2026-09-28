# sw54: prefill chunk length chosen from free VRAM (2026-09-28)

`ForwardRef::chunk_bytes(T, end)` estimates the chunk path's memory from the allocation
formulas: block scratch, layer buffers, the expert stream (checked against what
`create_expert_stream` allocates), the GEMM workspace, QSA scratch, and in hot-set mode the KV
mirror, plus 3% and 64 MiB. `pick_chunk` takes the longest chunk, up to 16,384 in steps of 1,024,
that fits the free VRAM less 256 MiB, then evens it out over the chunks the prompt needs.

`flashrt-engine` now does this by default (`--prefill-chunk 0`; `--prefill-chunk-max 16384`), and
`fr_bench` does it with `--prefill-chunk auto`. The expert stream's GEMM workspace no longer
reserves room for BF16 activations it never uses (22 KB per token less).

| Run | free before | chunk picked | estimate | used | prefill |
|---|---|---|---|---|---|
| 32,768, q8 | 11,479 MiB | 16,384 | 10,688 MiB | 10,216 MiB | **4,005.6 tok/s** |
| 245,760, q8 in VRAM | 8,521 MiB | 11,264 | 8,083 MiB | 7,708 MiB | **3,623.2 tok/s** (sw50: 3,361.7) |
| 245,760, host KV + hot 4,096 (mirror) | 11,427 MiB | 10,240 | 10,706 MiB | 10,276 MiB | **3,549.0 tok/s** (sw50: 3,306.5) |

"Used" is the VRAM in use after prefill less that before it. The estimates are 380-470 MiB high,
which is safe. Decode after the 245K prefills: 70.0 / 70.3 tok/s over 16 tokens.

**Engine** (`engine_smoke.txt`): `bench/engine_smoke.py` with a 32,000-token prompt,
`--mtp ... --spec 1 --ctx 65536 --cache-prior ...`, temperature 1.0. The engine picked chunks of
10,752 from 9,905 MiB free (the MTP head is loaded too).

| Request | prompt (reused) | prefill | decode |
|---|---|---|---|
| r1 | 32,000 (0) | **10.2 s**, including the head's catch-up and the cache rebuild (sw45: 33.6K in 16.7 s) | 110.6 tok/s |
| r2 | 32,128 (32,096) | 0.3 s | 107.9 tok/s |
| r3 | 32,144 (32,127) | 0.3 s | 98.3 tok/s |
| r4: r1's prompt, stopped | 32,000 (0) | 10.1 s | cancelled after 7 tokens |
