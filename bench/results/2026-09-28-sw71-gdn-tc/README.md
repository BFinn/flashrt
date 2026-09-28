# sw71: the chunked GDN on fp16 tensor cores (2026-09-28)

The chunked (WY) delta rule of sw70, rebuilt around `mma.m16n8k16` (fp16 in, fp32 accumulation).
- **`k_gdn_chunk_prep`**, per (64-token chunk, head), 8 warps, 48 KiB shared (two blocks per SM):
  - K K^T and Q K^T are computed by mma: the decays and beta are applied in the epilogue, giving A
    (fp32) and P.
  - T = (I + A)^-1 is solved in fp32 by column substitution in registers (a warp per 8 columns).
  - W = T diag(beta gamma) K and U~ = T diag(beta) V are computed by mma.
  - Q^ (scale and decay applied), K^T (decay-to-end applied), W and P are written as ready-made
    mma A fragments, one 16-byte load per lane.
  - All global loads are issued first; the block is otherwise latency-bound.
- **`k_gdn_chunk_state`**, per (head, 32 value columns), 8 warps:
  - The state stays fp32 in registers, as the accumulator of S0 = gamma_C S0 + K^T U.
  - An fp16 copy in shared memory is the B operand of [W; Q^] S0.
  - U and O follow; two barriers per chunk.
- **The same precision split as FLA's chunked kernels** (with fp16 in place of bf16).
- **Default on** (`FLASHRT_GDN_CHUNK=0`: the column kernel), for dk 128 and T >= 64 outside
  verify windows.

**`test_gdn`** (`test_gdn.txt`), against the fp32 column kernel, synthetic inputs of the
model's shape:

| T | outputs (relative L2) | final state | column kernel | chunked, fp16 mma |
|---|---|---|---|---|
| 1,000 | 5.0e-4 | 4.1e-4 | 0.42 ms | 0.24 ms |
| 3,000 | 5.0e-4 | 4.0e-4 | 1.23 ms | 0.71 ms |

- **nsys (T = 3,000):** per 512-token slab, prep 65 µs and state 53 µs. In total 0.23 µs per
  token, against 0.41 for the column kernel.
- **clock64 stamps in prep** show its load phases at about 10 µs each: about 70 MB per slab
  (q, k, v in; fragments out) against DRAM.
- **The state kernel:**
  - It does 86K mma per chunk over all heads, 2.9 µs at the measured fp16 peak (122 TFLOPS),
    against 6.7 µs measured.
  - 192 blocks at 102 registers fit two per SM, so it runs in two waves.

**Correctness gate** (`kld-chunk1.log`: `fr_kld --ctx 8192 --chunks 2 --prefill-chunk 1024`,
fp16 KV, `FLASHRT_GDN_CHUNK=1`):
- KLD mean **0.008396**, same top-1 96.996%, PPL ratio 1.00015.
- For comparison, sw62 and sw69 gave 0.0082-0.0090.

**Prefill** (`fr_bench --prefill-chunk auto --kv q8`, 16K chunks, one run each):

| depth | column kernel | chunked, fp16 mma | |
|---|---|---|---|
| 32K | 5,754.8 tok/s | **5,954.9** | +3.5% |
| 64K | 5,802.1 | **5,982.0** | +3.1% |

Next:
- The prep traffic: raw Q and K^T fragments shared by the 3 heads of a key group, and U~ in fp16.
- The state kernel in one wave (three blocks per SM, or fewer, larger blocks).
