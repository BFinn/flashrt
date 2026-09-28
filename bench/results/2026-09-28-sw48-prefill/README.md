# sw48: BF16 activations from the hyper-connection norm, expert tile width, 64K profile (2026-09-28)

**Change.** In prefill, `hc_mix`'s RMS norm writes `xn` in BF16 as well as FP32, into the GEMM
workspace. The down and inject products read that copy (`gemm::gemm_bf16`), so the two
`k_to_bf16` passes over T x 10,240 floats per mix are gone. The result is bit-identical: the
same round-to-nearest BF16 values go to the same cuBLAS call.

32,768 tokens, q8 KV, chunks of 8,192, one run each. `FLASHRT_MOE_J` forces the expert MMQ
tile width (J, tokens per tile):

| Arm | prefill |
|---|---|
| before (sw47, column GDN) | 2,806.8 tok/s |
| BF16 from the norm, J = 128 (default) | **2,896.4 tok/s** |
| J = 64 | 2,640.2 tok/s |
| J = 32 | 2,210.5 tok/s |

At chunks of 8,192 an expert averages about 160 tokens, so J = 128 fills only 62% of its two
tiles. Narrower tiles fill better but re-read the weights more often, and they lose. J stays at
128.

**64K profile** (`p64k_cuda_gpu_kern_sum.csv`, nsys): 65,536 tokens, **2,949.8 tok/s**
(sw41: 2,206.6). The first chunk includes the arena's `cudaHostRegister` (1.54 s); it is now
done at load (next commit).

| Kernel | sw41 | now |
|---|---|---|
| expert MMQ (Q2_0) | 5.42 s | 5.42 s |
| attention | 5.42 + 0.72 s (partials + combine) | 0.94 s (`k_attn_tc`) |
| GDN delta rule | 2.80 s | 1.27 s |
| indexer scores | 1.63 s | 1.65 s |
| Q3_K MMQ | 1.24 s | 1.23 s |
| BF16 conversions | 1.17 s | 0.23 s |
| hc norm / combine / gated mean | 0.64 / 0.70 / 0.66 s | 0.82 / 0.70 / 0.67 s |
| all kernels | 26.9 s | 19.4 s |
