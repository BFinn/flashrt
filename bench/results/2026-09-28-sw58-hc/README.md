# sw58: less hyper-connection traffic in prefill (2026-09-28)

Two changes in `hc_mix`'s prefill path:
- **The norm no longer writes xn** (T x 10,240 floats per mix). It writes each row's 1 / rms, and
  the gated mean recomputes xn from x with the norm's own expression (`k_gated_mean_x`). Same
  values.
- **One block per token for both norms** (`k_hc_combine_norm4`, the combine optional), with the
  4-output inject product computed in that block (`block_sum4`). It used to be a cuBLAS product
  of its own, reading the BF16 copy. The result is not bit-identical (full-precision xn, another
  summation order).

32,768 tokens, q8 KV, automatic chunks (16,384), one run each:

| | prefill |
|---|---|
| before (sw57, per 32) | 5,155.6 tok/s |
| now | **5,376.9 tok/s** (+4.3%) |
| 65,536 (nsys run) | 5,406.5 tok/s |

64K profile (`p64k_cuda_gpu_kern_sum.csv`): 11.8 s of kernels (sw55: 12.5 s). The norms took
1.03 s in two kernels and now take 0.66 s in one; the 4-output product's cutlass kernel (0.18 s) is
gone.

KLD (`fr_kld --ctx 8192 --chunks 2 --prefill-chunk 1024`): fp16 KV 0.008405 (96.86%), q8 KV
0.008685 (96.68%). At the noise floor.
