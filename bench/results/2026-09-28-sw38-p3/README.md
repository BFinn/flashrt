# sw38: where a prefill chunk's time goes (nsys, 2026-09-28)

16,384 tokens in chunks of 4,096, fp16 KV. The kernel sums (`pf16k_cuda_gpu_kern_sum.csv`)
are of the run after the first fixes; `pf16k.txt` its output: **1,883.8 tok/s**.

First profile (before the fixes), per 4K chunk: expert MMQ 0.51 s (its grid covered all 4,096
tokens for each of 512 experts), attention partials 0.26 s, planar-to-ggml conversion 0.17 s,
GDN delta rule 0.17 s, GDN conv 0.08 s, grouping helper 0.05 s. Fixes: the expert grid sized by
the largest expert's token count (gate and up share one routing), a word-level conversion with
staged stores, the GDN conv parallel over tokens. After: expert MMQ 0.39 s, conversion 0.08 s,
conv 0.016 s per chunk.
