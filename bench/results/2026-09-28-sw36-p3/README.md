# sw36: first chunked prefill, experts streamed to the GPU (2026-09-28)

`fr_bench --prefill-chunk C`: calls above the decode batch (64) take the chunk path: dense
layers as matrix-matrix products (ggml MMQ / cuBLAS), the MoE with each layer's 512 experts
copied from the host arena (676 MB per layer, one copy overlapping the previous layer),
converted to ggml Q2_0 on the GPU and run as grouped MMQ. 8,192-token wikitext prompt, fp16 KV.

| Arm | prefill tok/s | first 24 greedy tokens after |
|---|---|---|
| reference path (64-token batches, experts on the CPU) | 135.9 | reference |
| chunks of 2,048 | 1,317.1 | identical |
| chunks of 4,096 | 1,454.6 | differ from the 2nd token (numerics: see sw37's KLD) |
