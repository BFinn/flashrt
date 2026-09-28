# sw41: KV in VRAM against host KV during prefill (nsys, 64K, 2026-09-28)

65,536 tokens in chunks of 8,192.

| KV | prefill tok/s | attention partials |
|---|---|---|
| q8 in VRAM | 2,206.6 | 5.4 s |
| q8 in host memory + hot set 4,096 | 1,328.9 | 24.4 s (zero-copy reads of the host store) |

Other kernels, per 64K: expert MMQ 5.4 s, GDN delta rule 2.8 s, indexer scores 1.6 s, Q3_K MMQ
1.2 s, BF16 conversions for cuBLAS 1.2 s (of 26.9 s).
