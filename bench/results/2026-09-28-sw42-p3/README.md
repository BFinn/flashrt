# sw42: prefill at 245K, per chunk (2026-09-28)

245,760 tokens in chunks of 8,192. (sw40 ran the same two arms; its log order was misread at
first: host KV was the slow one there too.)

| KV | prefill | first chunk | at 131K | last chunk |
|---|---|---|---|---|
| q8 in VRAM | **1,958.6 tok/s** (125.5 s) | 1,739 | 1,911 | 1,680 |
| host + hot set 4,096 (before sw43's mirror) | 737.6 (333.2 s) | 1,658 | 697 | 492 |

The reference path took about 38 minutes for 245K (109 tok/s). Per-chunk speed falls with depth
(indexer scoring and selection grow with the blocks). P3 gate at 250K (≥ 1,700): met with KV in
VRAM; host KV fixed in sw43.
