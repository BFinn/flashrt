# sw39: prefill speed at 16K and 32K (2026-09-28)

After sw38's fixes, plus the next chunk's n-gram rows read while a chunk computes and the row
reader at depth 64. fp16 KV, wikitext.

| Prompt | chunks of 4,096 | chunks of 8,192 |
|---|---|---|
| 16,384 | 1,986.0 tok/s | 2,103.0 |
| 32,768 | **2,135.4** | **2,271.9** |

P3 gate at 32K (≥ 2,000): met.
