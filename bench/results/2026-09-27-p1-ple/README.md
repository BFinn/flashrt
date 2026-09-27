# P1: PLE (n-gram table) reads with RowReader (2026-09-27, 1 run each)

Samsung 990 PRO, 27.5 GB IQ4_NL table (`per_layer_token_embd`, 90-byte rows), cold O_DIRECT
reads, so no page cache. Prompt: the first 32,000 tokens of the window C wikitext prompt.
Tool: `tools/fr_ple`.

- The prompt needs 512,000 row reads (16 per token): 318,012 distinct rows, 310,867
  distinct 4 KiB pages.

| Reader depth | Whole prompt | 2,048-token chunks | Decode, 16 rows per token: p50 / p90 / p99 |
|---:|---:|---:|---|
| 64 | 0.497 s (625K reads/s) | 0.577 s | 134 / 167 / 217 µs |
| 256 | 0.365 s (851K reads/s) | 0.439 s | 269 / 288 / 306 µs |

- **Strata on the same prompt and table spent 21.6 s blocked on these reads** (P0 window C,
  `--stats`), at about 16K reads/s. That is the drive's queue-depth-1 rate.
- **For decode, a small pool is better.** Waking 256 threads costs more than the reads.
  PLE feeds layer 1 only, so a token's rows can be fetched while layer 0 runs.
- **Correctness:** 2,000 sampled rows were byte-identical to buffered reads.
