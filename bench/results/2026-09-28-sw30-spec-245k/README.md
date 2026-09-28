# sw30: speculative decode at 245K, fresh prefill with the head (2026-09-28)

245,760-token prefill (reference path, 109 tok/s, 37.6 min) with the MTP head catching up on
every batch; q8 host KV with a GPU hot set of 4,096 blocks; the state was saved with the head's
(`state-245k-q8-mtp.bin` + `.mtp`). Then greedy `--spec 2` (Q4_0 head, vocabulary 32,768 ranked +
6,247 prompt tokens), 3 windows of 128 tokens. Kernels as sw29 (drafts still eager).

| Window | tok/s | hit rate |
|---|---|---|
| 1 | 83.74 | 71.1% |
| 2 | 94.75 | 66.1% |
| 3 | 92.26 | 62.4% |

- 2.739 tokens per round: 83% of rounds keep both drafts (the text at this depth repeats
  earlier articles, and the head copies well). Verify 28.8 ms per round, draft 1.55 ms.
- Plain decode at 245K from comparable states: 63.6-80.2 tok/s (sw20-sw22).
- The expert cache has 6,495 slots (the head and the window buffers take about 1.6 GB), and
  its hit rate falls over the windows.
