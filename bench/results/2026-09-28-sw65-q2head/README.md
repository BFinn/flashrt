# sw65: the MTP head's experts at Q2_0 against Q4_0, teacher-forced (2026-09-28)

`--spec 1` at P2 conditions, reserve 256 MiB, `--teacher`, from the saved states, 6 windows of
128 tokens. The head only proposes drafts (verification keeps sampling exact), so its precision
shows up as acceptance. Q2_0 frees about 680 MB (513 more cache slots):

| Context | head | slots | tokens / round | hit rate | verify per round | tok/s per window | mean |
|---|---|---|---|---|---|---|---|
| 32K | Q4_0 | 7,399 | 1.435 | 91.89% | 12.87 ms | 108.5 / 96.0 / 102.3 / 93.7 / 104.3 / 104.3 | 101.5 |
| 32K | Q2_0 | 7,912 | 1.448 | 92.94% | 12.34 ms | 111.5 / 103.5 / 107.4 / 100.9 / 112.8 / 103.4 | **106.6** |
| 245K | Q4_0 | 7,129 | 1.488 | 86.05% | 17.39 ms | 71.0 / 84.2 / 84.4 / 73.8 / 81.8 / 79.4 | 79.1 |
| 245K | Q2_0 | 7,642 | 1.483 | 87.62% | 16.51 ms | 77.6 / 92.4 / 87.8 / 75.0 / 82.2 / 81.7 | **82.8** |

Acceptance is unchanged, and every window but one is faster (+5% on average).

The same Q4_0 32K configuration measured 104.1 tok/s in sw64: under teacher forcing an arm
still varies about ±2.5% run to run (the CPU miss timing). Here the per-window consistency
carries the result.

**Q2_0 head experts are now the default** (`--mtp-bits 2`) in the engine and `fr_bench`.
