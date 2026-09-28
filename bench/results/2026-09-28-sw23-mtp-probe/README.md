# sw23: the MTP draft head, acceptance probe at 2K (2026-09-28)

**What:** `MtpHead` (the NextN block, `arch/qwen4exp/mtp.cu`) runs over the prompt after the
target and, before each greedy decode token, drafts 4 tokens ahead. `fr_bench --mtp --draft 4`
compares the drafts with the tokens the target then decodes. Head: Q8_0 experts as in the GGUF
(2,649 MiB of VRAM), full 248K-token LM head, eager, one host sync per draft step.

**Result** (2K wikitext, 256 tokens, 253 starts):

| | |
|---|---|
| accepted drafts L | 0: 28.9%, 1: 30.8%, 2: 17.0%, 3: 7.5%, 4: 15.8% |
| per-step acceptance | 71.1%, 56.7%, 57.8%, 67.8% |
| tokens per verify round (1 + E[min(L, k)]) | k=1 1.711, k=2 2.115, k=3 2.348, k=4 2.506 |
| draft step | 0.94 ms |

- The head works first time: 71% of first drafts match the greedy target.
- The draft step is dominated by the full LM head (437 MB read) and launches: see sw26.
