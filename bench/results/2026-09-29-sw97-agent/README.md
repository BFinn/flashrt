# sw97: the server end to end, and an agentic coding session (2026-09-29)

Phase 2 of `docs/improvement-plan.md`, item 4. The server was built from the same commit as
sw96, with prefix reuse through host checkpoints, the MTP head's KV mirror and its fast load.
Engine: `--mtp --spec 2 --draft-vocab`, `--ctx 131072`, defaults otherwise. A fresh server (and
engine) per run, as temporary units (`sw97.sh`).

## Server smoke

`bench/server_smoke.py`: **all 15 checks pass** (`smoke.txt`), including the new `timings` check.
The first chat request's timings:

```json
{"cache_n": 0, "prompt_n": 28, "prompt_ms": 676.0, "predicted_n": 3, "predicted_ms": 28.1,
 "draft_n": 2, "draft_n_accepted": 2, "expert_cache": {"hits": 1036, "misses": 404}}
```

The engine starts in about 30 s. The MTP head's load took about 4.5 minutes before this build
(sw98).

## The agentic session (`bench/agent_trace.py`)

A 12-turn review of this repository. The model has two tools over the checkout, `read_file` and
`list_dir`, and a task: review how the engine reuses a previous conversation's state. Every tool
call is answered with the file's contents, up to 24,000 characters. When the model calls no tool,
the next file of a fixed list comes as a user message. Reasoning is on, and the template drops it
from earlier turns. Each turn generates up to 512 tokens. Two arms, three runs each: **g** greedy,
**t** temperature 0.6 (top-p 0.95, top-k 20, a fixed seed per turn). The model called tools in 8-9
of the 12 turns (`sw97.out`).

| Arm | Final context | Time to first token, all 12 turns | Decode | Hit rate (token-weighted) | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| g, greedy | 77,402 | 20.8 s | 101.9 ± 0.4 tok/s | 73.2% | 71.2% |
| t, temperature 0.6 | 66,429 | 19.1 s | 102.6 ± 0.3 tok/s | 70.4% | 74.6% |

- **Every turn reuses the whole previous conversation:** the prompt diverges from the previous
  sequence inside the last assistant turn, whose reasoning the template drops, and the checkpoint
  at the end of the previous prompt covers it. Time to first token is 1.0-2.6 s for 800-11,000
  new tokens, at 8K-77K context.
- **Runs repeat:** within an arm the tokens are the same in every run, and decode varies by less
  than 0.5%.
- **Decode follows the hit rate, turn by turn:** turns with 44-47% hits decode at 73-80 tok/s, and
  turns with 81-86% at 123-125. The low-hit turns are the short ones, a tool call written right
  after a file arrived. The engine refills the expert cache from each large prompt's routing (a
  file of code), and the generation (reasoning about that code) routes elsewhere. The cache
  recovers only over the next few hundred tokens, and a turn of 40-240 tokens ends before it
  has. That warm-up is P-2's target (sw99).

Files: `sw97.sh`, `sw97.out` (per-turn rows and a SUMMARY line per run), `sw97-summary.txt`,
`smoke.txt`, the server logs.
