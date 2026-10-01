# sw116: GSM8K, flashrt against llama.cpp on the same GGUF (2026-09-30)

**Why.** The KLD gate compares next-token distributions with llama.cpp position by position. A task
score checks what that does not: that generation through flashrt's whole stack (server,
template, tokenizer, expert cache, MTP speculation) solves problems as well as the reference.

**Setup** (`sw116.sh`, `bench/gsm8k_eval.py`):
- The first 500 items of GSM8K's test set (openai/grade-school-math `test.jsonl`, 1,319 items,
  sha256 `3730d312f6e34405…`).
- One chat request per item through each server's OpenAI API, with the GGUF's chat template and
  thinking off (`enable_thinking: false`). The prompt asks for step-by-step work and a last line
  `#### <answer>`. Greedy (temperature 0), at most 1,024 tokens, one request at a time.
- Scored on the number after `####`, else the response's last number, against the gold answer.
- **flashrt** (efa5d3f): `flashrt-server` with the engine as deployed: the MTP head, `--spec 2`
  (greedy argmax drafts are exact), `--ctx 16384`.
- **llama.cpp** (e4c893841 with the dev tree's uncommitted changes, the tree that produced the KLD
  base): `llama-server --jinja -ngl 99 -ot 'ffn_.*_exps=CPU' -fa on -c 8192 --parallel 1`, fp16 KV,
  its expert cache off (`llama.sh`).

**Result:**

| | right | 95% CI | cut at 1,024 tokens | mean tokens |
|---|---|---|---|---|
| flashrt | 480 / 500 = **96.0%** | 93.9-97.4% | 11 | 352 |
| llama.cpp | 483 / 500 = **96.6%** | 94.6-97.9% | 11 | 355 |

Paired: 477 right in both, 3 in flashrt only, 6 in llama.cpp only, 14 in neither. McNemar's exact
test: p = 0.51, no difference. 97.0% of the extracted answers are equal, and 27.8% of the responses
are identical character for character (the rest part ways at some token: GPU-hit and CPU-miss
arithmetic differ in the last bits, and greedy decoding then follows a different near-tie).

**The 9 disagreements** are mostly the token cap:
- Of the 6 items right only in llama.cpp, 5 are flashrt responses cut at 1,024 tokens (12, 241,
  304, 337, 423; 423 was cut in both). One is a different answer (102: 24 against 26).
- Of the 3 right only in flashrt, 2 are llama.cpp responses cut at 1,024 (139, 340). One is a
  different answer (409).
- The cut responses do not loop: they are long self-checking deliberations ("Wait. Let's
  recalculate..."), 11 in each engine, 7 on the same items.

**Conclusion.** flashrt matches llama.cpp on GSM8K within noise (96.0% against 96.6%, p = 0.51),
consistent with the KLD gate.

**Also seen: flashrt's decode slows over many short requests.** The first request decoded at 131
tok/s (server log); the rate per 50 items, completion tokens over request time, prompt included,
was 86, 64, 55, 53, 52, 52, 52, 51, 52, 51 tok/s. The scores are unaffected (greedy output does not depend on speed).
sw117 looks for the cause.

## Files

- `sw116.sh` (pilot `N=10` first, then `N=500`, resumed); `sw116-pilot.out`, `sw116.out`.
- `flashrt.jsonl`, `llama.jsonl`: every item (response, extracted answer, tokens, seconds).
  `python3 bench/gsm8k_eval.py --compare flashrt.jsonl llama.jsonl` reproduces the table.
- `flashrt-server.log`, `llama-server.log`, `llama.sh`.
