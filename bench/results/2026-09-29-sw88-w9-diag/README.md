# sw88: why window 9's prompt decodes slower (2026-09-29)

`fr_bench` at 32K, fresh prefill, q8 host KV + hot set 4,096, 384 greedy tokens:

| Prompt | decode | expert-cache hit rate | CPU miss time per token |
|---|---|---|---|
| window 9's (synthetic filler + an instruction) | 73.1 tok/s | **65.9%** | 5.95 ms |
| wikitext (the fr_bench prompt) | 107.0 | 93.4% | 1.24 ms |
| window 9's, `--spec 2` | 72.3 | 52.5% | 8.55 ms (2.22 tokens per round, verify 28.5 ms) |

**The cause is the expert cache.**
- It is filled from the prompt's routing counts. On wikitext the continuation routes like the
  prompt.
- Window 9's prompt repeats six sentences, and the model then writes a thinking block and an
  essay, which route to other experts.
- A third of the experts decode needs are not cached. Each verify window widens the set of
  missed experts, so speculation cannot pay.
