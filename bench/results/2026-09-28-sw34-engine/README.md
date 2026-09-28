# sw34: the engine process end to end (2026-09-28)

`flashrt-engine` (engine/main.cpp over engine/session.cpp) driven over its JSON-lines protocol by
`bench/engine_smoke.py`: 2K wikitext prompt, `--mtp ... --spec 2 --draft-vocab ... --ctx 32768`,
temperature 1.0, top-k 20, top-p 0.95, seed 1, 96 tokens per request.

| Request | prompt (reused) | prefill | decode | finish |
|---|---|---|---|---|
| r1: 2,048 tokens | 2,048 (0) | 18.9 s | 96 tokens, 109.0 tok/s | length |
| r2: r1's prompt + its output + 32 | 2,176 (2,143: the whole sequence) | 0.3 s | 114.7 tok/s | length |
| r3: r2's prompt + 16 other tokens | 2,192 (2,176: the checkpoint at r2's prompt end) | 0.2 s | 120.0 tok/s | length |
| r4: r1's prompt, stop after 5 tokens | 2,048 (0) | 16.4 s | 8 tokens | cancelled |

- Startup (weights, 32 GB expert arena, head, cache) takes 22 s.
- Two runs of the script gave the same r1 tokens and draft counts (deterministic).
