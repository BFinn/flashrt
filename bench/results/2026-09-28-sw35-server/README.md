# sw35: the Rust server end to end (2026-09-28)

`flashrt-server` (server/) over `flashrt-engine`: tokenizer and chat template from the GGUF,
OpenAI `/v1/chat/completions` and `/v1/completions`, Anthropic `/v1/messages` and
`/v1/messages/count_tokens`, streaming, reasoning, tool calls, stop strings. Engine arguments:
`--mtp ... --spec 1 --draft-vocab ranks.txt --ctx 65536` (+ `--cache-prior` in the second run).
Run as a test unit on port 8090 and stopped afterwards.

## Offline checks (no engine)

- **Tokenizer** (`--check-tokenizer`, wikitext test against llama.cpp's ids): 47,619 of 47,619
  equal, decode round trip identical, 8 ms for 47K tokens.
- **Chat template** (`--render`) against Python jinja2 configured like Hugging Face's
  apply_chat_template: 5 of 5 cases byte-identical (tools, multi-turn reasoning, thinking off,
  merged developer/system messages, parallel tool calls with nested arguments, tool results).

## `bench/server_smoke.py`: 11 of 11 checks pass (both runs)

Chat without thinking ("391"), streamed chat with reasoning, a tool call, the tool-result turn
with prefix reuse (368 of 398 prompt tokens cached), a stop string, raw completion, Anthropic
messages with thinking blocks, Anthropic streamed tool_use, count_tokens, and a client that
disconnects mid-stream (the next request is served 0.2 s later).

## Decode through the server: 2K wikitext raw completion, 256 tokens

| Request | prompt (reused) | first text | tok/s |
|---|---|---|---|
| before the fixes: 3 requests, cache filled from a 28-token first prompt | 2,035 (0) each | 16.5 s | 80 / 97 / 117 |
| with `--cache-prior`, greedy | 2,035 (0) | 16.7 s | 88 |
| same prompt, temperature 1.0 seed 1 | 2,035 (2,034) | 0.02 s | 100 |
| seed 2 | 2,035 (2,034) | 0.01 s | 118 |
| seed 1 again (identical text to the second row) | 2,035 (2,034) | 0.01 s | 117 |

Two fixes came out of the first run:
- **Checkpoint before the prompt's last token.** At the prompt's end it was unusable for the
  same prompt again, since the last prompt token always re-runs for its logits.
- **`--cache-prior`:** `cache-prior-calib32k.bin` holds the routing counts of a 32,768-token
  prefill of the calibration mix (1,024-token chunks round-robin from chat, code, agentic,
  general, multilingual and wiki.train; `fr_bench --count-half-life 0 --save-counts`). The engine
  fills its expert cache from it at startup.
