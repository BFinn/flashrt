---
name: flashrt-server
description: >
  Changing or testing flashrt's Rust HTTP server (server/src/*.rs: the OpenAI and Anthropic
  APIs, chat template, tokenizer, tool-call and reasoning parsing) or the JSON-lines protocol
  between it and flashrt-engine. Covers who owns what between server and engine, the
  protocol's contract and where it lives, the invariants the server keeps (structure from
  token ids, engine liveness, cancellation, request validation), the fake-engine test
  pattern, and the checks a change must pass before it counts. Use before touching server/
  or engine/main.cpp's protocol handling. For general Rust style, the rust-guidelines and
  apollo-rust-review skills apply on top of this.
---

# The flashrt server

## Who owns what

- **The server** (`server/`, Rust, axum + tokio) owns the text side: the tokenizer and chat
  template read from the model's GGUF, reasoning and tool-call parsing, stop strings, and the
  HTTP APIs. It never loads weights.
- **The engine** (`build/flashrt-engine`, C++/CUDA, `engine/main.cpp`, `engine/session.cpp`)
  owns the model and speaks token ids only. It serves one sequence at a time and keeps the
  last conversation's state for prefix reuse (one sequence; the checkpoint at the end of the
  prompt, and up to 8 host checkpoints taken during prefills since sw95).
- The server spawns the engine as a child process and talks to it over stdin/stdout.

| File | What it does |
|---|---|
| `main.rs` | args, `AppState`, routes, auth, `/health`, shutdown on SIGINT/SIGTERM |
| `engine.rs` | the engine child, the JSON-lines protocol, the reader task routing events by request id, liveness |
| `chat.rs` | one generation shared by both APIs: render, tokenize, queue, turn tokens into reasoning / text / tool-call events |
| `openai.rs` | `/v1/chat/completions`, `/v1/completions`, SSE streaming |
| `anthropic.rs` | `/v1/messages` (text, thinking, tool_use blocks), `/v1/messages/count_tokens` |
| `template.rs` | the GGUF's Jinja chat template on minijinja, with Python-compatible `tojson` |
| `tokenizer.rs` | byte-level BPE from the GGUF vocabulary and merges, llama.cpp's QWEN35 pre-tokenizer |
| `gguf.rs` | GGUF metadata only (no tensors) |
| `metrics.rs` | `/metrics` in Prometheus's text format, summed over the engine's `done` events (sw129) |

## The protocol is a contract

`docs/design.md` ("Engine protocol") is its specification: the `ready`, `generate`, `stop`,
`quit`, `token`, `progress`, `done` and `error` messages, and the request limits. A protocol
change touches all of these in one commit:
1. `docs/design.md`;
2. the engine (`engine/main.cpp`, `engine/session.cpp`);
3. `server/src/engine.rs` and its callers;
4. `bench/engine_smoke.py` (drives the engine directly) and the fake engines in the unit tests.

Add a capability to the `ready` event's `features` list rather than assuming it, so a server
and an engine from different commits fail clearly.

**Request limits are checked twice, on purpose.** The engine checks every request before any
GPU work and answers a bad one with an `error` event and no state change. The server checks the
same limits first, so a client gets an HTTP 400 with a message, not a failed generation. Keep
the two in step with `design.md`: token ids within the vocabulary, `max_new` ≥ 1, `temperature`
≥ 0 (0 is greedy), `top_k` 1..64, `top_p` in (0, 1], `min_p` in [0, 1), `seed` an integer in
0..2^53. The server side is `chat.rs` `sampling_of`: an absent value takes the server's default
(its command line), `top_k` 0 means 64 and above 64 is a 400 (never silently capped), and the
seed is masked to 53 bits.

## Invariants the server keeps

- **Structure comes from special-token ids, never from generated text.** Reasoning ends at the
  `</think>` id; tool calls sit between the tool-call ids. Control tokens and stray structure
  tokens (a `</think>` outside a reasoning section) are not output text. Text that merely
  looks like a tag is text.
- **The tool-call format** is the template's
  `<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`.
  Arguments are typed from the request's tool schema; a value without a schema, or one that
  does not parse, stays a string. The parser must never panic on arbitrary UTF-8: advance by
  the terminator actually found, on char boundaries (S-2).
- **The template sees what the model was trained on.** `tojson` matches Python's
  `json.dumps(ensure_ascii=False)` (", " and ": " separators, keys in insertion order); keep
  serde_json's `preserve_order`. A change here changes prompts, so compare renders before and
  after (`flashrt-server --model MODEL.gguf --render REQUEST.json`).
- **The tokenizer matches llama.cpp.** Special tokens are matched as whole strings before
  pre-tokenization. Check changes with `--check-tokenizer TEXT IDS` against a llama.cpp
  tokenization.
- **Engine liveness.** When the engine's output ends, it is down for good: requests in flight
  get an error event, new ones get 503 at once, `/health` answers 503
  `{"status":"engine down"}`, and the server exits a few seconds later for a supervisor to
  restart both. The server never restarts the engine itself.
- **Cancellation.** A client that disconnects, while queued or while generating, makes the
  server send `stop` for its request id. The engine's `done` event still arrives and ends the
  route. Do not drop a route without that event.
- **Shutdown.** On SIGINT/SIGTERM the server stops taking requests and sends `quit`. A service
  unit must use `KillMode=mixed`, so the signal reaches only the server (sw93).
- **Known limits** (`docs/engine.md`, "Known issues"): text only; `tool_choice` "required" or a
  named tool is not enforced; Anthropic thinking blocks carry an empty signature; without a
  `thinking` field the model still reasons and the reasoning is not returned.

## Tests

Unit tests live in `#[cfg(test)]` modules beside the code (28 as of sw129). The pattern for
anything that involves the engine is a **fake engine**: a `sh -c` script that prints a `ready`
line and then speaks the protocol (see `engine.rs` and `chat.rs` tests, `fake_state` in
`chat.rs`, which pairs it with `tokenizer::test_tokenizer()` and a one-line template). Use it
for timing and failure behaviour (dies after `ready`, dies mid-request, ignores `stop`), and
bound every wait with `tokio::time::timeout` so a regression fails instead of hanging.

Phase 3's server items (`docs/improvement-plan.md`, T-7, S-7) are done in part: fake-engine
protocol tests (`engine.rs`: `dead_engine_fails_fast`, a request in flight when the engine dies,
`shutdown_sends_quit`; `chat.rs`: a client leaving while queued), SSE framing (`openai.rs`
`stream_frames`), and the tokenizer's pre-tokenizer and eos checks (`model_round_trip` covers CJK
and emoji when `FLASHRT_TEST_MODEL` is set). Still open: the real chat template against a
checked-in expected render, and a tokenizer fixture that runs without the model (code, special
tokens in user text).

## Checks before a server change counts

Rust is not installed on the workstation: build and test on the GPU box (how to load the
toolchain there, and the box rules, are in `CLAUDE.local.md`). Do not build during another session's benchmark window.

1. `cargo test --release --manifest-path server/Cargo.toml`.
2. `cargo clippy --release --all-targets --manifest-path server/Cargo.toml`: it is clean; keep
   it so.
3. **End to end:** `bench/server_smoke.py --url ...` against a server on the current build, run
   as temporary `systemd-run` units, never a service left running. Templates:
   `bench/results/2026-09-28-sw86-server/sw86.sh` (the 11 original checks) and
   `2026-09-29-sw93-server/sw93.sh` (14 checks, plus killing the engine under the server and a
   SIGTERM to the server alone). `server_smoke.py` has 16 checks now (`metrics` since sw129). Add
   a check for new API behaviour.
4. **If the engine or VRAM budgeting changed** as well: `bench/engine_smoke.py`, with `--faults`
   after changes to `Session` or the forward's state. `fr_bench` allocates differently and
   cannot catch a first-request out-of-memory (sw86).
5. **If prompts or tokens can change** (template, tokenizer, special tokens): compare renders
   and tokenizations before and after; a change to the engine's outputs also needs the KLD gate
   (see the flashrt-cuda skill).
6. **Record** runs worth keeping in `bench/results/<date>-swN-<topic>/` with placeholder paths,
   run `bench/scrub.py`, and update `docs/engine.md` (runbook, known issues) and `design.md` if
   the protocol moved.

## Clean room

The tokenizer's pre-tokenizer pattern comes from llama.cpp (MIT). Parsers and API handling from
vLLM or SGLang (Apache-2.0) may be vendored with their notices. Strata's source may not be
opened (`docs/clean-room.md`).
