# sw93: the server end to end after phase 1 (2026-09-29)

Phase 1 of `docs/improvement-plan.md` (S-1, S-2, S-4, S-5, S-6), on the box, with the current
engine and server (`sw93.sh`, as sw86: `--spec 1`, 64K context, a cache prior). Three runs, each
on a fresh server:

1. **`bench/server_smoke.py`: all 14 checks pass** (`smoke.txt`). This covers sw86's 11 checks
   plus three new ones:
   - `top_k` 65 is rejected with HTTP 400 ("top_k 65 is above 64, the most this engine samples
     from"), where it used to be capped silently;
   - temperature -1 is rejected with HTTP 400;
   - `top_k` 0 is accepted and means no limit (it used to be clamped to 1, which is greedy).
2. **The engine killed under a running server** (`run12.out`, `server-b.log`):
   - `/health` answers 503 `{"status":"engine down"}` (it used to say "ok");
   - a chat request answers 503 in 0.5 ms (`after-kill.json`);
   - the server exits by itself within 4 s, for a supervisor to restart.
3. **SIGTERM to the server process alone** (`sw93-3.out`, `server-c.log`): the server stops
   taking requests and sends the engine `quit`. The engine exits with status 0, and the unit is
   gone in 4 s.
   - Run 1 stopped the unit with `systemctl stop`. That signals every process in the unit, so the
     engine got SIGTERM directly (`server-a.log`: "signal: 15").
   - A service unit for the server should set `KillMode=mixed`, so only the server gets the
     signal.

The server's unit tests went from 11 to 19 (`cargo test`; clippy is clean with
`--all-targets`). The new tests:
- **the tool-call parser:** a multibyte character right after `</function>`, which panicked
  before; and a fuzz loop over random UTF-8 around the tags;
- **fake engines** (shell scripts that speak the protocol):
  - an engine that dies after `ready` gets `EngineDown` within 1 s;
  - a request in flight when the engine dies gets an error event;
  - `shutdown` sends `quit`;
  - control tokens and a stray `</think>` do not reach the answer text;
  - a client that leaves while its request is queued sends `stop`;
- **sampling limits.**
