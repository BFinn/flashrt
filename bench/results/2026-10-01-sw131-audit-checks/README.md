# sw131: checks after the code audit's fixes (2026-10-01)

**What changed** (three read-only audits of the tree before publishing, then fixes):
- **Measurement tools:**
  - every CUDA call in `fr_bench` and `fr_kld` is checked;
  - a flag without its value, an unknown `--kv` value, or a prompt id outside the vocabulary is an
    error;
  - the slot count is guarded against low VRAM.
- **The forward:**
  - graph capture ends its capture when the body throws (`graph_capture.hpp`);
  - state files close on every path, check their writes, reject a negative position, and commit
    the position only after a complete load;
  - a prefill fault is checked where the prefill ends.
- **The engine:**
  - `quit` cancels the queued requests too (each answered `done`, `cancelled`);
  - `--kv f16` with the hot set is an error;
  - the usage text lists every flag.
- **The server:**
  - a streaming client that disconnects during the prefill stops the engine;
  - request numbers are checked, not truncated;
  - stop strings are limited to 16 of 256 bytes;
  - text the pre-tokenizer cannot split is a 400;
  - `/metrics` counts the server's finish;
  - 35 unit tests.
- **Scripts:** run.sh fails on a failed step, and `engine_smoke.py`'s default mode now checks its
  requests.
- **Licensing and CI:** NOTICE, SPDX headers, and CI permissions, pinned actions and lints.

## Checks (`sw131.sh`, `sw131.out`)

| Check | Result |
|---|---|
| Build | 0 warnings |
| sw112's fingerprint against sw130 (`fp-new.txt`) | **identical** |
| The 245K state loaded, teacher-forced 256 tokens, plain (`state-plain.txt`) | swaps 5,864, hits 111,665: **sw126's exactly** |
| The same with `--spec 2` (`state-spec2.txt`) | swaps 5,071, hits 182,814: **sw126's exactly** |
| `engine_smoke.py` default, plain and `--spec 2` (`default2-*.txt`) | 5 / 5 checks each |
| `engine_smoke.py --reuse` with the head (`reuse-mtp.txt`) | 5 / 5 |
| `engine_smoke.py --faults`, plain and with the head (`faults-*.txt`) | 18 / 18 each |
| The server with `--api-key`: `/v1/models` without the key | HTTP 401 |
| `bench/server_smoke.py --key` (`smoke.txt`) | **17 / 17**, including `disconnect in prefill cancels` (1 cancelled, 0 tokens generated) and `metrics` |

**One check was wrong at first** (`default.txt`). r3's new assertion wanted `reused` equal to r2's
prompt length, and the engine reported one less. The engine is right. A prompt's last token
always runs, for its logits (`engine/session.cpp`), so the end-of-prompt checkpoint holds the state
before it. The script's docstring had said "that prompt's length". The check now wants
`len(prompt2) - 1`, exactly, and passes (`default2-*.txt`).
