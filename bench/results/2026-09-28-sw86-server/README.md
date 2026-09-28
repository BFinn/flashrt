# sw86: the server end to end on the current engine (2026-09-28)

The first end-to-end server run since sw35:
- `flashrt-server` over `flashrt-engine` with `--spec 1` (sampled drafts at temperature > 0),
  64K context and the cache prior;
- run as a temporary unit on port 8090 and stopped afterwards (`sw86.sh`).

## First run: every request failed

`bench/server_smoke.py` stopped at its first chat request, HTTP 500. Reproduced with curl:
- the first request: `cudaMalloc checkpoint: out of memory` (`first-run-error-1.txt`);
- the next: `embed: out of memory` (`first-run-error-2.txt`).

The causes:
- **VRAM exhaustion.**
  - sw64 had lowered the VRAM left after the expert cache from 1,024 to 256 MiB, measured with
    `fr_bench`, which allocates everything up front.
  - The engine allocates the recurrent-state checkpoints (GDN states, conv histories, indexer
    rings, and the MTP head's) on the first request, after the cache has taken the free VRAM.
- **A sticky error.** The failed allocation left CUDA's last error set, so the next request
  failed at an unrelated `cudaGetLastError()` check.
- **Nothing logged.** The server returned the engine's per-request errors to the client but did
  not log them.

The fixes:
- `ForwardRef::reserve_checkpoint()` and `MtpHead::reserve_checkpoint()` run at startup, before
  the cache sizes itself.
- The engine clears the CUDA error state after a failed request.
- The engine's default reserve is 512 MiB (`fr_bench` keeps 256).
- The server logs engine errors.

## Second run: `server_smoke.py` passes 11 of 11 (`smoke.txt`)

- **The checks:** models; chat without thinking ("391"); streamed chat with reasoning; a tool
  call; the tool-result turn with prefix reuse (368 of 398 prompt tokens cached); a stop string;
  raw completion; Anthropic messages with thinking blocks; Anthropic streamed tool_use;
  count_tokens; a client that disconnects mid-stream (the next request answered 0.2 s later).
- **Speed:** streaming at 80.4 tok/s after the first token at short context (sw35: 69.0).
- **Startup:** about 275 s to ready (the weights and the expert arena load from disk).

The regression came from a change that was validated in `fr_bench` only. `bench/server_smoke.py`
now belongs in the check list for any change to VRAM budgeting or the engine.
