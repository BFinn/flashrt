# sw113: named constants, doorbell flags on the memory model, build flags (2026-09-30)

Phase 4 step 4 of `docs/improvement-plan.md` (H-4, H-5, H-6) and E-4, in c6dbe8f:
- **H-4:** the ggml type ids flashrt handles, its own converted types (Q3R, Q8P) and its file
  magics (state file, cache prior) are named in `core/formats.hpp`. Every bare number is gone.
- **H-5:** the mailbox flags between the GPU and the miss server are system-scope atomics.
  - On the GPU they use `cuda::atomic_ref`. `k_route` publishes a block's x and route record with
    one release store after `__syncthreads()`, instead of a fence in every thread and a
    `volatile` store. `k_moe_combine_db` polls relaxed, then fences acquire.
  - The host uses `std::atomic_ref`, the same acquire/release as before.
- **H-6:**
  - `FLASHRT_NATIVE` (`-march=native`) is off by default, and the documented build turns it on.
  - Configuring with a CUDA architecture below sm_90 fails with a message; "native" and "all"
    warn.
  - CMake says which kernels build with `--use_fast_math`, and why 177 is suppressed.
- **E-4:** on quit the engine joins its reader thread before the queue goes away. It used to
  detach it. The session is then freed explicitly.
- E-3 was already in place: the host's wait for the miss server gives up after 10 s.

## Checks

| Check | Result |
|---|---|
| Configure with `CMAKE_CUDA_ARCHITECTURES=86` / `native` | fails with the message / warns |
| Build, ctest | clean, 19 / 19 |
| sw112's fingerprint (7 runs) | **identical** to sw112's `before` (`sw113-fingerprint.txt`, logs in `fingerprint/`) |
| `engine_smoke --reuse` / `--faults` with the head | 5 / 5 and 18 PASS |
| `server_smoke.py` | 15 / 15 |

## Speed (H-5 changes the decode's synchronization)

`fr_bench --teacher` (same tokens every run), q8 KV with the hot set, old binary (c6dbe8f's
parent) against new, interleaved, one decode per run. Logs are in `ab/`.

| tok/s, mean ± sd | old | new | change |
|---|---:|---:|---:|
| window 9's 32K prompt, 320 tokens, plain (n = 3) | 87.18 ± 0.53 | 88.08 ± 0.15 | +1.0% |
| the same with the head, `--spec 2` (n = 12) | 109.40 ± 1.94 | 108.44 ± 2.14 | −0.9% (t = −1.2) |
| wikitext 8K, 256 tokens, plain (n = 3) | 108.34 ± 1.58 | 110.03 ± 0.45 | +1.6% |

- The head arm's first 6 pairs read −1.9%, on two slow new runs; 6 more pairs (`sw113b.sh`,
  `sw113c.out`) brought it to −0.9%, inside the noise.
- The plain arms read slightly faster. Fewer system fences per layer is a plausible cause.
- **No measurable change.** The head arm's run-to-run spread (106-111) is larger than any
  difference here.

## Files

- `sw113.sh`: the whole check (configure, build, ctest, fingerprint, A/B, smoke). `sw113.out` is its
  output.
- `sw113b.sh`: more head-arm pairs (`sw113b.out`: 3 pairs; `sw113c.out`: 6 pairs, `ARMS=...`).
