# flashrt: notes for coding agents

Read these before writing code:
- `docs/engine.md`: the engine as built. It covers the decode, speculation and prefill flows,
  why each piece is shaped as it is (with links to the evidence), what was rejected, the
  runbook, and known issues. **Start here.**
- `docs/sweet-spots.md`: the best configurations measured, where each tuning track stopped and
  why, the untested paths ranked, and every `FLASHRT_*` toggle.
- `docs/design.md`: architecture, engine protocol, phase gates and their status.
- `docs/background.md`: what was measured before flashrt existed, and which ideas were rejected.
- `docs/interfaces.md`: how generic flashrt is, and the C++ seams.
- `docs/clean-room.md`: what may and may not be copied.

Machine-specific setup (where to build and measure, paths, box rules) lives in
`CLAUDE.local.md`, which is not committed.

## Build and test

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build
cargo build --release --manifest-path server/Cargo.toml
(cd build && ctest --output-on-failure)            # FLASHRT_TEST_MODEL=<gguf> for gemm and moe_q2
cargo test --release --manifest-path server/Cargo.toml
```

## Rules

- **Clean room.**
  - Strata's ideas are fine, its source code is not. Do not open Strata's `src/` while
    writing flashrt code.
  - These may be vendored with their notices: ggml, llama.cpp and ik_llama.cpp (MIT), vLLM
    and SGLang (Apache-2.0), CUTLASS (BSD).
- **Every speed claim comes from `bench/`,** with the arm, depth, run count and date.
  - Each run's script, logs and a README go in `bench/results/<date>-<topic>/`.
  - Quote measured numbers, not estimates, unless an estimate is labelled as one.
  - Paths in scripts use the placeholders defined in `docs/engine.md` (Runbook), never a
    home directory.
- **Correctness gate.** Changes that affect outputs need a KL-divergence check against the
  llama.cpp reference (`tools/fr_kld`) before any speed number counts.
- **Paired decode comparisons.** A/B decode runs use `fr_bench --teacher`, so every arm routes
  the same tokens. Sampled drafts are the exception: teacher forcing keeps argmax drafts, so
  measure them on sampled runs over 6+ windows.
- **The server is part of the check.** After a change to the engine or to VRAM budgeting, run
  `bench/server_smoke.py` against a server on the current build. `fr_bench` allocates
  differently and cannot catch everything.
- **Generality waits.** Build qwen4exp end to end first. Design the architecture add-on API
  when a second architecture arrives.
- **Commit trailer.** End commit messages with the session attribution line the harness
  provides.
