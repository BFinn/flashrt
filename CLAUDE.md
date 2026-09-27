# flashrt: working notes for Claude

Read `docs/design.md` (architecture, engine protocol, phase gates) and
`docs/clean-room.md` (what may and may not be copied) before writing code.
Machine-specific details live in `CLAUDE.local.md`, which is not committed.

## Workflow: edit here, build and measure on the GPU box

- **This checkout (the Mac) is the source of truth.** Edit, commit, and push to GitHub
  from here. The research vault is reachable only from here.
- **The GPU box is a build and benchmark target.** It cannot reach GitHub or the vault.
  - Deploy with `git push the box main`. The remote checkout updates its files on push.
  - Then build and run over ssh.
- **Bring results back.** Copy anything worth keeping (benchmark JSON, logs, summaries)
  into `bench/results/<date>-<topic>/` with `scp`, and commit it here.

## Build (on the GPU box)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build
source ~/.cargo/env && cargo build --release --manifest-path server/Cargo.toml
```

## Rules

- **Clean room.** Strata's ideas are fine, its source code is not. Do not open Strata's
  `src/` while writing flashrt code. ggml, llama.cpp and ik_llama.cpp (MIT), vLLM and
  SGLang (Apache-2.0), and CUTLASS (BSD) may be vendored with their notices.
- **Every speed claim comes from `bench/`,** with the arm, depth, run count and date. The
  phase gates in `docs/design.md` are the targets. Quote measured numbers, not estimates,
  unless an estimate is labelled as one.
- **Correctness gate.** Changes that affect outputs need a KL-divergence check against the
  llama.cpp reference before any speed number counts.
- **Generality waits.** Build qwen4exp end to end first. Design the architecture add-on API
  when a second architecture arrives.
- **Commit trailer.** End commit messages with the session attribution line the harness
  provides.
