# sw129: phase 6, reproducible by someone else (2026-10-01)

What was added (`docs/improvement-plan.md`, phase 6):
- **X-1, `bench/reference/`:** window 9's token ids, the drafter's vocabulary ranking, the llama.cpp
  reference as a patch on upstream ec92815 (it applies cleanly; the tree behind window 9 and the
  KLD base), `kld-base.sh` to rebuild the KLD base, and the Strata and wikitext pins.
- **X-2:**
  - `bench/run.sh`, plain bash: `preflight | build | test | decode | window9 | kld | server | all`;
  - a `Dockerfile` (CUDA 12.9 devel to runtime), built by the `docker` workflow;
  - the README's "Will it run on my machine".
- **Server metrics:** `/metrics` in Prometheus's text format, summed over the engine's `done`
  events. Those events gained `cache.miss_ms` and `cache.slots`, optional to a reader
  (`docs/design.md`).

## Checks (`sw129.sh`)

| Check | Result |
|---|---|
| `bench/run.sh all` on the development box (`run-all.out`) | preflight ok; build; ctest 21 / 21; decode; window 9 (n = 1); server smoke **all checks passed**, the new `metrics` check included |
| `bench/run.sh kld` (`run-kld.out`) | KLD mean **0.008910**, same top token 96.581%: sw126's values exactly |
| `engine_smoke.py --faults`, plain and `--spec 2` (`faults-*.txt`) | 0 checks failed in either |
| Server: `cargo test`, `cargo clippy -D warnings` | 28 / 28 (2 new: `metrics::tests`), clean |
| GitHub `docker` workflow (run 36821456890) | the image builds (5.5 min); `flashrt-server --help` runs in it; no missing libraries |
| GitHub `ci` workflow | green |

**run.sh's window 9** (n = 1) against sw128 (n = 5), decode tok/s at 1K / 32K / 134K / 250K:
- P: 101.2 / 94.4 / 95.2 / 93.6 (sw128: 103.5 ± 2.1 / 95.5 ± 0.8 / 93.9 ± 0.9 / 91.9 ± 1.7);
- G: 132.1 / 111.4 / 103.0 / 102.7 (127.9 ± 4.4 / 111.3 ± 4.1 / 103.7 ± 2.3 / 104.9 ± 2.7);
- S: 123.9 / 112.6 / 112.8 / 109.5 (118.8 ± 16.5 / 95.5 ± 17.0 / 106.7 ± 2.1 / 106.4 ± 2.9; it
  samples its own text).

**run.sh's decode step** (`fr_bench`, fresh prefill of window 9's 32K prompt, 256 tokens):
- prefill 5,958 tok/s; plain 95.0 tok/s; `--spec 2` 97.4 tok/s (75.4% hits).
- The engine reaches 111.4 on the same prompt with `--spec 2` (81.7% hits). The two tools differ
  in more than the run length, and the cause is not isolated, so run.sh quotes each tool's own
  reference values.

**Slots in the engine** (the new `done.cache.slots`): 8,431-8,504 without the head and 7,536-7,636
with it, against fr_bench's 8,809 (245K). The engine reserves 512 MiB and allocates the checkpoints
up front (sw86).
