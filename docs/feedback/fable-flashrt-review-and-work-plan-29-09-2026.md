# flashrt: external review and work plan (2026-09-29)

**Repository:** https://github.com/BFinn/flashrt, reviewed at commit `27856aa` (main, 2026-09-29).
All file:line references are as of that commit.

**Purpose of this document.** A code review of flashrt turned into a backlog that a coding agent
can execute item by item. Each item says where the problem is, what to change, and how to verify
it. Items carry an ID (R = results/benchmarks, E = engine robustness, S = server, T = tests/CI,
H = hygiene/structure, G = generality, P = performance, X = reproducibility), a priority (P0-P3)
and an effort estimate (S < 2 h, M < 1 day, L > 1 day). A suggested order is at the end.

## Ground rules for the agent (from the repo's own `CLAUDE.md`; they still apply)

- Read `docs/engine.md` first, then `docs/sweet-spots.md` and `docs/design.md`.
- Every speed claim comes from `bench/`: each run gets a `bench/results/<date>-<topic>/` folder
  with the script, logs and a README; run `bench/scrub.py <folder>` before committing.
- Output-affecting changes need the KLD gate (`tools/fr_kld`, 2 x 8K wikitext, `--fast`) before
  any speed number counts; adopted configs today measure 0.0082-0.0092.
- Decode A/B runs are teacher-forced (`fr_bench --teacher`); sampled drafts are measured over
  6+ windows.
- After any change to the engine or VRAM budgeting, run `bench/server_smoke.py` against a
  server on the current build.
- Clean room: Strata's ideas are fine, its source is not.
- End commit messages with the session attribution line the harness provides.
- Machine paths use the placeholders in `docs/engine.md` (Runbook): `$FLASHRT`, `$MODELS`,
  `$BENCH`, `$DATA`, `$STRATA`, `$LLAMA_CPP`.

## 1. Review summary

**What is good.** The engine works end to end for its one target (Qwen3.8-Flash-Next, GSQ-RCO
Q2_0, RTX 5080 16 GB + Ryzen 9 7900X): decode, MTP speculation with exact speculative sampling,
chunked prefill, engine process, OpenAI/Anthropic server. Every number in the README traces to
a raw log in `bench/results/` (the sw91 means were re-derived from `sw91.out`; all 12 cells
match). The KLD gate (0.0082-0.0092 against llama.cpp's own path-to-path band of 0.0078-0.0086)
is a credible correctness story. The kernels are competent hand-written tensor-core code
(`k_attn_tc`, `moe_q2`, the chunked GDN, the AVX-512 Q2_0 kernel). The evidence trail (97 result
folders, 85 READMEs, a decision table with a sweep behind each row, a rejected-ideas list) is
unusually complete. 2.2-2.6x over llama.cpp at every depth holds.

**What is not.** Three findings change the picture:

1. The headline comparison against Strata was not run on the same protocol: flashrt got no
   prefix reuse at any depth while both baselines did, and the Strata baseline build flags its
   own outputs as incorrect. Neither is disclosed (section 2, R-1..R-4).
2. The "generic runtime" described in `docs/design.md` and `docs/interfaces.md` does not exist
   in the code. `core/` is 1,189 lines of GGUF/arena/pool/JSON/platform/RowReader; the expert
   cache, cache policy, doorbells, miss server, windows, speculation and KV all live under
   `arch/qwen4exp/` and `kernels/cuda/sample.cu`, with shapes hard-wired (section 6).
3. Robustness gaps that a server will hit: a failed request leaves the engine session in an
   inconsistent state; the Rust server cannot detect a dead engine; a tool-call parser slice can
   panic on multibyte text (sections 3, 4).

Plus a layer of agent-loop debt: stale header comments, a 3,804-line `blocks.cu`, 20
`FLASHRT_*` toggles keeping superseded kernels alive, CI that compiles no CUDA and runs 3 of 14
tests (sections 5, 7).

## 2. Results and benchmark integrity (R)

### R-1 (P0, M) flashrt got no prefix reuse on window 9's protocol; the baselines did

- **Evidence.** `bench/results/2026-09-29-sw91-depthbench/sw91.out`: every SUMMARY row has
  `new == depth`; `G1.log:3-6` and `P1.log` read `prefill of 250711 tokens` at the 250K depth.
  Strata reused prefixes: `bench/results/2026-09-27-w9-validation/W9-G-strata-greedy-r1-20260927-114428.log:36,38`
  ("134029 tokens = 32768 reused + 101261 read", "250712 = 131072 reused + 119640 read").
  llama.cpp reused via LCP (`w9-all.out:4-5`, 103,288 / 118,735 new tokens).
- **Why.** `engine/session.cpp:279-295` reuses either the whole previous sequence or the one
  checkpoint, which is saved at the end of the prompt (`session.cpp:325`). Window 9's prompts
  all end with the same instruction sentence, so the shared prefix is shorter than the
  checkpoint and the engine falls to `reset()` at `session.cpp:290`. This is also a product
  defect: any chat whose history grows in front of a fixed tail (system prompt + instruction,
  tool results, etc.) re-prefills everything.
- **Consequence for the results.** Strata carried an expert cache already warmed by the
  previous depth's 384-token answer; flashrt refilled its cache from whole-prompt routing counts
  at every depth. Part of the 5-13% "warm-up gap" attributed to the synthetic prompt
  (`README.md:40-42`, sw88) is this protocol difference.
- **Do.**
  1. Periodic checkpoints during prefill: save the recurrent state (GDN S + conv, PLE history,
     indexer ring; the KV caches need nothing) every N tokens (start with every chunk, or every
     4,096 tokens) into a small ring in host RAM. `ForwardRef::ckpt_parts()`
     (`forward_ref.cu:255-266`) already enumerates the pieces; the same is needed in `MtpHead`
     (`save_checkpoint(h_carry)`). Restore to the latest checkpoint at or before the longest
     common prefix, then prefill only the tail.
  2. On a reuse, keep the warmed expert cache; only `refill_cache()` when the new tokens are
     >= 4,096 (`session.cpp:328`), and even then blend the live counts rather than replacing them.
  3. Correct `bench/results/2026-09-29-sw87-depthbench/README.md:14-16` ("None of the engines
     reused a prefix" is wrong for Strata) and note the difference in the sw91 README and in
     `README.md` (Results) until the re-run replaces those numbers.
- **Verify.** `bench/engine_smoke.py` gains a case: prompt A, then prompt A' = A[:-k] + tail
  (k small) reports `reused >= len(A) - k - N`. Re-run `bench/results/2026-09-29-sw91-depthbench/sw91.sh`
  and confirm the `new` column shows reuse at 134K and 250K. KLD gate unchanged.

### R-2 (P0, S) The Strata baseline build flags its own output as incorrect

- **Evidence.** `W9-G-strata-greedy-r1-20260927-114428.log:12-15`: "*** WARNING: --expert-cache
  is enabled and the GPU hit path is NOT CORRECT ... Any timing from this run is real; any
  OUTPUT from it is not. ***".
- **Do.** State it wherever the Strata numbers appear (`README.md:30,33`, `docs/sweet-spots.md:53`,
  `docs/engine.md:337`, `bench/README.md`). Its acceptance counts (e.g. 197/314) come from a
  wrong target, so its greedy tok/s with MTP is not a clean number. If a fixed Strata exists,
  re-run the baseline; otherwise present Strata's numbers as "as measured, with this caveat".

### R-3 (P1, S) Strata's prefill figure counts reused tokens

- **Evidence.** `README.md:43-45` says "Strata: 1,170-2,030". The 2,031 figure is
  250,712 / 123.4 s, i.e. it counts the 131,072 reused tokens. Strata's own log gives
  1,173 / 1,089 / 969 tok/s on new tokens (same log, lines 34-38).
- **Do.** Quote per-new-token prefill for every engine. flashrt's own README range
  "5,570-5,950" also mixes two measures (engine `prompt_ms` vs window 9's TTFT delta); use one
  (`sw91-summary.txt` has both) and say which. The corrected comparison is more favourable to
  flashrt (about 5x, not 2.7-5x).

### R-4 (P1, S) Small-n sampled cells shown without spread

- **Evidence.** `sw91-summary.txt`: t=1.0 at 32K is 76.4 / 65.4 / 91.4 (77.7 +- 13.1). The
  README table drops every sd. "Level with Strata at 32K" and the 1K "win" (82.2 +- 4.2 vs
  80.5 +- 4.1) are within noise.
- **Do.** Print mean +- sd and n in the README table; run n >= 5 for the sampled arms (also for
  Strata's if it is re-run); interleave arms within a session; log GPU clocks and temperature
  (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu`) in the summary lines.

### R-5 (P2, S) Disclose what was tuned on the benchmark

The swap budget 32 default was chosen on window 9's protocol (sw89/sw90) and costs 1% on
natural text (`docs/sweet-spots.md:20`). It is disclosed in the sweet-spots doc; say it in the
README next to the table, or make the budget adaptive (P-2) so the point is moot.

### R-6 (P2, S) Natural-text numbers have no baseline beside them

`README.md:47-52` gives fr_bench wikitext numbers (143 / 97 tok/s) with no Strata number, while
`bench/results/2026-09-27-p0c/README.md:79` has Strata greedy at 95.7 / 103.0 (131K / 245K,
n=2). Either put them side by side with the protocol differences stated, or drop the
"continuing natural text" section from the README and leave it in the docs. Also: "the expert
cache hits 90-95%" holds at 32K (92.9-96.4%) but not at 245K (83.3-88.8%, `sw85`); fix the
sentence.

## 3. Engine robustness (E)

### E-1 (P0, M) A failed request leaves the session inconsistent

- **Where.** `engine/session.cpp:304` (`cache_release()` before chunked prefill),
  `session.cpp:318,327` (`cache_restore()` only on the success/cancel paths),
  `engine/main.cpp:183-186` (the catch only reports and clears `cudaGetLastError`).
- **Problem.** Any exception between 304 and 327 (KV full, doorbell timeout, OOM, a bad token)
  leaves `cache.slots == nullptr`, `cache_filled == false`, `m.fwd->pos()` ahead of `m.seq`,
  the MTP head's KV possibly ahead of the target, and a window possibly uncommitted
  (`window_pos0_ >= 0`, which makes the next `forward_window` throw at `forward_ref.cu:229`).
  The next request then fails or computes on stale state.
- **Do.** Wrap the body of `Session::generate` in try/catch (or an RAII guard) that, on any
  exception: commits or discards an open window (`commit(0)` is the documented undo), calls
  `cache_restore()` if the cache was released, synchronises the stream, and resets
  `m.seq`/`fwd`/`mtp`/`h_carry` to a known state (simplest: `reset()` everything and clear
  `m.seq`, so the next request starts cold but correct). Rethrow after cleanup. Make
  `chunk_min`, `max_ctx` and `prompt longer than the context` checks happen before any state is
  touched.
- **Verify.** New `bench/engine_smoke.py` cases: (a) a prompt one token longer than `--ctx`
  followed by a normal request; (b) a `stop` during prefill followed by a normal request;
  (c) an out-of-range token id mid-prompt followed by a normal request. All three must answer the
  second request with the same tokens as a fresh engine would (greedy).

### E-2 (P1, S) Validate requests before mutating state

`session.cpp:273-277` checks the prompt, but `to_request` (`main.cpp:55-70`) accepts any
`max_new`, `top_k` (the sampler throws at `kernels/cuda/sample.cu:196` for top_k > 64, after
prefill), negative temperatures, and `seed` as a double. Validate and clamp in `to_request`, and
return the error before the GPU does any work.

### E-3 (P2, S) `doorbell_end_token` busy-waits with no bound

`arch/qwen4exp/moe_fast.cu:951-952` spins on `served != seq` forever if the server thread dies
without setting `failed`. Add a timeout consistent with the GPU side's 10 s and report the same
diagnostic string.

### E-4 (P2, S) Engine process lifecycle

`engine/main.cpp:190` detaches the reader thread and returns while a request could still be
running in the CUDA stream; and `{"op":"quit"}` is never sent by the server (S-1). Join or
cancel cleanly, and destroy the session before exit so the miss server and CpuPool threads stop
(their destructors are correct; they are just never reached on the quit path).

## 4. Rust server (S) -- `server/src/`

### S-1 (P0, M) Engine death is invisible; requests hang or get 400s

- **Where.** `engine.rs:75` (the reader loop ends on the first read error and nothing records
  it), `engine.rs:56-57` (stderr inherited, `kill_on_drop`, no `child.wait()`),
  `openai.rs:140,243` and `anthropic.rs:170` (every `chat::start` error, including a broken
  stdin, maps to `400 invalid_request_error`), `main.rs:191` (`/health` always says ok).
- **Do.** Keep an `alive: AtomicBool` set false when the reader ends or `wait()` returns;
  `generate()` returns an "engine down" error when it is false; map that to 503; `/health`
  reports it; either exit the process (let the supervisor restart) or respawn the engine and
  reload. Send `{"op":"quit"}` on shutdown (`axum::serve` at `main.rs:195` has no graceful
  shutdown today).
- **Verify.** Unit test with a fake engine script that exits after `ready`: the next request
  must return 503 within a second, not hang.

### S-2 (P0, S) Tool-call parser can panic on multibyte text

`chat.rs:107`: after a value terminated by `</function>` (11 bytes) the cursor advances by
`"</parameter>".len()` (12). If the next byte is inside a multibyte character, the slice is not
on a char boundary and the task panics; the client sees "generation ended without a result",
nobody sends `stop`, and the GPU runs to `max_new`. Advance by the length of the terminator that
was actually found, and fuzz `parse_tool_call` (a `proptest`/`cargo-fuzz` target, or a loop over
random UTF-8 around the tags).

### S-3 (P1, M) Tokenization and rendering block tokio workers under a global lock

`tokenizer.rs:164` holds the cache mutex for the whole `encode`; `bpe` (`tokenizer.rs:181-197`)
is O(n^2) via `Vec::remove` over unbounded pre-tokens. A 1 MB run of letters wedges a worker and
serialises every other request's tokenization. Move render + encode into `spawn_blocking`, take
the lock per word, cap the word length (or use a heap-based merge), and bound the cache by bytes.

### S-4 (P1, S) Client disconnect while queued is not a cancellation

`chat.rs:234-242`: `stop` is sent only when a `send!` fails, i.e. after the first token. A client
that leaves while queued still pays the whole prefill. Add a `tokio::select!` branch on
`tx.closed()` (or the request's cancellation token) that sends `stop` immediately.

### S-5 (P1, S) Sampling parameter mapping

`chat.rs:212`: `top_k` is clamped to 1..64, so `top_k: 0` (the usual "disabled") becomes greedy
and values above 64 are capped silently. Treat 0 (and absent) as "off" = 64 (the kernel's
`kMaxTopK`), and reject > 64 with a 400 that says so.

### S-6 (P1, S) Special tokens leak into output text

`chat.rs:279` decodes every id through `token_bytes`; `<think>`, a stray `</think>` in Content
mode, `<|im_start|>` etc. appear literally. Skip ids whose `tokenizer.ggml.token_type` is
control/special (llama.cpp's behaviour), except the ones the section state machine consumes.

### S-7 (P2, S) Tokenizer guards

`tokenizer.rs:50` checks `tokenizer.ggml.model == "gpt2"` but never `tokenizer.ggml.pre`; a
qwen2/llama-bpe GGUF tokenizes silently wrong. Assert the pre-tokenizer type the regex was copied
for. Turn `--check-tokenizer` (`main.rs`) into a CI test with a checked-in fixture (text + ids
for a few hundred lines including CJK, emoji, code, `<|im_start|>` in user text). `eos`
defaults to 0 silently; fail instead.

### S-8 (P2, S) Request-shape issues

`openai.rs:38-45` validates array content but does not flatten it (success depends on the
template); unparsable `tool_calls.arguments` become `{}` silently (`openai.rs:55`) -- keep the
string. `template.rs:79-81`: `chat_template_kwargs` can override `messages`, `tools` and
`add_generation_prompt` because extras are inserted last; insert them first.

### S-9 (P3, S) Hardening

Explicit body limit (axum's default 2 MiB may reject legitimate 200K-token prompts; set it
deliberately), constant-time API key compare (`main.rs:200-204`), a queue cap with 429, a
per-request timeout, CORS. Stop strings apply only to answer text (`chat.rs:250,283` pass `&[]`)
and cannot match leading whitespace; document or fix.

## 5. Tests and CI (T)

Today: 14 ctest targets; `.github/workflows/ci.yml` compiles no CUDA and runs 3 of 14 (cpu_pool,
q2_0, moe_cpu) plus `cargo test`/clippy. The same agent wrote the code and the tests, so the
tests are self-grading; independent references matter more than usual.

### T-1 (P0, M) Compile the CUDA tree in CI

No GPU is needed to compile. Install the CUDA toolkit on the hosted runner (NVIDIA's apt repo
or the `nvidia/cuda:12.x-devel` container), build with
`-DCMAKE_CUDA_ARCHITECTURES=120 -DFLASHRT_NATIVE=OFF`, and run the CPU tests from that build. A
typo in `blocks.cu` currently passes CI green.

### T-2 (P1, M) Sanitizer jobs

ASan+UBSan for the CPU build and tests; TSan for `test_cpu_pool` and `test_moe_cpu` (the
`CpuPool` generation/futex protocol and the miss server are exactly what TSan finds). Make
`test_q2_0` return ctest's skip code (77) when AVX-512 VNNI is absent instead of silently
skipping (`tests/test_q2_0.cpp:113`), so coverage is visible.

### T-3 (P1, M) Pure-CPU unit tests for the untested core

- `core/json.cpp` (217 lines, the engine protocol): escapes, `\u` surrogate pairs (`json.cpp:134-137`),
  malformed input throws, nesting limit, number edge cases, round trip of `dump`.
- `core/gguf.cpp`: a synthetic 2-shard GGUF written by the test (tiny tensors), metadata merge,
  offset/bytes computation, truncated file throws.
- `core/row_reader.cpp`: page-merging of rows that share a 4 KiB page (`row_reader.cpp:120`
  region) on a temp file; rows crossing a page boundary.
- `qsa_state_io` / `save_state` / `load_state` (`forward_ref.cu:95-141`): round trip, wrong
  model rejected, fp16 file into q8 cache.
- `expert_arena`: stride/alignment.

### T-4 (P1, M) GPU tests that are missing for the decode path

- `test_gdn.cpp` compares chunked vs column kernels (GPU vs GPU, tol 1e-2) and never touches the
  decode kernel `k_gdn_delta_reg` (`blocks.cu:788`, used for all T < 16 and every window) or
  `gdn_rewind`. Add a CPU double reference of the delta rule; test T = 1..8; test rewind: run T
  tokens, `gdn_rewind` to n, compare the state to a fresh n-token run (bit-exact expected).
  Same pattern for `ple_rewind`.
- `test_spec_sample.cpp:64` uses one draft row; `spec_k = 2` is the default
  (`engine/session.hpp:31`). Extend to rows = 3..4 (rejection at j > 0, residual sampling) and
  the `ids` remap of the trimmed vocabulary.
- `test_sample.cpp`: add `min_p > 0`, `temperature != 1`, `top_k in {1, 64}`, `top_p = 1`.
- `test_linear_multi.cpp`: the epilogues (`epi` 1 and 2, `blocks.hpp:93`) are untested.
- `test_moe_q.cpp`: Q8_0 only; production is Q2_0 (type 42). Add it.
- `test_moe_cpu.cpp`: add `n_tok = 4` (the maximum) and an unrouted token (the zeros path).

### T-5 (P1, M) Test the CacheManager policy and the doorbell protocol in isolation

The decayed-LFU policy (admit 2.0, margin 1.5, decay 0.7 / 4 tokens, budget) is pure host logic
in `moe_fast.cu` and can be driven by a synthetic access stream with an oracle (e.g. compare the
hit rate to `tools/cache_sim.py` on the same trace). The mailbox protocol can be tested with a
stub miss server that delays, drops, or errors, checking the 10 s timeout path writes the error
word and `doorbell_end_token` reports it.

### T-6 (P2, S) ctest hygiene and a GPU workflow

Add `LABELS gpu`/`model` and `TIMEOUT` to the tests; split the timing loops out of `test_gdn`,
`test_hc_decode`, `test_linear_multi`, `test_moe_hits`, `test_moe_q2` into `bench_*` targets so
`ctest` is short and not load-sensitive; a workflow gated on `runs-on: [self-hosted, gpu]` for
the box. `test_gemm.cpp:80,84` hard-codes scratch for K <= 12288; derive from the tensors.
`test_gemv` (the only ground truth for the ggml wrappers) needs a llama.cpp tree; build it from
the vendored headers plus a small quantize copy so it runs anywhere.

### T-7 (P2, S) Server tests

There are 9 unit tests, none for `chat::start`'s state machine, `normalize_messages`, `to_chat`,
SSE framing, `engine.rs` or `gguf.rs`. Add a fake-engine protocol test (a script that speaks the
JSON-lines protocol), a Qwen chat-template fixture test (the real template, rendered against a
checked-in expected string), and SSE framing tests. `bench/server_smoke.py` stays as the
end-to-end check on the box.

## 6. Generality (G) -- what would a second model cost

**State today.** `docs/interfaces.md` describes `ArchModule`, `QuantPack`, `MoeBlock`,
`ModelSpec`, `StateSpec`; none exists (zero hits in the tree). `engine/session.cpp:29` does
`using namespace qwen4exp` and instantiates `ForwardRef` and `MtpHead` directly. Everything
`docs/design.md:31` attributes to `core/` (expert cache, cache policy, doorbells, miss server,
window scheduler, speculative sampling, KV) lives in `arch/qwen4exp/moe_fast.cu`, `blocks.cu`
and `kernels/cuda/sample.cu`.

**Hard-wired shapes** (each is a throw or a silent assumption):

| Assumption | Where |
|---|---|
| Experts are Q2_0 (type 42), planar-repacked; arena, cache slots, CPU kernel, dp4a hits, `moe_q2` all assume it | `arch/qwen4exp/spec.cpp` (`t.type != 42`), `quant/q2_0/*`, `moe_fast.cu`, `kernels/cuda/moe_q2.cu` |
| `d_model % 512 == 0`, `d_ff <= 4096`, top-k <= 16, E <= 1024 | `moe_fast.cu:561,974`, `blocks.cu:1915` |
| Hyper-connections: `hc == 4`, rank 320 for the fused decode kernels | `blocks.cu:583,1647,1719` |
| Attention head_dim 256, GQA group <= 16, q8 hot set needs head_dim 256 | `blocks.cu:3165,3303` |
| GDN state 128 (windows, prefill), conv <= 8 | `blocks.cu:1983,3015,1982` |
| Indexer dim 128 in graph mode | `blocks.cu:3301` |
| Verify windows <= 8 tokens | `moe_fast.cu:33,622`, `blocks.hpp:79` |
| Token embedding Q3_K for graph mode | `blocks.cu:3630` |
| Q3_K matrices with K <= 8192 for Q3R | `kernels/cuda/q3r.cu:267` |

`CLAUDE.md`'s "generality waits" is a legitimate choice. The problem is that `docs/design.md`
(Layers table) and `docs/interfaces.md` describe the intent as if built.

### G-1 (P1, S) Make the docs describe what exists

Rewrite `docs/design.md:27-40` to say which layer holds what today, and mark `docs/interfaces.md`
as a design sketch with a "status: not implemented" line and the table above.

### G-2 (P2, L) Extract the first real seam: the offload machinery into `core/`

Move `ExpertCache`, `expert_cache_fill`, `CacheManager`/`CachePolicyConfig`, `MissServer`,
the mailboxes and `doorbell_*` out of `arch/qwen4exp/moe_fast.cu` into `core/expert_cache.*`
and `core/doorbell.*`, parameterised by an expert-shape struct (`d_model`, `d_ff`, `n_expert`,
`top_k`, blob bytes) and a CPU expert callback (today `q2_0::moe_cpu`). Keep `k_route`,
`moe_hits` and `k_moe_combine_db` in the arch (they are shape-specialised). This is the piece
that is the same for every offloaded MoE, and it is where a second model would start. Do it
with the KLD gate and a teacher-forced A/B to prove nothing moved.

### G-3 (P3, L) A second architecture as the forcing function

Do not design the `ArchModule` API in the abstract. Pick a second GGUF MoE model that fits the
box (a GQA + SwiGLU MoE with Q4 experts is the closest cheap neighbour) and let its needs define
the seam: `Spec`/`plan` become per-arch, `Forward` becomes an interface with
`forward/forward_window/commit/checkpoint`, and `quant/` gains a second pack (`QuantPack` as
sketched: `blob_bytes`, `repack`, `cpu_expert`, `gpu_hits`, `gpu_prefill`). Expect weeks, not
days; G-2 first makes it tractable.

## 7. Hygiene and structure (H)

### H-1 (P1, M) Split `arch/qwen4exp/blocks.cu` (3,804 lines)

The namespace is reopened five times (`blocks.cu:23,1934,2080,3429,3536`): hyper-connections and
linear, GDN, QSA, the reference MoE, PLE and embedding are concatenated. Split into
`hc.cu`, `gdn.cu`, `qsa.cu`, `moe_ref.cu`, `ple.cu` (and a `blocks_common.cuh` for `ck`, the
scratch and the mma helpers). No behaviour change; the KLD gate and `fr_parity` prove it.

### H-2 (P1, S) Stale comments and names

- `arch/qwen4exp/forward_ref.hpp:2-4` says "Every routed expert runs on the CPU (no VRAM expert
  cache yet)"; `blocks.hpp:2-4` says "Fusion and graph capture come after parity". Both are
  years of work out of date (in agent time).
- `ForwardRef` is the whole forward (graphs, windows, chunks, checkpoints). Rename it
  (`Forward`, `Qwen4expForward`) and rewrite the header comment to describe the three modes.
- `engine.md:1` and `sweet-spots.md:1` carry a date in the title; keep a changelog line instead.

### H-3 (P1, M) Retire superseded kernels behind the `FLASHRT_*` toggles

20 env toggles (`docs/sweet-spots.md:120-143`) keep v1 and v2 side by side: `k_hc_down`/`_down2`,
`k_hc_up_mix`/`_mix2`, `k_gdn_delta`/`_reg`/`_col`, `k_idx_scores`/`128`/`_tc`,
`k_route_topk`/`_w`, `moe_q2` vs MMQ, etc. The evidence for each decision is in
`bench/results`. Delete the losing paths (keep the A/B result folder as the record), and keep a
toggle only where the doc says the alternative is still useful (`FLASHRT_MOE_AB64` off for KLD,
`FLASHRT_HC_Q8` up/down, `FLASHRT_ARGMAX_DRAFTS`). Fewer paths means fewer untested branches.

### H-4 (P2, S) Magic numbers and constants

`spec.cpp` (`t.type != 42`), state-file magics in `forward_ref.cu:101`, the cache-prior magic in
`session.cpp:124`, `kTypeQ3R = 1000` / `kTypeQ8P = 1001` (`gpu_weights.hpp`): name them in one
header (`core/ggml_types.hpp`, `core/file_magics.hpp`).

### H-5 (P2, S) Doorbells on the documented memory model

`moe_fast.cu:490,519-533` use `volatile` + `__threadfence_system()` for the mailbox flags;
`moe_fast.cu:854,908` use `__atomic_load_n/__atomic_store_n` on the host. It works on this
box; `cuda::atomic_ref<uint32_t, cuda::thread_scope_system>` (libcu++) with acquire/release is
the documented contract and costs nothing. Same for the `done` poll.

### H-6 (P2, S) Build flags

- `CMakeLists.txt:11,14`: `-march=native` on by default makes binaries non-redistributable;
  default OFF and let the box's `CLAUDE.local.md`/preset turn it on.
- `CMakeLists.txt:110`: `--use_fast_math` applies to the whole `flashrt_gemv` library (MMQ,
  `moe_q2`, `q3r`, `ggml_gemm`) while `blocks.cu` is compiled without it; document which kernels
  depend on it or scope it per file.
- `CMakeLists.txt:90-92`: `CMAKE_CUDA_ARCHITECTURES` silently defaults to 120 while the inline
  PTX (`mma.sync m16n8k32.s8`, `cp.async`, `ldmatrix`) needs sm_80+ and PDL needs sm_90+; add a
  configure-time check with a clear message.
- `-Xcudafe=--diag_suppress=177` hides unused-variable warnings; remove it and fix the warnings.
  Consider `-Werror` for CXX and `-Xcompiler=-Wall` for CUDA host code.
- `kernels/cuda/ggml_stubs.cpp` must be linked last and never beside libggml (eight "stubs
  last" comments); an object-library with an explicit link order, or building the vendored ggml
  as its own static library, removes the trap.

### H-7 (P2, M) `ForwardRef::forward()` control flow

`forward_ref.cu:442-569` multiplexes decode / window / chunk x graph / eager x PLE lookahead
through member flags (`in_chunk_`, `in_window_`, `combine_pending_`, `have_access_`) with graph
invalidation by pointer comparison (`same_buffers`, `blocks.hpp:62`). Split into
`forward_decode_graph`, `forward_eager`, `forward_chunk`, and make graph validity a version
counter on the scratch rather than address equality. Behaviour-neutral; KLD gate proves it.

### H-8 (P3, S) Kernel-launch error checks

120 `<<<...>>>` launches, 43 `cudaGetLastError` checks. Add a `launch_check()` after each
launch site in debug builds (a macro that compiles to nothing in Release), so a bad launch
configuration is caught at the launch and not at the next sync.

## 8. Performance (P) -- from the repo's own data

These are the open items the docs already rank; listed so the plan is in one place.

### P-1 (P1, M) The MTP head's pass over the prompt

Prefill with the head is 2,150-4,130 tok/s against 5,570-5,950 without it (`sw87`, `sw91`),
because `mtp_catchup` (`session.cpp:155-170`) runs the head in `prefill_batch`-sized slices on the
slow path. Give `MtpHead::forward` a chunk path like the target's (its experts are already in
VRAM, so it only needs the batched kernels), and overlap it with the target's next chunk.
Expected: prefill with the head back near the target's rate; TTFT at 250K from 44 s toward 45 s
instead of 117 s.

### P-2 (P1, M) Adaptive swap budget and a better cache prime

`sw88`: hit rate 66% on window 9's prompt vs 93% on wikitext; `sw89`: budget 32 recovers 9-12%
there but costs 1% on wikitext (`sw90`). Make the budget a function of the observed hit rate
over the last N tokens (large while low, 8 when high), and when priming the cache from a prompt,
weight the routing of the last few thousand tokens (the instruction tail) up rather than
halving counts every 4,096 tokens uniformly. Measure on both prompt kinds, teacher-forced.
(R-1 may shrink this gap first; re-measure after R-1.)

### P-3 (P2, M) Per-round draft length

Fixed `--spec K` per run; the head's probability did not gate well (`sw29`). With sampled
drafts, q is better calibrated (`sweet-spots.md:107`); try K chosen per round from the sum of
kept q mass and the round's expected new experts (the union of misses is what a longer window
costs, `sw28`).

### P-4 (P2, M) N-gram / prompt-lookup drafts stacked with the MTP head

Greedy at 245K keeps 3.5 tokens per round because the text repeats context (`sw33`);
`sweet-spots.md:108` ranks this second. Cheap to try: when a prompt-lookup match exists, extend
the MTP draft with it, verify in one window.

### P-5 (P2, S) Decode kernels outside the miss window

`sweet-spots.md:72-77`: Q3R at ~600 GB/s, LM head ~730, hc mix ~590 against ~900 GB/s
achievable; multi-CTA `k_idx_select` (70 us/layer at 245K), parallel hot-set CLOCK
(`k_hot_select` serial, 17 us/layer). Each a few percent; sum to maybe 5-8% at depth.

### P-6 (P3, owner) EXPO on

DDR5-3600 with EXPO off; the CPU misses run at DRAM bandwidth. Estimated ~10% decode at 245K
(`sweet-spots.md:118`). A BIOS change; re-run the P1 gate afterwards.

## 9. Reproducibility for third parties (X)

### X-1 (P2, M) Ship what a re-run needs

Not in the repo: `strata-ids.json` (the window 9 token ids; regenerable from
`bench/strata_depthbench.py:15-36` given a llama-server), `mtp-vocab/ranks.txt`, the saved
states, the KLD base `kl8k-f16.bin`, the exact llama.cpp commit + the uncommitted patches
(`w9-validation/README.md:10`: "branch mtp e4c893841 + uncommitted patches"). Commit the ids
and ranks (small), a script that rebuilds the KLD base, and the llama.cpp patch as a `.diff`
with its base commit.

### X-2 (P2, M) Drivers that run without systemd

The `bench/results/*/sw*.sh` scripts and the runbook use `systemd-run` units with placeholders.
Provide a plain-bash equivalent (`bench/run.sh <arm>`) and a Dockerfile (CUDA 12.8 devel, GCC
12, CMake 3.28, Ninja, Rust 1.82) so the build is reproducible; document the minimum box
(16 GB VRAM, 64 GB RAM, AVX-512 VNNI, NVMe for the PLE table).

### X-3 (P3, S) Versioning

No tags. Tag the state reviewed here (`v0.1.0` matches the engine's `ready` message) so results
folders can name a build.

## 10. Suggested order

1. **Week 1 -- integrity and safety:** R-1 (reuse fix + doc corrections), R-2, R-3, R-4, E-1,
   E-2, S-1, S-2, S-5, S-6. Then re-run sw91 with n >= 5 and replace the README table.
2. **Week 2 -- tests and CI:** T-1, T-2, T-3, T-4, T-5, T-7, H-2, G-1. This is the base for
   everything after, because it makes refactors provable.
3. **Week 3 -- structure:** H-1, H-3, H-6, H-7, H-4, H-5, E-3, E-4, S-3, S-4, S-7, S-8.
4. **Week 4 -- performance and generality:** P-1, P-2, P-3/P-4 (pick by the P-2 result), G-2,
   X-1, X-2.
5. **Later:** G-3, P-5, P-6, S-9, T-6, X-3.

## Appendix A. Numbers verified during the review

| Claim (README) | Raw source | Found |
|---|---|---|
| flashrt greedy `--spec 2`: 106.6 / 83.7 / 81.0 / 74.8 | `sw91.out` rows G1-G3 | 107.31/107.56/104.90; 82.80/84.41/83.84; 80.56/81.33/81.22; 72.89/75.75/75.89 -- means match, n=3 |
| flashrt t=1.0 `--spec 2`: 82.2 / 77.7 / 81.0 / 76.4 | same, S1-S3 | 32K cell is 76.43/65.39/91.41 (+-13.1) |
| flashrt no MTP: 94.7 / 81.7 / 77.4 / 73.0 | same, P1-P3 | match, n=3 |
| Strata greedy 87.0 / 96.0 / 85.0 / 80.4; llama.cpp 37.4 / 37.2 / 32.8 / 30.7 | `w9-all.out`, `w9-summary.txt` | match, n=4 (Strata t=1.0 250K is n=3: one run stopped at 1 token) |
| Prefix reuse: "one growing conversation" | `sw91.out` (`new == depth`), Strata log lines 36/38 | flashrt none; Strata 32,768 and 131,072 reused |
| Hit rate 66% vs 93% (sw88) | `w9_plain.txt:11`, `wiki_plain.txt:11` | 65.87% / 93.36%, n=1 each, budget 8 |
| 143 / 97 tok/s sampled drafts (sw85) | `spec1_32k_sampled.txt`, `spec1_245k_sampled.txt` | 143.1 / 97.1 over 6 windows, one run |
| KLD 0.0082-0.0092; llama.cpp vs itself 0.0078-0.0086 | `p1-kld/kl8k-*.log`, `sw62/*.log` | flashrt 0.008947, 0.008545, 0.008983; ub512 0.008589, q8 0.007764 -- comparable protocol (same 8,190 scored tokens, same FP16 base) |
| Prefill 5,570-5,950 | `sw91.out`, `sw91-summary.txt` | engine `prompt_ms`: 5,690-5,960; TTFT-delta: 5,565; two measures mixed |
| Strata prefill 1,170-2,030 | Strata log lines 34-38 | per new token: 1,173 / 1,089 / 969; 2,031 counts reused tokens |

## Appendix B. Size of the tree at `27856aa`

Own code 20,308 lines (C++/CUDA 16.4k, Rust 1.9k, tests 2.0k); vendored ggml 23,460 lines
(unmodified, MIT); docs 1,853 lines; 97 result folders with 717 files and 85 READMEs; 279
commits between 2026-09-27 13:14 and 2026-09-29 08:44 (+04:00), all with `Claude-Session`
trailers.
