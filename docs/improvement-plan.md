# Improvement plan (from the external reviews of 2026-09-29)

Three reviews of commit `27856aa` are in `docs/feedback/`:
- `fable-...md`: a line-level review with a backlog. Item IDs below (R-1, E-1, ...) are its IDs.
- `chatgpt-...txt` and `grok-...txt`: assessments of scope, generality, benchmarks and direction.

This plan merges them, orders the work, and records what is not adopted and why. Each phase
ends with something measurable. The repo's rules still apply: KLD gate, `bench/` evidence,
teacher-forced A/B runs, and the server smoke test after engine or VRAM changes.

## What the reviews agree on

| Finding | Fable | ChatGPT | Grok |
|---|---|---|---|
| The engine solves its one target well; 2.2-2.6x llama.cpp holds; correctness discipline is strong | yes | yes | yes |
| The generic runtime in `design.md` and `interfaces.md` is a sketch, not code | G-1 | §5 | yes |
| CI compiles no CUDA | T-1 | §4 | (GPU CI suggested) |
| `blocks.cu` (3,800 lines) should be split; the `FLASHRT_*` toggles are debt | H-1, H-3 | | yes |
| The MTP head's prompt pass is the clearest open performance item | P-1 | §10.5 | yes |
| The cache warm-up is where decode speed at depth goes | P-2 | §10.4 | yes |
| Report hit rate and acceptance beside tok/s; add workloads beyond wikitext and window 9 | R-4 | §10.6 | §5 |
| A second model should define the API, not the other way round | G-3 | §10.1 | §2 |
| Pin exact baseline versions; add a one-command reproduction and a tag | X-1..X-3 | | §5, §6 |

Only Fable reports the findings that change published claims (R-1, R-2, R-3) and the
robustness bugs (E-*, S-*). I checked the main ones against the tree before building this plan:

- **R-1 holds.** In sw91 every depth prefills all of its tokens (`new == depth`). Strata reused
  32,768 tokens at 134K and 131,072 at 250K (`W9-G-strata-greedy-r1-*.log`, lines 36 and 38).
  `engine/session.cpp` keeps one checkpoint, at the end of the prompt. Window 9's prompts all end
  with the same instruction, so the shared prefix is shorter than that checkpoint, and the
  engine resets. The sw87 README's "None of the engines reused a prefix" is wrong for Strata.
- **R-2 holds.** Strata's log warns that its GPU hit path "is NOT CORRECT", and that its tokens
  diverge from a cache-off run. Its timing is real; its acceptance counts come from a wrong
  target.
- **E-1 holds.** `cache_release()` runs before the prefill loop, but `cache_restore()` runs only
  on the success and cancel paths. An exception in between leaves the session inconsistent.
- **S-2 holds.** `chat.rs` advances `"</parameter>".len()` (12 bytes) after a value ended by
  `</function>` (11 bytes). A multibyte character right after it panics the task.
- **S-5 holds.** `top_k` is clamped to 1..64, so the usual "0 = off" turns into greedy decoding.

## Phase 0: correct the published claims (hours, docs only)

**Status: done 2026-09-29.** The README, the docs and the affected result READMEs carry the
corrections; the numbers themselves are unchanged until phase 2 re-measures them.

1. **R-2:** add the Strata caveat everywhere its numbers appear (`README.md`, `docs/sweet-spots.md`,
   `docs/engine.md`, `bench/README.md`, the w9 and sw87-sw91 READMEs): "timing as measured; the
   build warns that its cache path changes outputs".
2. **R-1 (docs part):** fix the sw87 README, and state in sw91 and in the `README.md` results
   section that flashrt re-prefilled every depth while Strata and llama.cpp reused prefixes.
   Say that the warm-up explanation (sw88) is not yet separated from this difference.
3. **R-3:** quote prefill per new token for every engine (Strata: 1,173 / 1,089 / 969). Use one
   flashrt measure (engine `prompt_ms`) and name it.
4. **R-4:** give mean ± sd and n in the README table. Say that the 1K sampled lead and the
   "level at 32K" are within noise.
5. **R-5, R-6:** say next to the table that the swap budget was chosen on window 9's protocol.
   Put the natural-text numbers beside Strata's p0c wikitext numbers with the protocol
   differences, or move them to the docs. Fix "hits 90-95%": that holds at 32K, and 245K
   measures 83-89%.
6. **G-1:** `design.md` (Layers) says where each piece lives today. `interfaces.md` gets
   "status: design sketch, not implemented" and Fable's table of hard-wired shapes. Remove
   `flashrt autotune` from `design.md` or mark it as not built.
7. **H-2 (done 2026-09-30):** fix the stale header comments (`forward_ref.hpp`, `blocks.hpp`) and drop the dates
   from the doc titles.

**Done when:** every number in `README.md` names its protocol, n and spread, and no doc
describes code that does not exist.

## Phase 1: a failed request must not poison the next one (1-2 days)

**Status: done 2026-09-29** (`bench/results/2026-09-29-sw92-faults`, `2026-09-29-sw93-server`). The
test for E-1 compares the state after the prompt by its logits, not by tokens: greedy tokens
depend on the expert cache's content, so they differ from a fresh engine's even when nothing
failed (sw92). E-3 (a bounded host-side doorbell wait) was done with it.
Found on the way: the fast-path KLD (window 3, hot set 512) moved from 0.008931 (sw78) to 0.009124
before phase 1, still inside the gate band. **Bisected (sw94):** the step is a97bac1, the
warp-per-token prefill routing, whose softmax sums in another order. A last-bit change in the
routing probabilities changes the expert cache's first fill, and so which tokens hit on the GPU
and which miss to the CPU. `FLASHRT_ROUTE_WARP=0` on the current build gives back 0.008931 and
12,021 swaps exactly. Not a bug. The fast-path KLD moves by about 0.0002 under such perturbations
while the chunk path does not, so phase 4's refactors, which should be bit-identical, are checked
by exact equality of both.

| Item | Change | Test |
|---|---|---|
| E-1 | RAII guard in `Session::generate`: on any exception, close an open window, restore the expert cache, sync, reset the sequence state, then rethrow | `engine_smoke.py`: over-long prompt / stop during prefill / bad token id, each followed by a normal request that must match a fresh engine (greedy) |
| E-2 | Validate `max_new`, `top_k`, temperature and seed in `to_request` before any GPU work | same, invalid values give an error line and no state change |
| S-1 | Server tracks engine liveness; requests get 503; `/health` reports it; `quit` on shutdown | fake-engine script that exits after `ready`: 503 within 1 s |
| S-2 | Advance by the terminator actually found; fuzz `parse_tool_call` over random UTF-8 around the tags | unit test + fuzz loop |
| S-4 | Send `stop` when the client disconnects while queued (`tx.closed()` in a `select!`) | unit test with the fake engine |
| S-5 | `top_k` 0 or absent means 64; > 64 returns 400 | unit test |
| S-6 | Skip control and special tokens in output text, except those the section parser consumes | unit test on a token sequence |

**Done when:** those tests pass, `server_smoke.py` passes on the box, and the KLD result is
unchanged (the fixes do not change outputs).

## Phase 2: a fair re-measurement (2-3 days, then a benchmark window)

**Status: done 2026-09-29.** Host checkpoints (sw95; restores bit-exact on the cold run's chunk
grid). Window 9 re-measured at n = 5 (sw96): reuse matches the reference engines', and decode
did not move. So P-2 was decided from the result: sw99 simulated the cache (the loss is the
warm-up, the optimum ~90% hits), and sw100-sw101 made faster admission the default (+3.6-6.6% at
1K-134K with the head). The agentic workload is `bench/agent_trace.py` (sw97). The README table
is replaced. P-1's first step (the head's KV mirror, sw98) and the head's 257 s load (a pinned
thread; now 5 s) came with it.

1. **R-1, the engine part:** checkpoints during prefill. Save the recurrent state (GDN state and
   conv, PLE history, indexer ring; the KV needs nothing) every 4,096 tokens (or every chunk) into
   a small ring in host RAM, for the target and the MTP head. Restore the latest checkpoint at or
   before the common prefix. On reuse, keep the warm expert cache, and blend in the new routing
   counts rather than replacing them.
   - For chat, a history that grows at the end already reuses. This fix matters for prompts with
     a fixed tail (instructions after the context), edited system prompts and agent frameworks
     that rewrite the middle, and for comparable benchmarks.
   - Test: `engine_smoke.py` case A, then A[:-k] + tail, must report `reused >= len(A) - k - 4096`;
     KLD unchanged.
2. **Re-run window 9's protocol:** n >= 5 per arm, arms interleaved in one session, GPU clock and
   temperature logged. Report the cache hit rate and draft acceptance beside tok/s for every row
   (the engine's `done` event already has the acceptance).
3. **Decide P-2 from the result.** If the gap to Strata closes with reuse, the swap budget can go
   back to being tuned on natural text. If not, build the adaptive budget (large while the hit
   rate is low, 8 when it is high) and weight the prompt's tail up in the prime. Measure both prompt
   kinds, teacher-forced.
4. **A small agentic workload** (ChatGPT §10.6, Grok §5): a scripted multi-turn trace with code
   context, tool calls and tool results. Report TTFT, tok/s, hit rate and misses per token. It
   shows how the cache behaves when routing diverges from the prompt, which is the regime that
   matters for coding use.

**Done when:** the README table is replaced by the new numbers.

## Phase 3: tests and CI that can prove refactors (1 week)

**Status (2026-09-29):** T-1 done: the `cuda` CI job builds the whole tree for sm_120 in
`nvidia/cuda:12.9.1-devel-ubuntu24.04` (about 6 minutes) and runs `ctest -L cpu`; tests carry the
label `cpu` or `gpu`. T-3 done except the state-file round trip: `test_json`, `test_gguf` and
`test_row_reader`. Writing them found bugs, now fixed. `json`: `nan`, `inf`, `+1` and hex
parsed as numbers; a surrogate that was not half of a pair decoded to garbage; dumps lost digits.
`gguf`: alignment 0 divided by zero; an offset past the end underflowed; a huge array count ran
into `bad_alloc`; shard 2 used shard 1's alignment. From S-7: the tokenizer accepts only the
qwen35 pre-tokenizer and fails without a valid eos id.

- **T-1:** compile the CUDA tree in CI, in the `nvidia/cuda:12.9` devel container or with the
  toolkit from NVIDIA's apt repo, for sm_120, and run the CPU tests from that build.
- **T-2:** ASan+UBSan on the CPU tests; TSan on `test_cpu_pool` and `test_moe_cpu`.
  `test_q2_0` returns 77 when AVX-512 VNNI is absent.
- **T-3:** pure-CPU tests for `core/json.cpp`, `core/gguf.cpp` (a synthetic 2-shard file),
  `core/row_reader.cpp`, and the state-file round trip.
- **T-4:** the decode GDN kernel (`k_gdn_delta_reg`) against a CPU double reference for T = 1..8,
  and `gdn_rewind` / `ple_rewind` bit-exact against a fresh run; `test_spec_sample` with 3-4
  rows and the vocabulary remap; the `linear_multi` epilogues; Q2_0 in `test_moe_q`.
- **T-5:** the cache policy driven by a synthetic trace and compared with `tools/cache_sim.py`; the
  doorbell timeout path with a stub miss server.
- **T-7 and S-7:** a fake-engine protocol test, the real chat template against a checked-in
  expected render, SSE framing, and a tokenizer fixture (CJK, emoji, code, special tokens in
  user text); assert `tokenizer.ggml.pre`; fail on a missing `eos`.
- **T-6:** `LABELS` and `TIMEOUT` on the tests; timing loops move to `bench_*` targets.

**Done when:** CI builds CUDA, the sanitizer jobs are green, and every kernel on the decode path
has a reference test.

## Phase 4: structure, behaviour-neutral (1 week)

Each step is proven by `fr_parity`, the KLD gate and a teacher-forced A/B showing no change.

1. **H-3 (done: sw112, 19 toggles removed, outputs identical):** delete the losing kernels behind the `FLASHRT_*` toggles. The result folders stay
   as the record. Keep only `FLASHRT_MOE_AB64`, `FLASHRT_HC_Q8` and `FLASHRT_ARGMAX_DRAFTS`,
   plus any toggle `sweet-spots.md` names as still useful. Do this before H-1, so the split
   moves less code.
2. **H-1 (done: sw112, outputs identical):** split `blocks.cu` into `hc.cu`, `gdn.cu`, `qsa.cu`, `moe_ref.cu`, `ple.cu` and a
   common header.
3. **H-2 and H-7 (done: sw112, outputs identical):** rename `ForwardRef` (it is the whole forward pass). Split `forward()` into
   decode-graph, eager and chunk paths, and invalidate graphs by a version counter, not by
   comparing addresses.
4. **H-4, H-5, H-6:** name the magic numbers in one header; use `cuda::atomic_ref` for the
   doorbells; make `-march=native` opt-in; scope `--use_fast_math`; check the CUDA
   architecture at configure time.
5. **E-3, E-4, S-3, S-8** (S-3 and S-8 done 2026-09-29: tokenization runs on the blocking pool with
   the cache locked per word, BPE merges from a heap, byte-identical to llama.cpp on the wikitext
   reference; content arrays join their text; bad tool arguments are a 400; template kwargs
   cannot replace the conversation; S-9's body limit is 64 MiB): bound the host-side doorbell wait; shut the engine down cleanly;
   move tokenization to `spawn_blocking` with a per-word lock and a cap; fix the request-shape
   issues.

## Phase 5: performance (after phases 1-3)

In expected-value order. Each needs its KLD check and a `bench/results` folder.

1. **P-1: a chunk path for the MTP head's prompt pass** (all three reviews). **Done 2026-09-30:**
   the head's KV mirror (sw98) and its catch-up in calls of 1,024 rows with grouped expert GEMMs
   (sw102). 131K with the head 35.3 → 23.6 s; the head costs 5-7% of prefill at depth; 250K
   time to the first token on window 9 24.0 s (a whole cold 250K prompt: about 50 s, an estimate). Prefill with the
   head runs at 2,150-4,130 tok/s against 5,690-5,960 without it. The head's experts are
   already in VRAM, so it needs the batched kernels, overlapped with the target's next chunk.
   Expected: 250K TTFT from about 117 s toward 45 s (estimate).
2. **P-2** as decided in phase 2. **Done 2026-09-30** (sw99-sw109): the simulator found the loss in
   the warm-up; faster admission, seeded counts at 0.03x, 64 uploads per step, deterministic
   commits at the next step. Window 9 with the head +8-25% (ahead of Strata at every depth); the
   agent session +12%. Left: cheaper uploads, or a prime that predicts the answer (the tail blend
   helped window 9 in simulation, and the engine does not keep the prompt's tail routing).
3. **P-3, P-4:** choose the draft length per round from q's kept mass and the round's expected
   new experts; add prompt-lookup drafts stacked on the MTP head. Try both; keep whichever
   the P-2 result favours.
4. **P-5:** multi-CTA `k_idx_select`, a parallel hot-set CLOCK, Q3R and LM-head bandwidth. A few
   percent each at depth.
5. **Quality beyond KLD** (Grok): a few hundred items of a reasoning eval (for example GSM8K)
   through flashrt and llama.cpp on the same GGUF, with a matching score expected. KLD is
   necessary but not sufficient.

## Phase 6: reproducible by someone else

- **X-1:** commit the window 9 token ids and the drafter vocabulary ranking, a script that
  rebuilds the KLD base, and the llama.cpp patch as a `.diff` with its base commit. Pin the
  Strata commit.
- **X-2:** a plain-bash `bench/run.sh` alongside the systemd scripts; a Dockerfile for the build;
  document the minimum box (16 GB VRAM, 64 GB RAM, AVX-512 VNNI, an NVMe drive for the PLE table).
- **Server metrics** (Grok): expose the hit rate, miss time, acceptance and cache slots on an
  endpoint; the engine already tracks most of them.
- **A "will it run on my machine" section** (Grok): what is sm_120-only, what needs AVX-512, how
  the VRAM budget scales.
- **X-3:** tag `v0.1.0` after phase 2, so result folders can name a build.

## Later, when there is a reason

- **G-2:** move the offload machinery (expert cache, policy, miss server, doorbells) into
  `core/`, parameterised by an expert-shape struct. This is the part every offloaded MoE shares.
- **G-3:** a second model as the forcing function for the arch API. The reviews suggest, in
  order: a different quant of Flash-Next (tests the quant seam), a Qwen3-Next-style GDN MoE (tests
  mixer reuse), then a plain GQA MoE (tests that the cache core is generic).
- **Several users:** one engine per sequence (Grok), not continuous batching in this design.
- **Hardware breadth:** a 32 GB card and other RAM speeds. This needs access to another box.
- **P-6, EXPO:** a BIOS change, so it is the owner's call. The estimate is about 10% at 245K.

## Not adopted

- **A learned expert predictor for prefetch** (ChatGPT §10.4). sw63 measured prefetching
  forecast experts: prefetch catches 66-68% of misses, but its uploads read host DRAM, which is
  the misses' own bottleneck, about twice per miss removed. The cache-admission side (P-2) is the
  usable form of this idea.
- **Separate `ForwardReference` and `ForwardOptimized` classes** (ChatGPT §10.3). The reference
  path exists, and parity plus KLD already validate against llama.cpp. The rename and split in H-7
  cover readability without a second forward pass to maintain.
- **Continuous batching, a general graph IR, the vision projector:** out of scope for a batch-1
  runtime aimed at one model. All three reviews agree on the first two.
- **A hardware matrix across GPUs and CPUs:** there is one box. Documenting the assumptions
  (phase 6) is the part that can be done now.
