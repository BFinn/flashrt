# sw94: what moved the fast-path KLD after sw78 (2026-09-29)

sw92 measured the fast path (window 3, hot set 512, chunks of 2048) at KLD 0.009124, against
sw78's 0.008931, still inside the gate band. `sw94.sh` builds `fr_kld` at each code commit from
sw78 to phase 0, in a separate clone, and runs the same command: 2 × 8K wikitext against the
FP16-KV llama.cpp base. The runs are deterministic: every value repeats to the last digit, and
the swap count with it.

| Commit | Change | KLD | Swaps |
|---|---|---|---|
| b165391 | sw78's build | 0.008931 | 12,021 |
| a5c31ff | moe hits v2 (decode) | 0.008931 | 12,021 |
| 99303f7 | its revert | 0.008931 | 12,021 |
| b4373fa | prefill routing ranks among candidates | 0.008931 | 12,021 |
| **a97bac1** | **prefill routing, a warp per token (sw81)** | **0.009124** | **11,974** |
| 795d2c5 | GDN prefill: q/k norm in the conv | 0.009124 | 11,974 |
| e8d5f96 | shared BF16 conversion | 0.009124 | 11,974 |
| d45c3fd | moe_q2 packed scales | 0.009124 | 11,974 |
| d2da2b0 | sampled drafts | 0.009124 | 11,974 |
| 55c143f | GDN state-block variants dropped | 0.009124 | 11,974 |
| 19696d7 | checkpoints allocated up front, reserve 512 MiB | 0.009124 | 11,974 |
| 0d38d92 | swap budget 32 in the engine and fr_bench (fr_kld keeps the policy default, 8) | 0.009124 | 11,974 |

(sw92 then measured 0.009124 again at 8e6d6b5, the phase 0 commit, and after phase 1.)

**The step is a97bac1.** Its kernel (`k_route_topk_w`) computes the router softmax with the sum in
another order than the block kernel it replaced (its comment says so: "p may differ in the last
bit"). The chosen experts follow the same rule (higher p first, ties to the lower index), so
routing differs only where two candidates were within that last bit. It is not wrong, but the
KLD moved.

**Why a last-bit change moves the fast-path KLD by 2%.** The prefill's routing counts fill the
expert cache, and the cache's content decides which experts the scored tokens find on the GPU and
which miss to the CPU. The two paths are not bit-identical (`docs/engine.md`, Correctness
machinery). The swap count changes with the fill (12,021 against 11,974). The chunk-path KLD,
which never uses the cache, stayed at 0.008879 over the same commits (sw81-sw83, sw92).

**Confirmed on the current build** (`sw94b.sh`): with `FLASHRT_ROUTE_WARP=0` (the block kernel),
the fast path gives back 0.008931 and 12,021 swaps exactly, so nothing else moved it. Plain
`--fast` (no chunked prefill) measures 0.008688 and 26,147 swaps with either kernel. That
configuration's prefill does not take the chunk path, where this routing kernel runs.

**What it means for the gate.** The fast-path KLD is a deterministic function of the build, but a
change that only reorders a sum in the prefill moves it by about 0.0002. A fast-path shift of
that size is inside this measured perturbation, not by itself a regression. A change that
touches outputs should be judged on the chunk-path KLD as well, which does not have this
sensitivity, and on the fast path over more than one configuration.
