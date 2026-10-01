# sw94: what moved the fast-path KLD after sw78 (2026-09-29)

sw92 measured the fast path (window 3, hot set 512, chunks of 2048) at KLD 0.009124, against
sw78's 0.008931, still inside the gate band. `sw94.sh` builds `fr_kld` at each code commit from
sw78 to phase 0, in a separate clone, and runs the same command: 2 × 8K wikitext against the
FP16-KV llama.cpp base. The runs are deterministic: every value repeats to the last digit, and
the swap count with it.

| Commit | Change | KLD | Swaps |
|---|---|---|---|
| 3842cfa | sw78's build | 0.008931 | 12,021 |
| 7506ff1 | moe hits v2 (decode) | 0.008931 | 12,021 |
| 1139850 | its revert | 0.008931 | 12,021 |
| 1ca257b | prefill routing ranks among candidates | 0.008931 | 12,021 |
| **2b074a9** | **prefill routing, a warp per token (sw81)** | **0.009124** | **11,974** |
| 9a5aef6 | GDN prefill: q/k norm in the conv | 0.009124 | 11,974 |
| 867982a | shared BF16 conversion | 0.009124 | 11,974 |
| 817bcde | moe_q2 packed scales | 0.009124 | 11,974 |
| e8b90c1 | sampled drafts | 0.009124 | 11,974 |
| 0aca9c2 | GDN state-block variants dropped | 0.009124 | 11,974 |
| 8423ab7 | checkpoints allocated up front, reserve 512 MiB | 0.009124 | 11,974 |
| dd3e678 | swap budget 32 in the engine and fr_bench (fr_kld keeps the policy default, 8) | 0.009124 | 11,974 |

(sw92 then measured 0.009124 again at 2ef1d18, the phase 0 commit, and after phase 1.)

**The step is 2b074a9.** Its kernel (`k_route_topk_w`) computes the router softmax with the sum in
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
