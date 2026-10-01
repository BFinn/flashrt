# sw107: deterministic upload commits, and the budget per step (2026-09-30)

sw106 found that sw104's settings made runs differ. The cache committed an upload when a
`cudaEventQuery` found it done, and with 64 in flight that depended on timing.

**The change** (`arch/qwen4exp/moe_fast.cu`):
- An upload commits two steps after the step that issued it (`cudaEventSynchronize`, which almost
  never waits: the upload has had a whole token's time).
- The swap budget counts the uploads started per step. Counted in flight, it halved the upload
  rate once uploads stay pending two steps: in `sw107.sh`, window 9 with the head fell to 96.9 tok/s
  at 3,800 swaps. `sw107b.sh` has the per-step budget.

**Determinism is back** (`sw107b.out`, each twice):

| KLD configuration | Run a | Run b | Swaps |
|---|---:|---:|---:|
| `--fast` | 0.008986 | 0.008986 | 159,442 both |
| `--fast --window 3 --prefill-chunk 2048 --kv-hot 512` | 0.008659 | 0.008659 | 100,269 both |

Both are inside the gate band (fast path 0.0087-0.0092 as recorded; the plain `--fast` value was
0.008602 with the old settings). `fr_bench`'s hit rates and swap counts repeat exactly as well.

**Speed** (as sw104, teacher-forced, 3 runs; `sw107b.out`): window 9 plain 85.5 / 82.9 / 84.8 tok/s
(81.4% hits), wikitext 107.2 / 106.8 / 106.1 (93.4%), window 9 with the head 103.3 / 102.3 / 101.1
(71.8%). Against sw104 (another session) that looked like a cost with the head (108.5). **Measured
in one session it is none** (`sw107c.out`: the timing-dependent build, 9af6062, built in a separate
clone, alternating with this one):

| Build | Window 9 with the head, 3 runs | Mean |
|---|---|---:|
| timing-dependent commit | 95.1, 104.6, 108.5 | 102.7 ± 6.9 |
| deterministic commit | 104.8, 106.8, 103.5 | 105.0 ± 1.7 |
