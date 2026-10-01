# sw118: the expert cache leaked slots under speculation; fixed (2026-09-30)

**Symptom (sw116, sw117).** Over many short independent requests (GSM8K through flashrt-server,
MTP head, `--spec 2`), decode slowed from about 130 to about 55 tok/s and stayed there. Per request
(sw117, `bench/gsm8k_eval.py` now keeps the server's `timings`): the expert cache's hit rate fell
from 83% to 12% over 200 requests, while draft acceptance held at 80-87%.

**Diagnosis.** The new engine flag `--cache-check` logs after each request the host and device
tables, slot owners and uploads in flight (`gsm8k/check-before.log`). The cache lost usable slots:
7,179 of 7,800 experts resident after the first request, 2,961 after 60, and the difference sat in
slots whose owner no longer mapped to them. Host and device tables agreed, so no slot was ever freed
or reused.

**Cause.** `cache_manager_step` sorts the step's missed experts by count and removes duplicates
with `std::unique`, which only removes adjacent ones. In a verify window the same expert can miss
for several tokens, and when other experts have the same count, `std::sort` can leave the copies
apart. The expert was then admitted twice in one step, uploaded into two slots, and both commits
applied: the table kept the second slot and the first stayed owned by it, never free and never
evicted. One-token decode never repeats a key within a step, so only speculative decoding leaked.
tools/cache_sim.py de-duplicates correctly, so the simulator never had this.

**Fix** (749cb02): ties broken by key, so copies are adjacent and `unique` keeps one.

## Checks

| Check | Result |
|---|---|
| `--cache-check` over 100 GSM8K requests (`gsm8k/check-fixed.log`) | 7,736 resident + 64 in flight throughout, 0 table or owner mismatches; hits 82-86% per request |
| KLD fast path / verify windows of 3 (hot set 512) | 0.008960 / 0.008824 (before: 0.008996 / 0.008913) |
| ctest | 19 / 19 |
| `engine_smoke --reuse` / `--faults` with the head; `server_smoke.py` | 5 / 5, 18 PASS; 15 / 15 |

**Speed across requests** (GSM8K through the server, the first 200 items, decode tok/s and expert
cache hits per 25 requests):

| requests | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 |
|---|---|---|---|---|---|---|---|---|
| before (sw117) | 118 (73%) | 91 (57%) | 79 (43%) | 69 (31%) | 65 (21%) | 63 (17%) | 62 (13%) | 61 (12%) |
| **fixed** | **147 (84%)** | **148 (83%)** | **148 (83%)** | **144 (83%)** | **143 (82%)** | **145 (82%)** | **146 (83%)** | **146 (83%)** |

**Single runs** (teacher-forced, old and new interleaved, 3 each): within noise, because one
320-token run leaks little. Window 9 32K plain 86.95 → 86.76 tok/s (hits 82.09 → 82.04%: the
tie-break reorders admissions), with the head 109.94 → 108.58 (hits 72.19 → 72.91%), wikitext 8K
107.56 → 108.34.

**GSM8K scores** (first 200 items, `gsm8k/`): fixed 193, the old build 197 in sw117 and 194 in
sw116's run, llama.cpp 195. The four items where fixed and sw117 differ (12, 45, 102, 161) were also
wrong in sw116's run of the old build: near-ties that flip with the cache's history (GPU-hit and
CPU-miss arithmetic differ in the last bits). Fixed against llama.cpp: 0 against 2 right in one
only, p = 0.50.

**What it affected.** Every run with the MTP head leaked, a little per window: long sessions
without a cache refill (the server between requests shorter than 4,096 new tokens, agent sessions)
the most. Runs that refill at each prompt (window 9's depths, 4,096+ new tokens) and plain decode
little or not at all. sw119 re-measures window 9 and the agent session.

## Files

- `sw118.sh`, `sw118.out`; `logs/` (KLD, A/B runs); `smoke/`.
- `gsm8k/`: the sw117 run before the fix, the fixed run, llama.cpp's first 200 (from sw116), the
  cache-check logs before and after.
