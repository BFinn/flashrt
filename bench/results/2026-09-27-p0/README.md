# P0 measurements, windows A and B (2026-09-27, 13:07-13:18 and 13:29-13:50)

The target box: RTX 5080 16 GB (PCIe Gen5), Ryzen 9 7900X, DDR5 at 3600 MT/s. The live
server was stopped for both windows. Scripts: `bench/p0/window_a.sh` and `window_b.sh`.
Logs: `p0a.out` and `p0b.out`. The routing traces (`.npy`, about 1 GB) are not in git.
They are in `$BENCH/p0{a,b}-20260927/` on the box.

## 1. Host memory and the link [M]

| Probe | Result |
|---|---|
| `membw`, AVX-512 reads, 4 KiB pages, 1 / 6 / 12 threads | 42.6 / 48.6 / 47.0 GB/s |
| `membw`, THP (60-64% huge) | 48.9 / 47.4 / 45.9 GB/s: no gain |
| `h2dbw` pinned (`cudaHostAlloc`), 1.38 MB (one expert) / 16 MB / 64 MB chunks | 45.4 / 48.6 / 48.7 GB/s |
| `h2dbw` registered, 4 KiB pages | 42.0 / 47.4 / 47.6 GB/s |
| `h2dbw` registered, THP (100% huge) | 49.6 / 53.8 / 54.5 GB/s |
| `h2dbw` zero-copy kernel reads, THP | 45.1 / 48.8 / 47.4 GB/s |
| Link with 6 CPU reader threads running | link 25-26, CPU 24-28, **sum 48-52 GB/s** |
| Link with 12 CPU reader threads running | link 25-26, CPU 24-26, sum 49-50 GB/s |

**What this changes:**
- **Host DRAM delivers about 50 GB/s of reads,** not the 33.6 GB/s STREAM triad (triad
  includes writes).
- **The Gen5 link is not a 20 GB/s cap.** It moves 45-55 GB/s, even in 1.38 MB expert-sized
  chunks. The 20 GB/s seen during prefill came from the engines' copy patterns.
- **CPU misses and PCIe misses share one DRAM budget of about 50 GB/s,** split roughly
  evenly when both run.
  - Expert misses cost DRAM bandwidth whichever path serves them.
  - The CPU/PCIe split decides where compute and latency land, not how many bytes move.
  - This fits Strata's measured `--pcie-frac` optimum of 0.35.
- **Huge pages help DMA** (registered copies +10-15%), not CPU reads.

## 2. The depth prompts are filler

`strata-ids.json`, used by every depth benchmark so far, is **134 distinct tokens
repeated**: 0.005 unique trigrams per token at 32K. Window B re-measured on wikitext-2 (raw
test + valid, 12,084 distinct tokens and 0.70 unique trigrams per token in the first 140K).

- **Prefill numbers on the filler are optimistic.**
  - Strata at 32K: 1,134 tok/s on the filler (window 9) against 683 on wikitext.
  - llama.cpp with routing traced: 690 against 473.
- **Decode is less affected.** Strata greedy: 96.0 on the filler (mean of 4) against 87.0 on
  wikitext (1 run) at 32K. At 131K it was 85.0 against 100.8.
- **All future depth benchmarks should use natural text.** `p0b-20260927/wiki-ids.json`
  on the box has depths of 1K / 32K / 131K.

## 3. Routing (route_trace, llama.cpp dev tree, sampled at t=1.0 / top_p 0.95 / top_k 20, 2,048 decoded tokens)

From `trace_stats.txt`, wikitext at 32K and 131K:

| Statistic | 32K | 131K |
|---|---:|---:|
| Share of routes to the hottest 10% of experts (median layer) | 69.1% | 55.5% |
| Experts shared with the previous token | 42.3% | 41.4% |
| Distinct experts in a 3-token window, as a share of 3k | 0.70 | 0.70 |
| Distinct experts in a 4-token window, as a share of 4k | 0.64 | 0.64 |
| Experts touched per layer by a 2,048-token prefill chunk | 72.7% | 73.4% |
| Same, 8,192-token chunk | 88.3% | 87.6% |

- **Verify windows are sub-linear in expert reads,** against the paper-based reading in the
  background notes. A 3-token window reads 2.1× the experts of one token, not 3×.
- **Prefill chunks do not touch every expert.** The (1 − 10/512)^2048 argument assumes
  uniform routing. An engine that streams only touched experts moves about 3.3× fewer
  expert bytes per prompt token at 8K chunks than at 2K.

## 4. Expert cache hit rate vs slots (`cache_sim.txt`)

Wikitext at 32K, global slots, cold start:

| Slots | LRU | windowed admit (3 in 32) | decayed LFU | oracle static profile | Belady |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 79.5% | 78.8% | 78.7% | 80.8% | 90.6% |
| 5,500 | 86.5% | 85.7% | 85.5% | 88.0% | 93.8% |
| 7,000 | 90.7% | 90.3% | 90.2% | 92.6% | 95.8% |
| 9,000 | 94.4% | 94.4% | 94.3% | 96.3% | 97.2% |

- **Belady has half the misses of LRU** at 5,500-7,000 slots, so a better policy or
  prefetch has real headroom.
- **Among online policies, LRU is as good as anything here.** Decayed-LFU with Strata's
  constants and windowed admission do not beat it on this trace.
- **A fixed profile close to the workload beats all adaptive policies.** The profile here is
  an oracle computed on the same trace.
- **Strata runs 7,002 resident slots** (serve log, KV streaming on).

## 5. QSA selection locality (`qsa_stats.txt`)

Per-layer LRU hot set of KV blocks (4 cells each; the selection is 513 blocks), counting
only blocks that existed before decoding:

| Depth | Step-to-step overlap | Miss rate, 1× hot set | 2× | 4× (about 8K cells) |
|---|---:|---:|---:|---:|
| 32K | 68.0% | 35.5% | 14.8% | 5.6% |
| 131K | 62.4% | 44.9% | 24.8% | 14.7% |

- Layers 3 and 7 have the worst locality: 31% misses at 4× on 131K.
- At 4× and 131K, the misses are about 75 blocks × 4,352 B per layer, which is **about
  4 MB per token**. That is small next to about 89 MB of expert misses per token at 86%
  hits.
- **KV streaming is cheap in bytes.** It frees about 3 GiB at 262K, about 2,300 expert slots.
- Strata's serve log agrees: 98.0% of KV block reads hit VRAM at 32K.

## 6. Prefill vs chunk size (single runs)

| Engine, 32K | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|
| llama.cpp `llama-bench` pp32768 (random tokens, mmap load, so the absolute numbers are low) | 475 | 574 | 701 |
| Strata `--prefill`, wikitext | 683 | 762 | 819 |

- The pinned-memory llama-bench rerun failed: this build uses `-lm dio`, not `-mmp 0`.
- Bigger chunks help both engines, but less than the 3.3× reduction in expert bytes would
  suggest. That points to something other than expert transfer bounding prefill at 4K-8K.

## Not done yet

- **Strata's per-stage breakdown.** `--stats` prints nothing in serve mode. The next window
  runs `strata generate ... --stats` and `--gpu-only-full` (the per-token GPU floor).
- **llama.cpp prefill sweep with `-lm dio`.**
- **Reference decode numbers on wikitext:** repeated runs of the llama.cpp dev tree and
  Strata, to replace window 9 as the gate baseline.
