# P0 window C (2026-09-27): Strata breakdown, PLE I/O, natural-text baselines

Target box: RTX 5080 16 GB, Ryzen 9 7900X, DDR5-3600, Samsung 990 PRO 2 TB. All runs use
wikitext-2 (raw test + valid) as the prompt, not the filler depth prompts. Script:
`bench/p0/window_c.sh`.

- First attempt, `p0c.out` (14:48-14:52): the `--gpu-only-full` arm hit its 56 GB cgroup
  cap. The killed process held VRAM past the 120 s wait, so the window ended early and the
  live server was restarted cleanly.
- Rerun, `p0c2.out` (15:02-15:52): `SKIP_DONE=1`, with a 10-minute VRAM wait.

## 1. Strata's own breakdown, 32K prompt, generate mode, tuned flags, greedy with MTP (1 run each)

From `gen-stats.first.log` (`--stats`):

| Item | Value |
|---|---|
| Decode | 84.8 tok/s, 11.8 ms per token |
| Speculation | 1.66 tokens per round, 70.4% of drafts accepted, so about 19.6 ms per round |
| Round time | wait for rings (GPU work) 10.9 ms, CPU expert pool 5.2 ms, host 0.7, commit 0.4, drafting 1.2 |
| CPU pool | gate/up 3.2 + down 1.7 ms per round, 36.1 GB/s during the row phases |
| Expert cache | 7,002 slots pre-filled from a profile, 82.8% hits; 90/256 of misses over PCIe |
| Verify union on the CPU | 2.63 distinct of 3.15 routed experts per layer per round |
| Prefill, 32K | 47.4 s (675 tok/s); **21.6 s of it in PLE reads**; 215,293 expert blobs streamed by DMA |

From `gen-stages.log` (`--gpu-stages`), GPU time per token on the captured graph, one token,
48 layers, excluding expert hits and the head:

| Stage | ms per token |
|---|---:|
| Mixer (hyper-connection read + attention + write) | 4.75 (36 GDN layers 3.36, 12 QSA layers 1.39) |
| FFN front + router | 2.65 |
| Post | 0.25 |
| **Total** | **7.65** |

The dense path reads about 3.5 GB. At about 960 GB/s that is about 3.6 ms, so this path
runs at roughly 45-50% of VRAM bandwidth (estimate).

## 2. PLE (n-gram table) reads are serialised

Strata `--stats` at 32K:

| Arm | Prefill | PLE blocked | Read p50 |
|---|---:|---:|---:|
| `--ple-inflight 64` (default) | 675 tok/s | 20.1 s | 4.5 ms |
| `--ple-inflight 256` | 676 | 20.0 s | 16.3 ms |
| `--ple-inflight 1024` | 671 | 20.3 s | 63.4 ms |
| `--ple-io mmap` | 580 | (PLE 29.2 s) | n/a |

Every arm plateaus at about 16K reads per second, for 320K reads of 4 KiB.
`tools/ssdrand` on the same file (O_DIRECT, random 4 KiB, `ssdrand.txt`):

| Queue depth | 1 | 16 | 64 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| IOPS | 16,001 | 243,331 | 713,684 | 1,090,009 | 1,115,626 |
| p50 latency | 63 µs | 64 µs | 80 µs | 179 µs | 228 µs |

- **Strata's rate equals the drive at queue depth 1.** Its Linux PLE path is effectively
  one synchronous reader, whatever `--ple-inflight` says.
- **At queue depth 64, the 32K prompt's rows would take about 0.45 s instead of 20 s.**
  That puts a Strata-shaped engine at about 1,150 tok/s for 32K prefill (estimate).
- All of a prompt's rows are known before prefill starts, so they can be read at full
  depth ahead of the layers.

## 3. llama.cpp prefill vs ubatch at 32K, `-lm dio` (`ubatch_dio.md`, random tokens, 1 run)

| ubatch | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|
| pp32768 tok/s | 911 | 1,103 | 1,196 |

For comparison, the mmap load in window A gave 475 / 574 / 701.

## 4. Baselines on natural text, 1K / 32K / 131K / 245K (growing conversation, 384 tokens)

Decode tok/s, runs interleaved. Draft acceptance is summed over the runs.

| Arm | Runs | 1K | 32K | 131K | 245K |
|---|---:|---:|---:|---:|---:|
| Strata 0.1.6, tuned, greedy | 2 | 84.0 (85.8, 82.2) | 90.1 (91.8, 88.4) | 95.7 (101.2, 90.3) | 103.0 (113.5, 92.5) |
| &nbsp;&nbsp;draft acceptance | | 75% | 73% | 87% | 95% |
| Strata 0.1.6, tuned, t=1.0 / top_p 0.95 / top_k 20 | 2 | 72.5 (73.9, 71.1) | 74.7 (73.6, 75.8) | 70.7 (77.5*, 63.9) | 62.2 (58.5, 66.0) |
| &nbsp;&nbsp;draft acceptance | | 60% | 61% | 75% | 62% |
| llama.cpp dev tree, 48-slot cache, no MTP | 1 | 39.4 | 42.2 | 38.9 | 32.8 |

\* That run stopped at 246 tokens (EOS).

**Prefill is not comparable between the engines here:**
- Strata re-reads the whole prompt at every depth: 530 / 683 / 824 / 1,222 tok/s over
  1K / 32K / 131K / 245K tokens.
- llama.cpp prefills only the new tokens: 478 / 649 / 543 / 376 tok/s over 1K / 31K / 99K
  / 114K new tokens.

**Reading the numbers:**
- **Decode at depth is mostly draft acceptance.** Deep wikitext continuations are easy to
  draft (greedy accepted 95% at 245K).
- **Greedy is not reproducible in Strata.** The two greedy runs generated different text,
  and speed follows the accepted drafts. Gate comparisons need acceptance reported
  alongside tok/s, and more than one prompt.
- **Sampling at t=1.0 costs 14-40% against greedy** on this text: 72.5 / 74.7 / 70.7 / 62.2
  tok/s. The P2 gate (≥80 at 32K, ≥72 at 250K, t=1.0) is above Strata's current numbers
  on natural text at both depths.
