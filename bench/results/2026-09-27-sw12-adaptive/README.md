# Speed work 11-12: adaptive expert cache, 32K with it, crash hunt, fp16 KV (2026-09-27)

Raw outputs are here and in `2026-09-27-sw11-adaptive` (which also has a 256-token routing
trace, `trace_2k.i16`: int16 [256][48][10]).

## The adaptive cache

The policy is decayed LFU with hysteresis, the rule `tools/cache_sim.py` calls `dlfu`, with the
same defaults:
- Every access adds 1 to a (layer, expert) count, and every 4 tokens all counts are multiplied
  by 0.7.
- A missed expert is admitted if its count is ≥ 2 and ≥ 1.5 times the weakest resident's.
- At most 8 uploads are in flight.

Mechanics (`arch/qwen4exp/moe_fast.cu`, `CacheManager`):
- **Host work overlaps the GPU.** A token's kernels are enqueued first. The host then updates the
  counts from the previous token's routing, commits finished uploads and picks victims while the
  GPU runs.
- **No in-flight kernel reads a slot being rewritten.** Evictions and commits reach the device
  table through one small kernel after the token. Uploads run on a copy stream: pinned arena →
  staging → unrepack into the slot. They wait on an event recorded after the token, and a new
  expert enters the table only after its upload has completed.
- **The arena is registered with CUDA** (32 GB, 1.6 s), so uploads are truly asynchronous.

### Results

| Arm | Runs | tok/s | Hit rate |
|---|---:|---|---:|
| 2K prompt, 256 tokens, static cache | 3 (sw12) | 78.7 / 80.0 / 80.5 | 86.17% |
| 2K prompt, 256 tokens, adaptive | 3 (sw12) | 81.2 / 80.9 / 81.1 | 89.08% (1,030 swaps) |
| 32K prompt, 3 windows of 128, static (sw9) | 1 prefill | 78.9 / 77.8 / 70.7 | 85.6 / 85.1 / 75.0% |
| 32K prompt, 3 windows of 128, adaptive | 1 prefill | **78.5 / 77.9 / 79.5** | 86.8 / 87.8 / 89.7% |
| New text (`fr_kld --fast`, cache from chunk 0's prompt), static (sw8) | 1 | | 65.61% |
| New text, adaptive (sw11) | 1 | | **89.32%** (28,797 swaps) |

- **The cache now follows the text.** On text the prompt did not contain, the hit rate goes from
  65.6% to 89.3%. At 32K, the third window no longer degrades.
- **Fast-path KLD with the adaptive cache:** 0.008686, median 0.00109, same top-1 96.62%, PPL
  ratio 1.0011 (`kl8k-fast-adaptive.log` in sw11). The gate (≤ 0.03) holds.

## Crash hunt

The two "unspecified launch failure" aborts (sw7 fr_kld, sw11 static run 2) match Xid 43 in the
kernel log with no MMU-fault Xid. That fits the doorbell combine kernel's 10 s `__trap()`. That
timeout used unsigned `%globaltimer` arithmetic, so a timer read that goes backwards would fire
it at once. The kernel now needs 10 s and 10M polls, uses signed arithmetic, and records a
timeout in the mailbox instead of trapping. The host then reports the layer and the miss server's
state. Tool stdout is line-buffered, so the log survives an abort. The 6 runs and the 32K run of
sw12 did not crash. The root cause is still open.

## fp16 KV cache

The QSA K/V values were already rounded to fp16 (matching llama.cpp's F16 cache), but they were
stored as F32. They are now stored as fp16: same numerics, half the VRAM. `fr_parity qsa` on
`ar65` (`kv16_ar65.txt`) is identical line for line to the F32-storage run. This is what lets a
250K context fit. It is not KV compression.
