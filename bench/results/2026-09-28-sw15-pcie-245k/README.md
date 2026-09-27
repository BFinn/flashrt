# Speed work 15: 245K decode after the depth fixes; PCIe misses (2026-09-27/28)

**Build:** sw14 (dp4a MoE hit kernels, coalesced indexer scoring, 8-bit radix select), fp16 KV
and the adaptive cache, plus PCIe misses.

**PCIe misses:** the GPU reads floor(frac × misses) of each layer's misses (at most 4) straight
from the mapped arena over PCIe, and the CPU serves the rest.

The 245K runs use a new state snapshot (`fr_bench --save-state` / `--load-state`, 6.5 GB file,
loads in 1.4 s). It lets decode at depth be measured without the 34-minute prefill. A
snapshot only counts for speed with the kernels that wrote it.

## 245,760-token wikitext prompt, 3 windows of 128 tokens (P1 gate depth, target ≥ 47 tok/s)

| Arm | Window 1 | Window 2 | Window 3 | Hit rate |
|---|---:|---:|---:|---:|
| sw13 (before the depth fixes), fresh prefill | 41.90 | 44.87 | 44.42 | 57.1% |
| **Fresh prefill (120.7 tok/s, 2,036 s), PCIe frac 0.5** | **55.80** | **51.97** | **62.32** | 60.3% |
| From the saved state, PCIe off | 57.40 | 55.05 | 60.20 | 59.0% |
| From the saved state, PCIe frac 0.5 | 55.27 | 54.43 | 56.27 | 57.8% |

- **The P1 gate at 250K holds:** 52-62 tok/s against the target of 47.
- **At this depth the fp16 KV (6 GB) leaves 3,458-3,511 slots** (14% of experts), so the hit rate
  is 51-71% and varies from window to window.

## PCIe misses at 2K (2 runs each, 256 tokens)

| Arm | tok/s | Misses split |
|---|---|---|
| PCIe off | 88.43 / 86.95 | 10,715 on the CPU |
| PCIe frac 0.5 | 86.73 / 87.37 | 7,736 CPU + 3,102 PCIe |

- **PCIe misses do not help,** neither at 2K (about 0.9 misses per layer) nor at 245K (about 4.3).
  A zero-copy read of a 1.32 MB expert lengthens the GPU's part of each layer by more than it
  saves the CPU. They are off by default now (`--pcie-frac` stays).
- **Fast-path KLD with PCIe misses:** 0.008847 (`kl8k-fast-pcie.log`), so the numerics are fine.
