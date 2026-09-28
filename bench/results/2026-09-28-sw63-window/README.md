# sw63: where a verify window's time goes now; can routing be forecast for prefetch? (2026-09-28)

P2 conditions: 32K from `state-32k-q8-mtp.bin`, host KV + hot set 4,096, temperature 1.0,
top-k 20, top-p 0.95, seed 1.

## Plain decode against one draft per round (3 windows of 128 tokens, one run each)

| Arm | tok/s per window | hit rate |
|---|---|---|
| plain (`plain_32k.txt`) | 95.6 / 100.5 / 97.6 | 94.75% |
| `--spec 1` (`spec1_32k.txt`) | 103.4 / 111.9 / 108.7 | 90.19% |

A round costs 1.05 ms of drafting, 13.30 ms of verifying and 0.20 ms of commit, for 1.569
tokens.

The P3-era build (e738c18, `old_spec1_32k.txt`), same command: draft 1.05 ms, verify 12.95 ms,
106.5 / 107.4 / 118.5 tok/s. Its tokens differ from token 9 on. The likely reason: the expert
cache's slot count depends on free VRAM, which this session changed, so other experts are GPU
hits (dp4a) or CPU misses (AVX-512) — same math, other rounding. The fast-path KLD covers this.

## Kernel time per plain token against per round (nsys `--cuda-graph-trace=node`, `kcmp2.py`)

Total 10.08 ms per plain token, 13.67 ms per round. The second token adds about 3.6 ms:

| Part | per token | per round | added |
|---|---|---|---|
| MoE hits (plain `k_moe_gate_up` + `k_moe_down`; window: grouped kernels) | 0.99 | 1.58 | +0.59 (the two tokens' experts barely overlap) |
| waiting for CPU misses (`k_moe_combine_db`) | 1.03 | 1.78 | +0.75 |
| attention (`k_attn_part`) | 0.43 | 0.77 | +0.34 |
| GDN delta rule (with the window backup) | 0.21 | 0.40 | +0.19 |
| hc mixes (`k_hc_down` + `k_hc_up_mix`) | 2.01 | 2.31 | +0.30 |
| dense mat-vecs (MMVQ, Q3R, BF16) | ~3.4 | ~4.3 | ~+0.9 (includes the draft head's) |

The target's LM head (248,320 x 2,560 Q5_K, about 437 MB) is read once per round, in about
0.49 ms (~890 GB/s): at bandwidth. Nothing stands out as waste. A window's extra cost is the
second token's experts plus modest growth elsewhere.

## Forecasting routing for prefetch (a temporary diagnostic, `forward_ref_prefetch_diag.cu.txt`)

Could the missing experts be uploaded a step ahead? Layer L's router was applied to earlier
hidden states:
- "attn-ahead": layer L's attention-mix output, one attention block early;
- "layer-ahead": layer L-1's FFN-mix output, one layer early.

The diagnostic measured recall of the actual top-10, and the share of actual cache misses the
forecast top-K contains. Plain decode, 192 tokens, no graphs (`diag_prefetch_*.txt`):

| Context | misses | forecast | top-10 | top-12 | top-16 | top-24 |
|---|---|---|---|---|---|---|
| 32K | 6.28% | recall, attn-ahead / layer-ahead | 59.8 / 68.2% | 65.6 / 74.0% | 73.7 / 81.0% | 82.1 / 87.1% |
| | | misses caught, attn-ahead / layer-ahead | 49.5 / 50.2% | 56.2 / 57.6% | 65.4 / 67.9% | 75.2 / 78.1% |
| 245K | 12.06% | recall, attn-ahead / layer-ahead | 55.8 / 65.7% | 61.4 / 71.5% | 69.5 / 78.5% | 78.8 / 85.0% |
| | | misses caught, attn-ahead / layer-ahead | 47.1 / 49.6% | 52.7 / 56.6% | 61.3 / 66.0% | 72.3 / 76.4% |

**Not pursued.**
- **Transfer time is not the limit.** Each upload takes about 29 µs per expert over PCIe Gen5,
  well under a decode layer's 200-290 µs.
- **Host DRAM bandwidth is the limit.** Every upload reads the expert from host DRAM, the same
  33.6 GB/s (STREAM; DDR5-3600, EXPO off) that bounds the CPU miss path.
- **A forecast that catches half the misses also uploads about as many unused experts.** So
  prefetching roughly doubles the host-DRAM reads per miss it removes. At 245K with top-16:
  about 190 MB of uploads per token against 77 MB of misses today, which slows the CPU misses it
  was meant to relieve.
