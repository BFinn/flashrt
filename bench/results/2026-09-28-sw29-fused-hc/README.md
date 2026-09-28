# sw29: fused hyper-connection kernels for windows; confidence-gated drafting (2026-09-28)

`k_hc_down` / `k_hc_up_mix` templated on the token count (1..4), each BF16 weight read once for
the window. Per token the arithmetic is the same as the one-token kernels: plain decode tokens
are unchanged (same 24 tokens as sw23), 102.49 tok/s at 2K.

2K, greedy, Q4_0 head, 32K vocab, window graphs:

| Arm | tok/s | tokens/round | drafts/round | verify ms/round |
|---|---|---|---|---|
| plain (no head) | 102.49 | 1 | | |
| `--spec 1` | **123.06** | 1.641 | 1.00 | 12.70 |
| `--spec 2` | **121.02** | 1.984 | 2.00 | 15.19 |
| `--spec 3` | 106.79 | 2.124 | 3.00 | 18.15 |
| `--spec 3 --draft-pmin 0.3` | 104.32 | 2.000 | 2.60 | 17.55 |
| `--spec 3 --draft-pmin 0.6` | 113.21 | 1.684 | 1.19 | 13.82 |

- Speculation now gains 20% at 2K with 1-2 drafts.
- Gating drafts on the head's own probability (over the trimmed vocabulary) does not beat a
  fixed K here: the probability is not calibrated enough.
