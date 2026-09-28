# sw81: prefill routing a warp per token; the GDN q/k norm in the conv kernel (2026-09-28)

**Routing.** The sw80 profile showed `k_route_topk` (softmax and top-10 over 512 router logits
per token) at 1.08 ms per 16K-token chunk, against ~36 µs for its 32 MB of logits.
- It ranked every expert against all 512 (O(E^2)).
- **Two steps** (`test_route_topk`, 16,384 tokens, coarse logits with many exact ties; exact
  against a CPU reference in ids, counts and weights):
  1. Ranking only the candidates (p >= the 10th-largest warp maximum, as decode's `k_route`):
     1.065 → 0.770 ms.
  2. **A warp per token:** 16 logits per lane in registers, max and sum by shuffles, then 10
     rounds of warp argmax (ties to the lower index): **0.041 ms (26x).**
  - `FLASHRT_ROUTE_WARP=0` gives the block kernel.
- The weights differ from the old kernel's in the last bit (another summation order).

**The q and k L2 norm** now runs in `k_gdn_conv_par`, a block per 128-dim head, as it already
did in decode (sw74). This saves one read and write of q and k per layer (0.094 s at 64K in
sw80). Same block reduction, so the same values.

**KLD** (`kld-chunk1.log`, fp16 KV, 1,024-token chunks): **0.008879** (96.83%), in the band of
earlier runs (0.0082-0.0090).

**Prefill** (q8 KV, automatic 16K chunks, one run each):

| depth | sw72 | sw81 | |
|---|---|---|---|
| 32K | 5,977.7 tok/s | **6,134.1** | +2.6% |
| 64K | 6,011.5 | **6,157.0** | +2.4% |
