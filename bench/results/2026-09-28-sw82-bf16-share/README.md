# sw82: prefill BF16 products of one input share one BF16 conversion (2026-09-28)

**Before:** each BF16 product converted its fp32 input to BF16 itself (`k_to_bf16`: 0.23 s at 64K
in sw80).
- The router and the shared expert's gate read the same x.
- So do GDN alpha and beta, and the indexer q and k.

**Now:**
- `linear_shared` (prefill branch) converts once (`gemm::to_bf16` into the staging area) and
  runs the BF16 products from it (`gemm::gemm_bf16`).
- The router and the shared-expert gate are computed together (`route_and_gate` in
  moe_stream.cu).
- It is the same conversion and the same cuBLAS call, so the same values.

**KLD** (fp16 KV, 1,024-token chunks): **0.008879**, identical to sw81.

**Prefill** (q8 KV, automatic 16K chunks, one run each):

| depth | sw81 | sw82 | |
|---|---|---|---|
| 32K | 6,134.1 tok/s | **6,186.7** | +0.9% |
| 64K | 6,157.0 | **6,221.8** | +1.1% |
