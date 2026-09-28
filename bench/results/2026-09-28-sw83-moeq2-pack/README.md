# sw83: moe_q2 with the activation scale and magic sum packed (2026-09-28)

**Before:** `block_mma` loaded each token's activation scale d and magic sum m for a pair of
64-weight blocks as four 4-byte shared-memory loads.

**Now:**
- `k_quant` and the gate/up epilogue write {d, m's bits} as float2.
- The stages hold them interleaved, so a token's pair is one 16-byte load, and the down kernel
  copies them with half the `cp.async`s.
- The same arithmetic, so the same values.

**`test_moe_q2`** (`test_moe_q2.txt`, T = 8,192 over 512 experts, gate/up + down):
5.89 → **5.71 ms** (137 → 141 TOPS). Errors unchanged (1.83e-4 relative).

**KLD** (fp16 KV, 1,024-token chunks): **0.008879**, identical to sw81 and sw82.

**Prefill** (q8 KV, automatic 16K chunks, one run each):

| depth | sw82 | sw83 |
|---|---|---|
| 32K | 6,186.7 tok/s | **6,216.1** (+0.5%) |
| 64K | 6,221.8 | **6,239.1** (+0.3%) |

The kernel's remaining gap to its measured arithmetic ceiling (~283 TOPS for gate/up) is the
per-32 scale arithmetic: about 7 ALU operations per accumulator element per 64-weight block.
They cannot be cut without per-64 activation scales, whose KLD cost was measured and rejected
in sw57.
