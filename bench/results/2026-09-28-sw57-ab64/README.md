# sw57: moe_q2 with activation scales per 64 against per 32 (2026-09-28)

Per 64 (`FLASHRT_MOE_AB64=1`) matches the Q2_0 weight block, so each 64-block's two MMAs
accumulate in int32, with one conversion and one scaled add per element instead of two
(`tools/bench_mma`: the arithmetic ceiling at 8 warps per SM is 364 TOPS against 283). This build
also has the down kernel with the tile's activations resident in shared memory.

`test_moe_q2`, 8,192 tokens: per 32 5.88 ms, per 64 5.07 ms. Relative error against MMQ: 1.8e-4
per 32, 1.35e-2 per 64.

32,768 tokens, q8 KV, automatic chunks (16,384), one run each:

| Activation scales | prefill | KLD fp16 KV | KLD q8 KV |
|---|---|---|---|
| per 32 (default) | 5,155.6 tok/s | 0.008385 (sw56) | 0.008176 (sw56) |
| per 64 | 5,272.2 tok/s (+2.3%) | 0.008678 (96.75%) | 0.008872 (96.58%) |

The per-64 KLD is higher. The q8 difference (+0.0007) is outside the run-to-run spread (about
±0.0002), so **per 32 stays the default**; per 64 is an option.
