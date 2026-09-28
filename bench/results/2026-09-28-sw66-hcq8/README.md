# sw66: hyper-connection down and up matrices as Q8P (2026-09-28)

The four hc matrix families (`hc_{attn,ffn}_{down,up}`, BF16, 1.26 GB) are read in full every
decode token, about 2 ms of a 10 ms token at 32K (sw63). GpuWeights can now convert them at
load to Q8P:
- int8 values `[rows][cols]` plus fp16 scales per 32 (Q8_0's values, planar);
- the fused decode kernels read it through a weight accessor (`WQ8P`);
- the other hc paths dequantize a mix's matrices into a BF16 scratch (13 MB) just in time.

This halves those bytes and frees about 590 MB for expert-cache slots. (The first run of this
script had the Q8P slots sized at 33/32 of the element count, not 17/16, so the scales overran
into the next tensor: NaNs and illegal accesses. Fixed before these numbers.)

**KLD with all four families as Q8P** (`FLASHRT_HC_Q8=1`), `fr_kld --ctx 8192 --chunks 2`:

| Run | KLD mean | same top-1 | BF16 (this session) |
|---|---|---|---|
| logits from prefill chunks, fp16 KV, `--prefill-chunk 1024` | 0.010554 | 96.39% | 0.0082-0.0087 |
| fast decode path, hot set 512 | 0.010502 | 96.58% | 0.0084-0.0086 |
| verify windows of 3, hot set 512 | 0.010660 | 96.34% | 0.0092 (sw33) |

That is about +0.002, beyond the spread. sw67 splits it by matrix kind.

**Speed**, teacher-forced (`--teacher`), P2 conditions, from the saved states, 6 windows of 128
tokens (Q8P arms: 426-560 more cache slots):

| Arm | BF16 | all Q8P |
|---|---|---|
| 32K plain | 92.9 / 95.4 / 96.8 / 98.4 / 98.7 / 94.2 (mean 96.1) | 103.3 / 106.4 / 106.3 / 104.6 / 104.0 / 100.0 (**104.1**, +8.3%) |
| 32K `--spec 1` | mean 103.1 (verify 12.72 ms) | mean **110.4** (+7.1%; verify 11.78 ms) |
| 245K `--spec 1` | mean 83.2 (verify 16.38 ms) | mean **85.2** (+2.4%; verify 15.90 ms) |
