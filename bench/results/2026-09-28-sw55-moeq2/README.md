# sw55: routed experts on int8 tensor cores straight from the planar arena layout (2026-09-28)

`kernels/cuda/moe_q2.cu` (`moe_q2::run`) replaces, for prefill chunks:
- the conversion of each layer's expert slice to ggml's Q2_0 layout;
- ggml's MMQ for gate, up and down;
- the SwiGLU kernel and the down input's quantization.

**What it does:**
- **Weights:** one 32-bit shared-memory load per lane gives both k-steps' `mma.m16n8k32` A
  fragments of a 64-weight block (shifts 0/2/4/6 and a mask: the planar layout's purpose).
- **Unsigned codes:** Q2_0's -1 is folded into the activation blocks' `kMagic - sum`, which also
  turns the int32 results into floats without I2F.
- **Gate and up run in one kernel.** Its epilogue computes SwiGLU and quantizes the down input
  (int8 per 32).
- **Tile list built on the GPU,** with no host sync. The tiles are 64 tokens x 128 weight rows.

`FLASHRT_MOE_Q2MMA=0` switches back to the MMQ path.

**`test_moe_q2`** (`test_moe_q2.txt`), layer 0's real experts, random activations and routing:
relative error 8.6e-5 / 1.9e-4 / 1.8e-4 against the MMQ path at 96 / 1,200 / 8,192 tokens. The
expert product at 8,192 tokens takes 6.18 ms against 10.44 ms (130 against 77 TOPS).

**Prefill** (q8 KV, automatic chunk length, one run each):

| Prompt | MMQ path | moe_q2 |
|---|---|---|
| 32,768 | 4,019.4 tok/s (chunks of 16,384) | **5,111.4 tok/s** (+27%) |
| 245,760 | 3,622.2 tok/s (chunks of 11,264) | **4,871.3 tok/s** (+34%; chunks of 13,824) |
| 65,536 (nsys run) | | 5,124.7 tok/s |

About 70 KB per token less (no converted slice, no float gate/up outputs). That lets 245K run
chunks of 13,824. Decode after prefill is unchanged (67-86 tok/s over 8-16 tokens).

**64K profile** (`p64k_cuda_gpu_kern_sum.csv`): 12.5 s of kernels (sw51: 16.9 s).
- **Expert products:** 1.19 s (gate+up, about 173 TOPS) + 0.92 s (down, about 112 TOPS); the
  MMQ path took 5.4 s.
- **Largest items now:** the Q3_K MMQ (1.25 s), GDN (0.97 s), attention (0.95 s), the hc
  elementwise kernels (about 2 s together).

The KLD gate is in sw56.
