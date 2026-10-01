# sw126: P-5's last decode items: the argmax on a cluster, the indexer's pooled keys in fp16 (2026-10-01)

**Why** (sw122's decode profile at 245K, plain):
- `k_argmax` took 79 µs per token. It is one CTA of 1,024 threads reading the 1 MB logits row at
  about 13 GB/s.
- `k_idx_scores128` took 37.4 µs per call, 12 calls per token (4.2% of GPU time). It reads 61,440
  fp32 pooled keys of 512 bytes at ~840 GB/s, so it is bound by their bytes.

**Changes:**
- **Argmax on an 8-CTA cluster** (787f6af, `FLASHRT_ARGMAX_CLUSTER=0` restores the old kernel).
  Each thread takes every 8,192nd value with four loads in flight, in rising index order. CTA 0
  takes the CTAs' results in rank order through distributed shared memory. Same rule as before:
  the largest value, the lowest index on ties, never NaN.
- **Pooled keys stored in fp16** (e926085).
  - The prefill's tensor-core scores already rounded both queries and keys to fp16 before their
    mma. Storing the keys rounded the same way (round to nearest) leaves prefill bit-identical.
  - The decode kernel reads 256 bytes per key instead of 512, with the warp's four keys loaded
    before the dot products. The queries stay fp32.
  - The buffer is allocated for the whole context (65,537 keys × 12 layers at 262K), so it frees
    about 200 MB. The expert cache takes that from free VRAM.
  - State files keep fp32 keys and convert on save and load, so the saved snapshots stay valid.

## Checks

| Check | Result |
|---|---|
| `test_argmax` (new): 262,144 and 40,001 values (unaligned), 33 and 1; random, many ties, the maximum at either end, NaN every 7th, all -inf | 24 / 24 equal the CPU reference; 5.4 µs per call against 39.0 (isolated) |
| sw112's fingerprint, base build (787f6af: argmax only) against sw122 | **identical** (`fp-base.txt`) |
| Fingerprint of the new build (fp16 keys) against sw122 (`fp-new.txt`) | KLD fast 0.008960 → **0.008910**; verify windows (win3, hot set 512) 0.008824 → **0.008513**; chunks (prefill path) **identical**; greedy wikitext tokens identical (`16ed386cd612`); teacher-forced hit rates ±0.15 points; the sampled text differs (it samples from changed logits) |
| ctest | 21 / 21 (two need the model and skip) |

Both KLD values moved down, by about sw94's perturbation spread (~0.0002-0.0003): the gate passes,
and the change does not improve quality.

## Speed

Three arms interleaved, 4 rounds, teacher-forced from the saved states, 256 tokens (`ab/`):
- old: the base build with the argmax off;
- argmax: the base build;
- new: argmax plus fp16 keys.

| tok/s, means of 4 | old | argmax | new | new vs old |
|---|---:|---:|---:|---:|
| 245K plain | 94.48 | 95.80 | **98.63** | **+4.4%** |
| 32K plain | 113.97 | 111.80 | 113.87 | −0.1% |
| 245K `--spec 2` | 89.90 | 88.66 | **91.61** | +1.9% |
| 32K `--spec 2` | 105.24 | 106.09 | **108.92** | +3.5% |
| expert-cache slots (245K) | 8,674 | 8,674 | **8,819** | +145 |

Alone, the argmax (~0.5% expected) is not distinguishable from the run-to-run spread (about 2%).
It is kept because it is exact and removes 74 µs per token.

**Kernel times in context** (nsys, 245K plain, old against new, `p_*_cuda_gpu_kern_sum.csv`):
- `k_idx_scores128`: 37.5 → **21.0 µs** per call (−0.20 ms per token);
- argmax: 79.1 → **5.2 µs** per call;
- the expert cache: 8,663 → 8,809 slots.

## Found on the way

The sw124 window 9 run showed **prefill 9.5-12% slower than sw119**, and sw124's README missed it:
32K 5,665 → 5,060 tok/s of new tokens. The cause is sw121's 8-CTA selection, which prefill
sub-batches also run, while sw121 had measured decode only. sw127 measures and fixes it.
