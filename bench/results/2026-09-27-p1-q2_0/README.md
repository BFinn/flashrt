# P1: Q2_0 CPU expert kernel on the target box (2026-09-27, 16:5x)

Ryzen 9 7900X (Zen 4, AVX-512 VNNI), DDR5-3600. Nothing else ran on the box: no live
service, and the agent stack was stopped. `tools/bench_q2_0` (1 run each, 3 s) runs whole
experts (gate, up, SwiGLU, down; 1.38 MB each) picked at random from a 3.3 GB THP arena of
random codes. Each expert runs on one thread. Raw output: `bench_q2_0.txt`.

| Threads | Tokens per pass | Experts/s | GB/s of weights read | µs per expert per thread |
|---:|---:|---:|---:|---:|
| 1 | 1 | 9,994 | 13.8 | 100 |
| 1 | 3 | 5,164 | 7.1 | 194 |
| 6 | 1 | 35,233 | **48.7** | 170 |
| 8 | 1 | 34,999 | 48.4 | 229 |
| 12 | 1 | 34,351 | 47.5 | 349 |
| 8 | 3 | 31,906 | 44.1 | 251 |
| 12 | 3 | 33,742 | 46.6 | 356 |

- **Six cores reach the DRAM read ceiling.** `membw` measured 48.6 GB/s on 6 threads (P0
  window A). The other six cores are free for the host loop and the PLE reader.
- **One core is compute-bound at 13.8 GB/s.** Three tokens cost 1.9× one token on one core,
  but from DRAM with 8-12 threads a 3-token pass still runs at 44-47 GB/s of weights, which
  is 3× the token work per byte.
- **Strata's CPU pool** measured 36.1 GB/s during its row phases, with 8 workers
  (`2026-09-27-p0c`).

Correctness (`test_q2_0.txt`): the scalar reference matches ggml's
`ggml_vec_dot_q2_0_q8_0_generic` bit for bit. The AVX-512 kernel is within 1e-6 of output
RMS for 1-4 tokens, and within 1e-7 relative L2 through a whole expert.

## Expert arena load (`tools/fr_load`, 2026-09-27, 1 run each)

All 24,576 experts, read from the GGUF with O_DIRECT by 12 threads and repacked into a
31.69 GiB THP arena (4 KiB-strided blobs):

| Variant | Map / pre-fault | Load | Total | Read rate |
|---|---:|---:|---:|---:|
| Arena pre-faulted by 12 threads (`fr_load_pretouch.txt`) | 13.1 s | 5.8 s | 18.9 s | 5.87 GB/s |
| Loader first-touches the pages (`fr_load.txt`) | 0.0 s | 12.5 s | **12.5 s** | 2.73 GB/s |

- The SSD delivers about 5.9 GB/s. Page faulting (zeroing 31.7 GiB) costs about as much as
  the read.
- About 40-44% of the arena lands on huge pages with THP, since no hugetlb pool is
  configured.
- **Verification:** for 64 random (layer, expert) pairs × 3 matrices, a random row was read
  straight from the GGUF, dotted with ggml's semantics, and compared with the AVX-512 kernel
  on the arena blob. 0 mismatches, worst relative error 2.4e-6. One real expert's full FFN
  matches the reference to 1e-7 relative L2.
