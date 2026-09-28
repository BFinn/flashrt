# sw52: prefill chunks of 16,384 (2026-09-28)

A longer chunk gives each expert more tokens: about 320 at 16,384 against 160 at 8,192. That
fills the 128-wide MMQ tiles better (83% against 62%). q8 KV, one run each:

| Prompt | chunks of 8,192 (sw51) | chunks of 16,384 |
|---|---|---|
| 32,768 | 3,689.4 tok/s | **4,005.3 tok/s** (+8.6%) |
| 65,536 | 3,655.2 tok/s | **3,975.4 tok/s** (+8.8%) |
| 245,760, q8 in VRAM | 3,361.7 tok/s (sw50) | out of memory (`cudaMalloc expert stream`) |

At 245K the q8 KV (3.2 GB) leaves too little room for the chunk buffers. sw53 measures them, and
sw54 picks the chunk length from free VRAM.
