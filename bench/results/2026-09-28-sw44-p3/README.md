# sw44: KLD of the host-KV prefill path (2026-09-28)

| Run | KLD mean | same top-1 |
|---|---|---|
| sw44 first script: chunk logits, `--prefill-chunk 1024 --kv-hot 512` (no `--fast`) | **0.008813** | 96.51% |
| `--fast --prefill-chunk 2048 --kv-hot 512`: chunked prefill, then the decode path | **0.008598** | 96.75% |

(The chunk-logits log is `chunk1024-hot512.log` of the first run of this script; the fast run was
repeated after `fr_kld` learned to lend the expert cache's VRAM to each chunk's prefill.)
