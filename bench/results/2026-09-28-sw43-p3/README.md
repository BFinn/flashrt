# sw43: host KV with a VRAM mirror during prefill (2026-09-28)

With host-resident KV (the depth decode configuration), prefill chunks now write the host store
and a full VRAM mirror (sized to the prompt) and attend from the mirror; it is freed with the
chunk buffers, before the expert cache is built. 245,760 tokens, chunks of 8,192, `--kv-hot 4096`.

- **Prefill: 1,940.4 tok/s (126.7 s),** per chunk 1,719 → 2,313 → 1,666 as in the VRAM run
  (sw42: 737.6 without the mirror).
- Decode after it: 70.5 tok/s over 16 tokens (normal).
