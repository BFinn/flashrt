#!/bin/bash
# P1 KL gate: llama.cpp noise floors, then flashrt, against the f16-KV base (kl8k-f16.bin)
cd $BENCH/kld
DEV=$LLAMA_CPP/build/bin
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
C=$DATA/wikitext-2-raw/wiki.test.raw
while systemctl --user is-active -q kld-ref; do sleep 10; done
wait_vram() { for _ in $(seq 300); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 600 ] && return 0; sleep 1; done; }
COMMON=(-m $M -f $C -c 8192 --chunks 2 --no-warmup -lm dio -ngl 99 -ot "ffn_.*_exps=CPU" -fa on -t 12 --kl-divergence-base kl8k-f16.bin --kl-divergence)
wait_vram; echo "== llama.cpp f16 KV, ub 512 $(date +%T)"
LD_LIBRARY_PATH=$DEV $DEV/llama-perplexity "${COMMON[@]}" -b 512 -ub 512 -ctk f16 -ctv f16 > kl8k-ub512.log 2>&1
wait_vram; echo "== llama.cpp q8_0 KV, ub 16 $(date +%T)"
LD_LIBRARY_PATH=$DEV $DEV/llama-perplexity "${COMMON[@]}" -b 16 -ub 16 -ctk q8_0 -ctv q8_0 > kl8k-q8.log 2>&1
wait_vram; echo "== flashrt $(date +%T)"
$FLASHRT/build/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 > kl8k-flashrt.log 2>&1
echo "GATE_DONE $(date +%T)"
