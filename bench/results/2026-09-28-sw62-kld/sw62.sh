#!/bin/bash
# sw62: KLD gate of the sw61 build (every prefill default, the 4-accumulator GDN kernel): logits
# from prefill chunks, fp16 and q8 KV
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw62; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cd $BENCH/kld
run() { name=$1; shift; wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run chunk1024-f16 --prefill-chunk 1024
run chunk1024-q8 --prefill-chunk 1024 --kv q8
echo done > $O/DONE
