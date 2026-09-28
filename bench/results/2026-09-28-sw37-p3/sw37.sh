#!/bin/bash
# sw37: KLD of logits straight from prefill chunks (no --fast), chunks of 1024 and 4096
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw37; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cd $BENCH/kld
for c in 1024 4096; do wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --prefill-chunk $c > $O/kl8k-chunk$c.log 2>&1; echo "chunk $c rc=$?"; done
echo done > $O/DONE
