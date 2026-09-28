#!/bin/bash
# sw25: KLD gate for speculative verify windows (random rejected tails, rewinds)
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw25; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cd $BENCH/kld
wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --window 4 > $O/kl8k-fast-win4.log 2>&1; echo "win4 rc=$?"
echo done > $O/DONE
