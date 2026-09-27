#!/bin/bash
# speed work 9: decode at 32K depth (P1 gate depth). One prefill, 3 decode windows of 128 tokens.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw9; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 3 > $O/db_32k.txt 2>&1; echo "32k rc=$?"
echo done > $O/DONE
