#!/bin/bash
# sw36: first chunked prefill (experts streamed to the GPU): speed and sanity at 8K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw36; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for c in 2048 4096 64; do wait_vram; timeout 1800 $B/fr_bench $M --ids $I --n-prompt 8192 --gen 32 --prefill-chunk $c > $O/chunk$c.txt 2>&1; echo "chunk $c rc=$?"; done
echo done > $O/DONE
