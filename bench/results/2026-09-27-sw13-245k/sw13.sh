#!/bin/bash
# speed work 13: decode at 245,760 depth (P1 gate depth 250K), fp16 KV, adaptive cache. One prefill, 3 windows of 128.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw13; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 7200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 > $O/adapt_245k.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
