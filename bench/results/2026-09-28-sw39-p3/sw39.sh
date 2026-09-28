#!/bin/bash
# sw39: chunked prefill speed at 16K / 32K, chunks of 4096 and 8192
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw39; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for c in 4096 8192; do for n in 16384 32768; do wait_vram; timeout 1800 $B/fr_bench $M --ids $I --n-prompt $n --gen 16 --prefill-chunk $c > $O/n${n}_c$c.txt 2>&1; echo "n $n chunk $c rc=$?"; done; done
echo done > $O/DONE
