#!/bin/bash
# sw42: chunked prefill at 245K (chunks of 8192): host KV + hot set, and q8 KV in VRAM
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw42; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 8192 --kv q8 > $O/q8.txt 2>&1; echo "q8 rc=$?"
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 8192 --kv-hot 4096 > $O/hot.txt 2>&1; echo "hot rc=$?"
echo done > $O/DONE
