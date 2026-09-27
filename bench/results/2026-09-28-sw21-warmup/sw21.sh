#!/bin/bash
# KV compression 3 / cache warm-up: 2K swap budget 8 vs 32 (2 runs each, 512 tokens), then a fresh 245,760 prefill
# with decayed counts (half-life 4096), q8 + hot set 4096, budget 32, 6 windows, state saved (q8 format).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw21; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for r in 1 2; do
  for bgt in 8 32; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 512 --swap-budget $bgt > $O/b${bgt}_2k_r$r.txt 2>&1; echo "2k b$bgt r$r rc=$?"; done
done
wait_vram; timeout 7200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --swap-budget 32 --save-state $BENCH/state-245k-q8.bin > $O/hot_b32_245k_fresh.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
