#!/bin/bash
# speed work 12: crash hunt (6 short static + adaptive runs with diagnostics), then 32K adaptive, 3 windows.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw12; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for r in 1 2 3; do
  wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --static-cache > $O/static_2k_r$r.txt 2>&1; echo "static r$r rc=$?"
  wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 > $O/adapt_2k_r$r.txt 2>&1; echo "adapt r$r rc=$?"
done
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 3 > $O/adapt_32k.txt 2>&1; echo "32k rc=$?"
echo done > $O/DONE
