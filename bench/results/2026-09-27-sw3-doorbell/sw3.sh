#!/bin/bash
# speed work 3: doorbells. fr_bench at 2K: 3 runs doorbell, 2 runs host sync per layer.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw3; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for r in 1 2 3; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 > $O/db_2k_r$r.txt 2>&1; echo "db r$r rc=$?"; done
for r in 1 2; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 --no-doorbell > $O/sync_2k_r$r.txt 2>&1; echo "sync r$r rc=$?"; done
echo done > $O/DONE
