#!/bin/bash
# sw24: first speculative decode (greedy, MTP drafts), 2K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw24; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for k in 2 3; do wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --spec $k > $O/spec${k}_2k.txt 2>&1; echo "spec $k rc=$?"; done
echo done > $O/DONE
