#!/bin/bash
# sw29: fused hc kernels for windows; confidence-gated drafting; 2K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw29; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 > $O/plain.txt 2>&1; echo "plain rc=$?"
for k in 1 2 3; do wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --spec $k --draft-vocab $V > $O/spec${k}.txt 2>&1; echo "spec $k rc=$?"; done
for pm in 0.3 0.6; do wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --spec 3 --draft-vocab $V --draft-pmin $pm > $O/spec3_pmin$pm.txt 2>&1; echo "spec 3 pmin $pm rc=$?"; done
echo done > $O/DONE
