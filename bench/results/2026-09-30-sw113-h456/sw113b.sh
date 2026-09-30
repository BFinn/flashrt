#!/bin/bash
# sw113b: 3 more interleaved pairs of the head arm (sw113's first 3 were -2.5% on one slow run).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
T=$BENCH/sw100
O=$BENCH/sw113
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
C="--prefill-chunk auto --kv q8 --kv-hot 4096"
for arm in new old old new new old; do
  B=$FLASHRT/build/fr_bench; [ $arm = old ] && B=$O/fr_bench.old
  wait_vram; $B $M --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C --mtp $D --spec 2 --draft-vocab $V > $O/w9mtp-$arm-$(date +%s).txt 2>&1
  echo "$arm w9mtp $(grep -h '^decode:' $(ls -t $O/w9mtp-$arm-*.txt | head -1) | tail -1)"
done
echo ALLDONE
