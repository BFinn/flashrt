#!/bin/bash
# sw64: expert-cache capacity at decode: VRAM reserve 1024 (default) against 256 MiB (about +600
# slots), --spec 1 at P2 conditions (temperature 1.0, top-k 20, top-p 0.95; host KV + hot set
# 4096), from the saved states; 6 windows of 128 tokens each, --teacher (the ids file's own
# continuation: the same routing in every arm)
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw64; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
for r in 1024 256; do
  run spec1_245k_r$r --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --reserve-mib $r --teacher
  run spec1_32k_r$r --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --reserve-mib $r --teacher
done
echo done > $O/DONE
