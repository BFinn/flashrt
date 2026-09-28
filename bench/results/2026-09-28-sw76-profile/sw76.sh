#!/bin/bash
# sw76: decode kernel profile after sw71-sw75 (nsys --cuda-graph-trace=node), plain and --spec 1 at 32K,
# P2 conditions from the saved state, teacher-forced, one window of 128 tokens each
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
S32=$BENCH/state-32k-q8-mtp.bin
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw76; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for a in plain spec1; do
  X=""; [ $a = spec1 ] && X="--mtp $D --spec 1 --draft-vocab $V"
  wait_vram; $N profile -f true -o $O/p_$a --trace=cuda --cuda-graph-trace=node $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 1 --kv-hot 4096 --load-state $S32 $X $T1 --teacher > $O/p_$a.txt 2>&1
  $N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p_$a $O/p_$a.nsys-rep > /dev/null 2>&1
done
echo done > $O/DONE
