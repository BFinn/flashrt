#!/bin/bash
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw20; N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; exit 1; }
cd $O
for arm in "hot2 --kv-hot 4096"; do
  set -- $arm; name=$1; shift
  wait_vram
  $N profile -o prof_$name --force-overwrite true --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace=cuda \
    $B/fr_bench $M --ids $I --n-prompt 245760 --gen 64 --load-state $BENCH/state-245k.bin "$@" > prof_$name.txt 2>&1
  $N stats -q --report cuda_gpu_kern_sum --format csv -o prof_$name prof_$name.nsys-rep > /dev/null 2>&1
done
echo done > $O/PROF_DONE
