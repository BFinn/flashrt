#!/bin/bash
# sw41: nsys of chunked prefill at 64K: q8 KV in VRAM vs host KV + hot set
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw41; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
for mode in q8 hot; do
  if [ $mode = q8 ]; then A="--kv q8"; else A="--kv-hot 4096"; fi
  $N profile -f true -o $O/$mode --trace=cuda $B/fr_bench $M --ids $I --n-prompt 65536 --gen 1 --prefill-chunk 8192 $A > $O/$mode.txt 2>&1
  $N stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o $O/$mode $O/$mode.nsys-rep > /dev/null 2>&1
done
echo done > $O/DONE
