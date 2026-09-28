#!/bin/bash
# sw60: GDN column kernel with 4 accumulators per sum, 4 or 8 lanes per state column
# (FLASHRT_GDN_LPC): 32K prefill (q8 KV, automatic chunks) under nsys, one run each
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw60; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for l in 4 8; do
  wait_vram; FLASHRT_GDN_LPC=$l $N profile -f true -o $O/lpc$l --trace=cuda $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/lpc$l.txt 2>&1
  $N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/lpc$l $O/lpc$l.nsys-rep > /dev/null 2>&1
done
echo done > $O/DONE
