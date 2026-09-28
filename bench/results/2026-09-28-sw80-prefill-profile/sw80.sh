#!/bin/bash
# sw80: 64K prefill kernel profile after sw69-sw72 (q8 KV, automatic chunks), for the next prefill work
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw80; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $N profile -f true -o $O/p64k --trace=cuda $B/fr_bench $M --ids $I --n-prompt 65536 --gen 1 --prefill-chunk auto --kv q8 > $O/p64k.txt 2>&1
$N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p64k $O/p64k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
