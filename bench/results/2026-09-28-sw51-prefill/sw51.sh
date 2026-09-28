#!/bin/bash
# sw51: own expert grouping (instead of ggml's mm_ids_helper) and the fused mid-layer hc combine +
# norm, both meant bit-identical: speed and greedy tokens at 32K and 64K against sw50 (q8 KV,
# chunks of 8,192), then an nsys profile at 64K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw51; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n32k.txt 2>&1
wait_vram; $B/fr_bench $M --ids $I --n-prompt 65536 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n64k.txt 2>&1
wait_vram; $N profile -f true -o $O/p64k --trace=cuda $B/fr_bench $M --ids $I --n-prompt 65536 --gen 1 --prefill-chunk 8192 --kv q8 > $O/p64k.txt 2>&1
$N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p64k $O/p64k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
