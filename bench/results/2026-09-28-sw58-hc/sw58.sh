#!/bin/bash
# sw58: hc prefill with the norm writing 1/rms (not xn) and the inject product in the norm's block:
# prefill at 32K (q8 KV, automatic chunks), a 64K nsys profile, then the KLD gate
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw58; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/n32k.txt 2>&1; echo "32k rc=$?"
wait_vram; $N profile -f true -o $O/p64k --trace=cuda $B/fr_bench $M --ids $I --n-prompt 65536 --gen 1 --prefill-chunk auto --kv q8 > $O/p64k.txt 2>&1
$N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p64k $O/p64k.nsys-rep > /dev/null 2>&1
cd $BENCH/kld
run() { name=$1; shift; wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run chunk1024-f16 --prefill-chunk 1024
run chunk1024-q8 --prefill-chunk 1024 --kv q8
echo done > $O/DONE
