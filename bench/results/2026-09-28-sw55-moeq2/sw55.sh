#!/bin/bash
# sw55: moe_q2 (experts from the planar arena layout on int8 tensor cores) against the MMQ path
# (FLASHRT_MOE_Q2MMA=0): prefill at 32K and 245K (q8 KV, automatic chunks), one run each, and an
# nsys profile at 64K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw55; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for q in 0 1; do
  wait_vram; FLASHRT_MOE_Q2MMA=$q $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/n32k_q$q.txt 2>&1; echo "32k $q rc=$?"
done
for q in 1 0; do
  wait_vram; FLASHRT_MOE_Q2MMA=$q timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk auto --kv q8 > $O/n245k_q$q.txt 2>&1; echo "245k $q rc=$?"
done
wait_vram; $N profile -f true -o $O/p64k --trace=cuda $B/fr_bench $M --ids $I --n-prompt 65536 --gen 1 --prefill-chunk auto --kv q8 > $O/p64k.txt 2>&1
$N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p64k $O/p64k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
