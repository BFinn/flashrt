#!/bin/bash
# sw28: nsys kernel profile of speculative decode (k = 2) at 2K, and of plain decode
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw28; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $N profile -f true -o $O/spec2 --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace=cuda $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 --mtp $D --spec 2 --draft-vocab $V > $O/spec2.txt 2>&1; echo "spec2 rc=$?"
$N stats --report cuda_gpu_kern_sum --format csv -o $O/spec2 $O/spec2.nsys-rep > /dev/null 2>&1
wait_vram; $N profile -f true -o $O/plain --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace=cuda $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 > $O/plain.txt 2>&1; echo "plain rc=$?"
$N stats --report cuda_gpu_kern_sum --format csv -o $O/plain $O/plain.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
