#!/bin/bash
# sw38: nsys profile of chunked prefill (16K, chunks of 4096)
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw38; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
$N profile -f true -o $O/pf16k --trace=cuda $B/fr_bench $M --ids $I --n-prompt 16384 --gen 1 --prefill-chunk 4096 > $O/pf16k.txt 2>&1; echo "rc=$?"
rm -f $O/pf16k_*.csv; $N stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o $O/pf16k $O/pf16k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
