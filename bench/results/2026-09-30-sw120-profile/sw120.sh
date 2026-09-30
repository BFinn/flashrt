#!/bin/bash
# sw120: decode kernel profile for P-5 (nsys --cuda-graph-trace=node), plain and --spec 2 at 32K and
# 245K from the saved states (q8 host KV, hot set 4096), teacher-forced, one window of 128 tokens.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw120; mkdir -p $O
N=/usr/local/cuda-12.9/bin/nsys
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for ctx in 32 245; do
  S=$BENCH/state-${ctx}k-q8-mtp.bin; NP=$([ $ctx = 32 ] && echo 32768 || echo 245760)
  for a in plain spec2; do
    X=""; [ $a = spec2 ] && X="--mtp $D --spec 2 --draft-vocab $V"
    wait_vram; $N profile -f true -o $O/p_${a}_$ctx --trace=cuda --cuda-graph-trace=node $B/fr_bench $M --ids $I --n-prompt $NP --gen 128 --windows 1 \
      --kv-hot 4096 --load-state $S $X --teacher > $O/p_${a}_$ctx.txt 2>&1
    $N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p_${a}_$ctx $O/p_${a}_$ctx.nsys-rep > /dev/null 2>&1
    echo "$a $ctx: $(grep -h '^decode:' $O/p_${a}_$ctx.txt | cut -c1-90)"
  done
done
echo ALLDONE
