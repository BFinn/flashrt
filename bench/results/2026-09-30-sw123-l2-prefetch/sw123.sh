#!/bin/bash
# sw123: L2 prefetch of the next layer's hc mix weights while the MoE combine waits for the CPU
# misses (FLASHRT_L2_PREFETCH 0 off, 1 the attention mix, 2 the FFN mix too). Same binary, modes
# rotated, 3 rounds: teacher-forced from the saved states, 256 tokens; then nsys at 32K plain,
# modes 0 and 1, for the hc kernels' time; then the fingerprint (a hint cannot change outputs).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw123; mkdir -p $O; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local mode=$1 name=$2; shift 2
  wait_vram; FLASHRT_L2_PREFETCH=$mode $B/fr_bench $M "$@" > $O/$name-m$mode-$(date +%s).txt 2>&1
  echo "m$mode $name $(grep -h '^decode:' $(ls -t $O/$name-m$mode-*.txt | head -1) | tail -1 | cut -c1-80)"; }
S245="--ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --teacher --gen 256 --windows 1"
S32="--ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --teacher --gen 256 --windows 1"
for order in "0 1 2" "1 2 0" "2 0 1"; do
  for m in $order; do
    run $m plain245 $S245
    run $m plain32 $S32
    run $m spec32 $S32 --mtp $D --spec 2 --draft-vocab $V
  done
done
N=/usr/local/cuda-12.9/bin/nsys
for m in 0 1; do
  wait_vram; FLASHRT_L2_PREFETCH=$m $N profile -f true -o $O/p_m$m --trace=cuda --cuda-graph-trace=node $B/fr_bench $M $S32 > $O/p_m$m.txt 2>&1
  $N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p_m$m $O/p_m$m.nsys-rep > /dev/null 2>&1
done
bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw123 > $O/fp-new.txt 2>&1
diff $BENCH/sw122/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
echo ALLDONE
