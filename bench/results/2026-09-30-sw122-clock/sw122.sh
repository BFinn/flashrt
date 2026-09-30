#!/bin/bash
# sw122: P-5's second kernel, the hot set's CLOCK sweep in parallel (k_hot_select). Where a block
# sits never changes a value, so the outputs must equal the build before (sw121's fp-new.txt, 422f6fd);
# the current fr_bench is kept as the old arm, then teacher-forced A/B as sw121, and an nsys profile
# of 245K plain for the kernel's time.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
O=$BENCH/sw122; mkdir -p $O; cd $FLASHRT
[ -e $O/fr_bench.old ] || cp build/fr_bench $O/fr_bench.old
cmake --build build 2>&1 | grep -E 'error|warning|FAILED' | head
(cd build && FLASHRT_TEST_MODEL=$M ctest --output-on-failure 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw122 > $O/fp-new.txt 2>&1
diff $BENCH/sw121/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
ab() { local arm=$1 name=$2; shift 2
  local B=$FLASHRT/build/fr_bench; [ $arm = old ] && B=$O/fr_bench.old
  wait_vram; $B $M "$@" > $O/$name-$arm-$(date +%s).txt 2>&1
  echo "$arm $name $(grep -h '^decode:' $(ls -t $O/$name-$arm-*.txt | head -1) | tail -1 | cut -c1-80)"; }
S245="--ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --teacher --gen 256 --windows 1"
S32="--ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --teacher --gen 256 --windows 1"
for r in 1 2 3; do
  for arm in old new; do
    [ $r = 2 ] && arm=$([ $arm = old ] && echo new || echo old)
    ab $arm plain245 $S245
    ab $arm spec245 $S245 --mtp $D --spec 2 --draft-vocab $V
    ab $arm plain32 $S32
  done
done
N=/usr/local/cuda-12.9/bin/nsys
wait_vram; $N profile -f true -o $O/p_plain_245 --trace=cuda --cuda-graph-trace=node build/fr_bench $M --ids $I --n-prompt 245760 --gen 128 \
  --windows 1 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --teacher > $O/p_plain_245.txt 2>&1
$N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p_plain_245 $O/p_plain_245.nsys-rep > /dev/null 2>&1
echo ALLDONE
