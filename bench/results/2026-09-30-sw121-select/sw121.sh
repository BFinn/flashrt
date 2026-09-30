#!/bin/bash
# sw121: P-5's first kernel, the QSA block selection on an 8-CTA cluster per token (127693c..422f6fd).
# Its output must equal the single-CTA kernel's: sw112's fingerprint on the build before it
# (fbcf51a, exported and built here) and on the current one. Then teacher-forced A/B, old and new
# interleaved, 3 pairs: 245K plain and --spec 2, 32K plain, from the saved states.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
O=$BENCH/sw121; mkdir -p $O; cd $FLASHRT
if [ ! -x $O/base/build/fr_bench ]; then
  rm -rf $O/base && mkdir -p $O/base && git -C $FLASHRT archive fbcf51a | tar -x -C $O/base
  cmake -S $O/base -B $O/base/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DFLASHRT_NATIVE=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc > $O/base-build.log 2>&1
  cmake --build $O/base/build --target fr_bench fr_kld >> $O/base-build.log 2>&1 || { echo "base build failed"; exit 1; }
fi
cmake --build build 2>&1 | grep -E 'error|warning|FAILED' | head
(cd build && FLASHRT_TEST_MODEL=$M ctest --output-on-failure 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
FR_BUILD=$O/base/build bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw121-base > $O/fp-base.txt 2>&1
bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw121-new > $O/fp-new.txt 2>&1
diff $O/fp-base.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
ab() { local arm=$1 name=$2; shift 2
  local B=$FLASHRT/build/fr_bench; [ $arm = old ] && B=$O/base/build/fr_bench
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
echo ALLDONE
