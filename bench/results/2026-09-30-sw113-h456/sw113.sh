#!/bin/bash
# sw113: phase 4 step 4 (H-4 named constants, H-5 doorbell flags as system-scope atomics, H-6
# build flags) and E-4 (the engine joins its reader on quit). Run from the checkout with the new
# commit pulled but not yet built: the current build's fr_bench is kept as the old arm first.
# (1) configure-time CUDA architecture check; (2) build, ctest; (3) sw112's fingerprint, which
# must equal sw112's "before"; (4) H-5 changes the decode's synchronization, so a teacher-forced
# A/B, old and new interleaved, 3 pairs each; (5) engine_smoke and server_smoke.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
T=$BENCH/sw100
O=$BENCH/sw113; mkdir -p $O; cd $FLASHRT
cp build/fr_bench $O/fr_bench.old
for a in 86 native; do
  rm -rf $O/cfg-$a
  cmake -S . -B $O/cfg-$a -G Ninja -DCMAKE_CUDA_ARCHITECTURES=$a -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc > $O/cfg-$a.log 2>&1
  echo "configure sm $a: rc=$? $(grep -hE 'needs sm_90|cannot be checked' $O/cfg-$a.log | head -1)"
  rm -rf $O/cfg-$a
done
cmake --build build 2>&1 | grep -E 'error|warning|FAILED' | head -20
(cd build && FLASHRT_TEST_MODEL=$M ctest --output-on-failure 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
bash bench/results/2026-09-30-sw112-h3/fingerprint.sh h456 > $BENCH/sw113-fingerprint.txt 2>&1
diff $BENCH/sw112-before.txt $BENCH/sw113-fingerprint.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
C="--prefill-chunk auto --kv q8 --kv-hot 4096"
ab() {   # arm (old|new), name, args...
  local arm=$1 name=$2; shift 2
  local B=$FLASHRT/build/fr_bench; [ $arm = old ] && B=$O/fr_bench.old
  wait_vram; $B $M "$@" > $O/$name-$arm-$(date +%s).txt 2>&1
  echo "$arm $name $(grep -h '^decode:' $(ls -t $O/$name-$arm-*.txt | head -1) | tail -1)"
}
for r in 1 2 3; do
  for arm in old new; do
    [ $r = 2 ] && arm=$([ $arm = old ] && echo new || echo old)
    ab $arm w9 --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C
    ab $arm w9mtp --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C --mtp $D --spec 2 --draft-vocab $V
    ab $arm wiki8k --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 8192 --gen 256 --teacher $C
  done
done
mv $BENCH/sw112/smoke $BENCH/sw112/smoke-h2 2>/dev/null
bash bench/results/2026-09-30-sw112-h3/smoke.sh
mv $BENCH/sw112/smoke $O/smoke
echo ALLDONE
