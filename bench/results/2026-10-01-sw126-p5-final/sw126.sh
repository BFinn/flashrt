#!/bin/bash
# sw126: P-5's last items. The argmax on an 8-CTA cluster (325dea3, FLASHRT_ARGMAX_CLUSTER) and the
# indexer's pooled keys in fp16 (d8a6e52). The base build (325dea3) is fingerprinted against
# sw122's (the argmax must change nothing); the new build's fingerprint carries the KLD gate for
# the fp16 keys. Then three arms interleaved, teacher-forced from the saved states, 256 tokens:
# old (base, FLASHRT_ARGMAX_CLUSTER=0), argmax (base), new (this build); then nsys of new at 245K.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
O=$BENCH/sw126; mkdir -p $O/ab; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
if [ ! -x $O/base/build/fr_bench ]; then
  rm -rf $O/base && mkdir -p $O/base && git -C $FLASHRT archive 325dea3 | tar -x -C $O/base
  cmake -S $O/base -B $O/base/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DFLASHRT_NATIVE=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc > $O/base-build.log 2>&1
  cmake --build $O/base/build --target fr_bench fr_kld >> $O/base-build.log 2>&1 || { echo "base build failed"; exit 1; }
fi
if [ -z "${SKIP_FP:-}" ]; then
  FR_BUILD=$O/base/build bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw126-base > $O/fp-base.txt 2>&1
  diff $BENCH/sw122/fp-new.txt $O/fp-base.txt && echo "ARGMAX FINGERPRINT-IDENTICAL" || echo "ARGMAX FINGERPRINT-DIFFERS"
  bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw126-new > $O/fp-new.txt 2>&1
  echo "fp16 keys, against sw122:"; diff $BENCH/sw122/fp-new.txt $O/fp-new.txt
fi
run() { local arm=$1 name=$2; shift 2
  local B=$FLASHRT/build/fr_bench A=1
  [ $arm != new ] && B=$O/base/build/fr_bench
  [ $arm = old ] && A=0
  wait_vram; FLASHRT_ARGMAX_CLUSTER=$A $B $M "$@" > $O/ab/$name-$arm-$(date +%s).txt 2>&1
  local f=$(ls -t $O/ab/$name-$arm-*.txt | head -1)
  echo "$arm $name $(grep -h '^decode:' $f | tail -1 | cut -c1-80) | $(grep -hoE 'expert cache: [0-9]+ slots' $f | head -1)"; }
S245="--ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --teacher --gen 256 --windows 1"
S32="--ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --teacher --gen 256 --windows 1"
for order in ${ORDERS:-old,arg,new arg,new,old new,old,arg old,arg,new}; do
  for a in ${order//,/ }; do
    run $a plain245 $S245
    run $a plain32 $S32
    run $a spec245 $S245 --mtp $D --spec 2 --draft-vocab $V
    run $a spec32 $S32 --mtp $D --spec 2 --draft-vocab $V
  done
done
N=/usr/local/cuda-12.9/bin/nsys
for a in old new; do
  B=$FLASHRT/build/fr_bench; A=1; [ $a = old ] && { B=$O/base/build/fr_bench; A=0; }
  wait_vram; FLASHRT_ARGMAX_CLUSTER=$A $N profile -f true -o $O/p_$a --trace=cuda --cuda-graph-trace=node $B $M $S245 > $O/p_$a.txt 2>&1
  $N stats --force-export=true --report cuda_gpu_kern_sum --format csv -o $O/p_$a $O/p_$a.nsys-rep > /dev/null 2>&1
done
echo ALLDONE
