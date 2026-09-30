#!/bin/bash
# sw107c: the cost of the deterministic commit, measured in one session. sw107b against sw104 (another
# session): window 9 with the head 102.2 against 108.5 tok/s at the same hit rate and swaps. Here the
# timing-dependent build (1c6fc82, in sw94's separate clone) and the current one alternate: fr_bench
# teacher-forced on window 9 with the head (--spec 2), 3 runs each.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
O=$BENCH/sw107c; mkdir -p $O; T=$BENCH/sw100; C=$BENCH/sw94/src
(cd $C && git fetch -q $FLASHRT main && git checkout -q 1c6fc82 && cmake --build build --target fr_bench > $O/build-old.log 2>&1) || { echo "old build failed"; exit 1; }
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1 bin=$2; shift 2; wait_vram; timeout 2400 $bin $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
A="--ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 --mtp $D --spec 2 --draft-vocab $V"
for r in 1 2 3; do
  run query-r$r $C/build/fr_bench $A
  run det-r$r $FLASHRT/build/fr_bench $A
done
echo done > $O/DONE
