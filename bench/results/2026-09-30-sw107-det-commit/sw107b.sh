#!/bin/bash
# sw107b: as sw107, with the budget per step (sw107: counted in flight, it halved the upload rate once each upload stays pending two steps; with the head 108.5 -> 96.9 tok/s). sw107 opened: uploads commit two steps after issue, not when a query finds them done (sw106:
# with 64 in flight that made the cache's content, and so the fast-path KLD, vary between runs).
# (1) The KLD, each configuration twice: the runs must repeat exactly. (2) Speed as sw104's
# (teacher-forced, fr_bench), 3 runs, against sw104's budget-64 arm with the timing-dependent commit.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw107b; mkdir -p $O; T=$BENCH/sw100
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
k() { local name=$1; shift; wait_vram
      (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast "$@" > $O/kld-$name.log 2>&1)
      echo "$name $(grep -h 'KLD mean' $O/kld-$name.log | awk '{print $3, $5, $7}') $(grep -hoE 'hit rate [0-9.]+%' $O/kld-$name.log) $(grep -hoE '[0-9]+ swaps' $O/kld-$name.log)"; }
k fast-a; k fast-b
k win3-a --window 3 --prefill-chunk 2048 --kv-hot 512; k win3-b --window 3 --prefill-chunk 2048 --kv-hot 512
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
for r in 1 2 3; do
  run w9-r$r --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096
  run wiki-r$r --ids $T/wiki_teacher.ids --n-prompt 32768 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096
  run w9mtp-r$r --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 --mtp $D --spec 2 --draft-vocab $V
done
echo done > $O/DONE
