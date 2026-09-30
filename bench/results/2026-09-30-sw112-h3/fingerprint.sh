#!/bin/bash
# sw112: a build's output fingerprint, for behaviour-neutral refactors (phase 4, H-3). Runs are
# deterministic (sw107, sw109), so a refactor that changes nothing must reproduce every value here
# exactly. Usage: fingerprint.sh LABEL (writes $BENCH/sw112/LABEL/, then LABEL.txt).
set -u
L=${1:?label}
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw112/$L; mkdir -p $O; T=$BENCH/sw100
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
k() { local name=$1; shift; wait_vram
      (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 "$@" > $O/kld-$name.log 2>&1)
      echo "kld $name $(grep -h 'KLD mean' $O/kld-$name.log | awk '{print $3, $5, $7, $9, $11}') $(grep -h 'same top' $O/kld-$name.log | awk '{print $4}') $(grep -hoE '[0-9]+ swaps' $O/kld-$name.log)"; }
k fast --fast
k win3 --fast --window 3 --prefill-chunk 2048 --kv-hot 512
k chunk --prefill-chunk 1024
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "bench $name $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-80) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1) $(grep -hE 'accepted|drafts' $O/$name.txt | tail -1 | cut -c1-80) $(grep -h '^tokens:' $O/$name.txt | md5sum | cut -c1-12)"; }
C="--prefill-chunk auto --kv q8 --kv-hot 4096"
run w9-teacher --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C
run w9mtp-teacher --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C --mtp $D --spec 2 --draft-vocab $V
run wiki-sampled --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 32768 --gen 256 $C --mtp $D --spec 2 --draft-vocab $V --temp 1.0 --top-k 20 --top-p 0.95 --seed 7
run wiki-greedy-plain --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 8192 --gen 128 $C
