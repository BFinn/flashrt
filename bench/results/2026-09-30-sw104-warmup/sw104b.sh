#!/bin/bash
# sw104b: more runs of sw104's arms (the wikitext column was noisy), a gentler early budget (64 for
# 64 tokens), and the draft head's case (--spec 2, argmax drafts under teacher forcing) on window 9,
# where the hit rates are lowest. 3 runs per arm, rotating.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw104; mkdir -p $O; T=$BENCH/sw100
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
ARMS=("base:" "seed:--cache-seed-scale 0.03" "early:--cache-seed-scale 0.03 --cache-early-budget 128 --cache-early-tokens 64"
      "early64:--cache-seed-scale 0.03 --cache-early-budget 64 --cache-early-tokens 64" "b64:--cache-seed-scale 0.03 --swap-budget 64")
for r in 3 4 5; do
  for arm in "${ARMS[@]}"; do
    name=${arm%%:*}; args=${arm#*:}
    run w9-$name-r$r --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 $args
    run wiki-$name-r$r --ids $T/wiki_teacher.ids --n-prompt 32768 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 $args
    run w9mtp-$name-r$r --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 \
      --mtp $D --spec 2 --draft-vocab $V $args
  done
  ARMS=("${ARMS[@]:1}" "${ARMS[0]}")   # rotate the order each round
done
echo done > $O/DONE-b
