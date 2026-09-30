#!/bin/bash
# sw104: the expert cache's warm-up, in the engine. sw99's simulator, extended: the counts the
# policy starts from are the prompt's routing counts, so for the first ~70 tokens an expert the
# answer needs cannot beat the weakest resident (count 56 on window 9). Starting at 3% of them
# (seed scale 0.03): window 9 78.0 -> 80.6% hits; with an upload budget of 128 for the first 64
# tokens as well: 84.1% (first 64 tokens 48 -> 73%); wikitext unchanged or +0.5.
# As sw100: fr_bench, window 9's 32K prompt and 32K of wikitext, each followed by the model's own
# greedy generation, decoded teacher-forced (320 tokens), plain decode, 2 runs per arm.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw104; mkdir -p $O; T=$BENCH/sw100
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
for r in 1 2; do
  for arm in "base:" "seed:--cache-seed-scale 0.03" "early:--cache-seed-scale 0.03 --cache-early-budget 128 --cache-early-tokens 64" \
             "b64:--cache-seed-scale 0.03 --swap-budget 64"; do
    name=${arm%%:*}; args=${arm#*:}
    run w9-$name-r$r --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 $args
    run wiki-$name-r$r --ids $T/wiki_teacher.ids --n-prompt 32768 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 $args
  done
done
echo done > $O/DONE
