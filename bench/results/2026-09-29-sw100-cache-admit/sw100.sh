#!/bin/bash
# sw100: P-2. The adaptive expert cache's admission, A/B in fr_bench. sw99's simulation of the
# engine's policy on window 9's generation: admit 2 / margin 1.5 (the default) 73.0% hits; admit
# 1.5: 75.5%; admit 1 / margin 1.2: 78.0% (and wikitext 87.9 -> 89.3%), at twice the uploads.
# Each prompt is followed by the model's own greedy generation (sw99's route_trace tokens), which
# every arm decodes teacher-forced, so every arm routes the same tokens: window 9's 32K prompt
# and 32K of wikitext, a fresh prefill each, 384 tokens, plain decode (no MTP), 2 runs per arm.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw100; mkdir -p $O; T=$BENCH/sw99
python3 - <<PY
import json
d = json.load(open("$BENCH/strata-ids.json"))
w9 = [e for e in d if e["depth"] == 32000][0]["ids"]
gen = open("$T/w9.tokens.txt").read().split()
open("$O/w9_teacher.ids", "w").write(" ".join(map(str, w9)) + " " + " ".join(gen))
wiki = open("$BENCH/p0c-20260927/wiki.prompt_ids.txt").read().split()[:32768]
gen = open("$T/wiki.tokens.txt").read().split()
open("$O/wiki_teacher.ids", "w").write(" ".join(wiki) + " " + " ".join(gen))
PY
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-60)"; }
for r in 1 2; do
  for arm in "2 1.5 32" "1.5 1.5 32" "1 1.2 32" "1 1.2 64"; do
    set -- $arm
    run w9-a$1-m$2-b$3-r$r --ids $O/w9_teacher.ids --n-prompt 32793 --gen 384 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 \
      --cache-admit $1 --cache-margin $2 --swap-budget $3
    run wiki-a$1-m$2-b$3-r$r --ids $O/wiki_teacher.ids --n-prompt 32768 --gen 384 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 \
      --cache-admit $1 --cache-margin $2 --swap-budget $3
  done
done
echo done > $O/DONE
