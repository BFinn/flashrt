#!/bin/bash
# sw88: why flashrt decodes slower on window 9's prompts (sw87) than in fr_bench: fr_bench, fresh
# prefill (automatic chunks), q8 host KV + hot set 4,096, 384 greedy tokens, plain decode, on
# (a) window 9's 32K prompt (synthetic filler + an instruction; strata-ids.json) and (b) the 32K
# wikitext prompt the fr_bench runs used; then (a) with --spec 2
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw88; mkdir -p $O
python3 -c "import json; d=json.load(open('$BENCH/strata-ids.json')); open('$O/w9_32k.ids','w').write(' '.join(map(str, d[1]['ids'])))"
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
run w9_plain --ids $O/w9_32k.ids --n-prompt 32793 --gen 384 --prefill-chunk auto --kv q8 --kv-hot 4096
run wiki_plain --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 32793 --gen 384 --prefill-chunk auto --kv q8 --kv-hot 4096
run w9_spec2 --ids $O/w9_32k.ids --n-prompt 32793 --gen 384 --prefill-chunk auto --kv q8 --kv-hot 4096 --mtp $D --spec 2 --draft-vocab $V
echo done > $O/DONE
