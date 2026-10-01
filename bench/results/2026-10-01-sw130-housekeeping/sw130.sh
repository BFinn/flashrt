#!/bin/bash
# sw130: housekeeping. (1) The P-5 toggles and the one-CTA argmax removed: the fingerprint must
# equal sw127's. (2) Why fr_bench and the engine differ on window 9's 32K prompt with --spec 2
# (sw129: 97.4 tok/s at 75.4% hits against 111.4 at 81.7%): both tools on that prompt alone,
# fresh, greedy, plain and --spec 2, 256 and 384 tokens.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$FLASHRT/bench/reference/mtp-vocab-ranks.txt
B=$FLASHRT/build; O=$BENCH/sw130; mkdir -p $O; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
if [ -z "${SKIP_FP:-}" ]; then
  bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw130 > $O/fp-new.txt 2>&1
  diff $BENCH/sw127/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
fi
python3 - bench/reference/w9-ids.json $O <<'PY'
import json, sys
item = next(i for i in json.load(open(sys.argv[1])) if 30000 < len(i["ids"]) < 40000)
open(sys.argv[2] + "/w9-32k.ids", "w").write(" ".join(map(str, item["ids"])))
json.dump([item], open(sys.argv[2] + "/w9-32k.json", "w"))
PY
n=$(wc -w < $O/w9-32k.ids)
for gen in 256 384; do
  for arm in plain spec2; do
    X=(); [ $arm = spec2 ] && X=(--mtp $D --spec 2 --draft-vocab $V)
    wait_vram
    $B/fr_bench $M --ids $O/w9-32k.ids --n-prompt $n --gen $gen --prefill-chunk auto --kv q8 --kv-hot 4096 "${X[@]}" > $O/fb-$arm-$gen.txt 2>&1
    echo "fr_bench $arm $gen: $(grep -h '^decode:' $O/fb-$arm-$gen.txt | cut -c1-70) | $(grep -h 'hit rate' $O/fb-$arm-$gen.txt | cut -c1-40)"
    wait_vram
    python3 bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $O/w9-32k.json --gen $gen --label eng-$arm-$gen \
      --log $O/eng-$arm-$gen.log -- "${X[@]}" > $O/eng-$arm-$gen.txt 2>&1
    echo "engine   $arm $gen: $(grep -h 'depth' $O/eng-$arm-$gen.txt | head -1 | cut -c30-200)"
  done
done
echo ALLDONE
