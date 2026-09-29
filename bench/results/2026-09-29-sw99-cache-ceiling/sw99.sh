#!/bin/bash
# sw99: how far can the expert cache go on window 9's generation? Routing traces from llama.cpp
# (route_trace; experts on the CPU, greedy), the prompt's and 384 generated tokens', for window
# 9's 32K prompt and for 32K of wikitext; then tools/cache_sim.py at the engine's cache sizes:
# the engine's policy (primed from the prompt, decayed LFU, swap budget 8 / 32 / 128) against
# the ceilings (static: the generation's own top pairs; belady: the optimum).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
O=$BENCH/sw99; mkdir -p $O; cd $O
python3 - <<PY
import json
d = json.load(open("$BENCH/strata-ids.json"))
item = [e for e in d if e["depth"] == 32000][0]
open("$O/w9_32k.txt", "w").write(" ".join(map(str, item["ids"])))
PY
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram
$FLASHRT/build/route_trace --model $M --ids $O/w9_32k.txt --n-prompt 32793 --gen 384 --prefill-trace --ctx 34000 --out $O/w9 > $O/trace-w9.log 2>&1; echo "trace w9 rc=$?"
wait_vram
$FLASHRT/build/route_trace --model $M --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 32768 --gen 384 --prefill-trace --ctx 34000 --out $O/wiki > $O/trace-wiki.log 2>&1; echo "trace wiki rc=$?"
for p in w9 wiki; do
  for b in 8 32 128; do
    python3 $FLASHRT/tools/cache_sim.py --trace $O/$p.decode_topk.npy --prime $O/$p.prefill_topk.npy --budget $b --windows 64 \
      --slots 7780,8634 --policies $([ $b = 32 ] && echo engine,static,belady || echo engine) > $O/sim-$p-b$b.txt 2>&1
    echo "== $p budget $b"; cat $O/sim-$p-b$b.txt
  done
done
echo done > $O/DONE
