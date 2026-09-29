#!/bin/bash
# sw98: P-1's first step. The MTP head's pass over a chunked prompt attended through the KV hot
# set (the per-token kernel), a cost that grows with depth; with a VRAM mirror of its KV during
# the prefill it takes the tensor-core attention, as the target's chunks do. A/B with
# FLASHRT_MTP_MIRROR=0/1, alternating, 2 runs each: a cold prompt of 131,072 wikitext tokens with
# the head (--spec 2), then 256 generated tokens at temperature 1.0 for the drafts' acceptance
# (the head's arithmetic changes with the attention kernel).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw98; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for arm in 1 0 1 0; do
  wait_vram
  FLASHRT_MTP_MIRROR=$arm python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS --n 131072 --gen 256 --temp 1.0 -- \
    --ctx 140000 --mtp $D --spec 2 --draft-vocab $V > $O/mirror$arm-$(date +%s).txt 2>&1
  echo "mirror $arm rc=$? $(grep -h '^r1:' $O/mirror$arm-*.txt | tail -1 | cut -c1-120)"
done
echo done > $O/DONE
