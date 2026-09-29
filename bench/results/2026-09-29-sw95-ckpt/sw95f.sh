#!/bin/bash
# sw95f: engine_smoke --reuse with the cancelled case's restored run chunked like its cold
# reference (sw95e: 96 new tokens ran as batches and flipped a near-tie). Every case must be
# bit-exact.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw95; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
smoke reuse-plain-f "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke reuse-mtp-f "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
echo done > $O/DONE-f
