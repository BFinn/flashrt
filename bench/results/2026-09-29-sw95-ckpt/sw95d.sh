#!/bin/bash
# sw95d: sw95c again, with the prompt's last token of each check on the reference path under the
# test hook (no expert-cache noise: sw95c's aligned "cancelled" case still differed by KL 0.00094)
# and every restore on a 64-token boundary (the chunked GDN's sub-chunks then match a cold run's).
# The --faults checks too, as they use the same first-token logits.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw95; mkdir -p $O
cmake --build $B > $O/build-d.log 2>&1 || { echo "build failed"; exit 1; }
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
smoke reuse-plain-d "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke reuse-mtp-d "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
smoke faults-plain-d "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512
smoke faults-mtp-d "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
echo done > $O/DONE-d
