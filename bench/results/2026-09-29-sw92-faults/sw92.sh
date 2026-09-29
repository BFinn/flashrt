#!/bin/bash
# sw92: phase 1 of docs/improvement-plan.md. engine_smoke.py --faults against the real model:
# invalid requests change nothing, and injected faults (after a prefill chunk, in a decode step
# or verify window) and a stop during the prefill leave the engine serving. Two arms: with the
# MTP head (--spec 2: the fault lands in a verify window) and without (a plain decode step).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw92; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() {   # label, engine args...
  local label=$1; shift
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS --n 2048 --gen 32 --faults -- \
    --ctx 32768 --prefill-chunk 512 "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
if [ "${KLD:-0}" != only ]; then
  run mtp --mtp $D --spec 2 --draft-vocab $V
  run plain
fi
# KLD gate, unchanged outputs expected (the fast path, and the chunk path as in sw83)
if [ "${KLD:-0}" != 0 ]; then
  wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast > $O/kld-fast.log 2>&1); echo "kld fast rc=$?"
  wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --prefill-chunk 1024 > $O/kld-chunk.log 2>&1); echo "kld chunk rc=$?"
fi
echo done > $O/DONE
