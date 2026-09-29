#!/bin/bash
# sw95: host checkpoints during the prefill (phase 2 of docs/improvement-plan.md, R-1).
# 1. engine_smoke.py --reuse, with and without the MTP head: a prompt with a fixed tail after
#    grown text, one with a change half way, and a cancelled prefill (the stale-checkpoint case)
#    must reuse the expected prefix and match a cold run's state.
# 2. engine_smoke.py --faults again (phase 1's checks, now with the ring).
# 3. The tail's cost: a 32K prompt cold with --ckpt-tail 64 against 0 (prompt_ms), 3 runs each.
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
if [ "${PART:-all}" = all ] || [ "${PART}" = reuse ]; then
  smoke reuse-mtp "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
  smoke reuse-plain "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
  smoke faults-mtp "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
  smoke faults-plain "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512
fi
if [ "${PART:-all}" = all ] || [ "${PART}" = tail ]; then
  for t in 64 0 64 0 64 0; do
    wait_vram
    python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS --n 32768 --gen 16 -- \
      --ctx 65536 --mtp $D --spec 2 --draft-vocab $V --ckpt-tail $t > $O/tail$t-$(date +%s).txt 2>&1
    echo "tail $t rc=$?"
  done
fi
echo done > $O/DONE
