#!/bin/bash
# sw95c: engine_smoke --reuse with the fixed-tail pattern set up before the cold references (sw95b's
# tail case compared a reference prefilled without the tail batch), on the build with the faster
# MTP head load (test_mtp_convert first: bit-identical conversion on the real head).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw95; mkdir -p $O
cmake --build $B > $O/build-c.log 2>&1 || { echo "build failed"; exit 1; }
FLASHRT_TEST_MTP=$D $B/test_mtp_convert > $O/mtp-convert.txt 2>&1; echo "test_mtp_convert rc=$? $(grep 'head' $O/mtp-convert.txt)"
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
smoke reuse-plain-c "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke reuse-mtp-c "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
echo done > $O/DONE-c
