#!/bin/bash
# sw95b: the tail checkpoint made adaptive (taken only once prompts show a fixed tail, as long as
# that tail): build, the GPU tests, then engine_smoke.py --reuse again in both arms; with the MTP
# head's prefill mirror (sw98) in the build, --faults again in the MTP arm.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw95; mkdir -p $O
cmake --build $B > $O/build-b.log 2>&1 || { echo "build failed"; exit 1; }
(cd $B && FLASHRT_TEST_MODEL=$M ctest --output-on-failure > $O/ctest-b.txt 2>&1); echo "ctest rc=$? $(grep -E 'tests passed|tests failed' $O/ctest-b.txt)"
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
smoke reuse-plain-b "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke reuse-mtp-b "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
# the MTP head's prefill mirror (sw98) must be freed when a request fails mid-prefill
smoke faults-mtp-b "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
echo done > $O/DONE-b
