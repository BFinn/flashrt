#!/bin/bash
# sw115c: the verify phase's cost for windows of 2..8 tokens (lookup drafts would verify up to 7),
# teacher-forced --spec 1..7 with round logs, on the 32K wikitext state and the code-edit text
# (sw115b's code.ids). 2 windows of 256.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw115c; mkdir -p $O
NP=18219   # sw115b: the code edit's prompt length
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for k in 1 2 3 4 5 6 7; do
  wait_vram; $B/fr_bench $M --ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --teacher --gen 256 --windows 2 \
    --mtp $D --spec $k --draft-vocab $V --round-log $O/wiki32k_spec$k.rounds > $O/wiki32k_spec$k.txt 2>&1
  echo "wiki32k spec $k rc=$? $(grep -h 'per round' $O/wiki32k_spec$k.txt)"
  wait_vram; $B/fr_bench $M --ids $BENCH/sw115b/code.ids --n-prompt $NP --prefill-chunk auto --kv q8 --kv-hot 4096 --teacher --gen 256 --windows 2 \
    --mtp $D --spec $k --draft-vocab $V --round-log $O/code_spec$k.rounds > $O/code_spec$k.txt 2>&1
  echo "code spec $k rc=$? $(grep -h 'per round' $O/code_spec$k.txt)"
done
echo ALLDONE
