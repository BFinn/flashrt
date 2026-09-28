#!/bin/bash
# sw26: MTP drafter variants (Q4_0 experts, trimmed vocabulary): acceptance probe and speculative decode, 2K
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw26; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --draft 3 > $O/probe_q4.txt 2>&1; echo "probe q4 rc=$?"
wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --draft 3 --draft-vocab $V > $O/probe_q4_v32k.txt 2>&1; echo "probe q4 v32k rc=$?"
wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --draft 3 --draft-vocab $V --draft-vocab-n 16384 > $O/probe_q4_v16k.txt 2>&1; echo "probe q4 v16k rc=$?"
for k in 1 2 3; do wait_vram; timeout 900 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --mtp $D --spec $k --draft-vocab $V > $O/spec${k}_q4_v32k.txt 2>&1; echo "spec $k rc=$?"; done
echo done > $O/DONE
