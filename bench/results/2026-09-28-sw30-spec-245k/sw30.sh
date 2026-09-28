#!/bin/bash
# sw30: 245K fresh prefill with the MTP head (q8 KV, hot set 4096), state saved with .mtp; then speculative decode k=2
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw30; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 4000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --kv-hot 4096 --mtp $D --spec 2 --draft-vocab $V --save-state $BENCH/state-245k-q8-mtp.bin > $O/spec2_245k_fresh.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
