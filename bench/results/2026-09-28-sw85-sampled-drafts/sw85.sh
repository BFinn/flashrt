#!/bin/bash
# sw85: sampled drafts with speculative sampling (the default at temperature > 0) against argmax
# drafts (--argmax-drafts): distribution tests (400 seeds; --spec 1 and 2), then decode at P2
# conditions (temperature 1.0, top-k 20, top-p 0.95, seed 1; sampled text, so the arms' tokens
# differ) from the saved states, 6 windows of 128 tokens
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
S32="--n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin"
S245="--n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin"
B=$FLASHRT/build; O=$BENCH/sw85; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
run dist_spec1_32k $S32 --gen 16 --windows 1 --mtp $D --spec 1 --draft-vocab $V $T1 --dist-test 400
run dist_spec2_32k $S32 --gen 16 --windows 1 --mtp $D --spec 2 --draft-vocab $V $T1 --dist-test 400
run spec1_32k_sampled $S32 --gen 128 --windows 6 --mtp $D --spec 1 --draft-vocab $V $T1
run spec1_32k_argmax $S32 --gen 128 --windows 6 --mtp $D --spec 1 --draft-vocab $V $T1 --argmax-drafts
run spec2_32k_sampled $S32 --gen 128 --windows 6 --mtp $D --spec 2 --draft-vocab $V $T1
run spec1_245k_sampled $S245 --gen 128 --windows 6 --mtp $D --spec 1 --draft-vocab $V $T1
run spec1_245k_argmax $S245 --gen 128 --windows 6 --mtp $D --spec 1 --draft-vocab $V $T1 --argmax-drafts
run spec2_245k_sampled $S245 --gen 128 --windows 6 --mtp $D --spec 2 --draft-vocab $V $T1
echo done > $O/DONE
