#!/bin/bash
# sw84: what speculative sampling (drafts drawn from the head's q, accepted with min(1, p/q)) would
# accept against the argmax drafts used now: fr_bench --accept-probe, --spec 1 at P2 conditions
# (temperature 1.0, top-k 20, top-p 0.95), from the saved 32K and 245K states, 2 windows of 128
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw84; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
run probe_32k --n-prompt 32768 --gen 128 --windows 2 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --accept-probe
run probe_245k --n-prompt 245760 --gen 128 --windows 2 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --accept-probe
run probe_32k_full --n-prompt 32768 --gen 128 --windows 2 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --mtp $D --spec 1 $T1 --accept-probe
echo done > $O/DONE
