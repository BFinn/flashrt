#!/bin/bash
# sw33: sw32 rerun after the graph-scratch fix, plus the sw31 arms that aborted
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw33; mkdir -p $O
T1="--temp 1.0 --top-k 20 --top-p 0.95"
S32=$BENCH/state-32k-q8-mtp.bin; S245=$BENCH/state-245k-q8-mtp.bin
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --window 3 --kv-hot 512 > $O/kl8k-fast-win3-hot512.log 2>&1); echo "kld win3 rc=$?"
run spec2_q2_t1_32k --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $S32 --mtp $D --mtp-bits 2 --spec 2 --draft-vocab $V $T1
run spec2_q2_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --mtp-bits 2 --spec 2 --draft-vocab $V $T1
run spec1_q2_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --mtp-bits 2 --spec 1 --draft-vocab $V $T1
run spec3_q2_g_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --mtp-bits 2 --spec 3 --draft-vocab $V
run spec2_t1_32k --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $S32 --mtp $D --spec 2 --draft-vocab $V $T1
run spec2_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1
run dist_245k --n-prompt 245760 --gen 16 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1 --dist-test 300
run spec2_t1_w11_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1 --workers 11
echo done > $O/DONE
