#!/bin/bash
# sw31: distribution test; Q2_0 drafter; P2 conditions (temperature 1.0, top-p 0.95, top-k 20) at 32K and 245K, plain vs speculative
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw31; mkdir -p $O
T1="--temp 1.0 --top-k 20 --top-p 0.95"
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
run spec2_chain_2k --n-prompt 2048 --gen 256 --mtp $D --spec 2 --draft-vocab $V
run dist_2k --n-prompt 2048 --gen 64 --mtp $D --spec 2 --draft-vocab $V $T1 --dist-test 400
run spec2_q2_2k --n-prompt 2048 --gen 256 --mtp $D --spec 2 --draft-vocab $V --mtp-bits 2
run spec2_t1_32k_fresh --n-prompt 32768 --gen 128 --windows 3 --kv-hot 4096 --mtp $D --spec 2 --draft-vocab $V $T1 --save-state $BENCH/state-32k-q8-mtp.bin
S32=$BENCH/state-32k-q8-mtp.bin; S245=$BENCH/state-245k-q8-mtp.bin
run plain_t1_32k --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $S32 $T1
run spec1_t1_32k --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $S32 --mtp $D --spec 1 --draft-vocab $V $T1
run spec2_t1_32k --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $S32 --mtp $D --spec 2 --draft-vocab $V $T1
run plain_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 $T1
run spec1_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 1 --draft-vocab $V $T1
run spec2_t1_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1
run spec2_g_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V
run dist_245k --n-prompt 245760 --gen 16 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1 --dist-test 300
run spec2_w11_2k --n-prompt 2048 --gen 256 --mtp $D --spec 2 --draft-vocab $V --workers 11
run spec2_t1_w11_245k --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $S245 --mtp $D --spec 2 --draft-vocab $V $T1 --workers 11
echo done > $O/DONE
