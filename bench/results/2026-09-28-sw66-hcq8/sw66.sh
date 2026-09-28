#!/bin/bash
# sw66: hc down/up matrices as Q8P (default) against BF16 (FLASHRT_HC_Q8=0): KLD gate (the fast
# decode path, windows of 3, logits from prefill chunks), then teacher-forced --spec 1 at P2
# conditions from the saved states, 6 windows of 128 tokens
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw66; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
kld() { name=$1; shift; wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1); echo "$name rc=$?"; }
kld fast-hot512 --fast --prefill-chunk 2048 --kv-hot 512
kld fast-win3-hot512 --fast --window 3 --prefill-chunk 2048 --kv-hot 512
kld chunk1024-f16 --prefill-chunk 1024
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
for q in 1 0; do
  export FLASHRT_HC_Q8=$q
  run spec1_32k_q$q --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --teacher
  run spec1_245k_q$q --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --teacher
  run plain_32k_q$q --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin $T1 --teacher
done
echo done > $O/DONE
