#!/bin/bash
# sw73: decode's small BF16 mat-vecs fused (linear_multi: alpha + beta + gates, router + shared-expert
# gate, indexer q + k; the default) against separate launches (FLASHRT_LINEAR_MULTI=0): KLD of
# verify windows; then teacher-forced decode at P2 conditions from the saved states (plain 32K;
# --spec 1 at 32K and 245K), 6 windows of 128 tokens
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw73; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
kld() { name=$1; shift; wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1); echo "$name rc=$?"; }
kld fast-win3-hot512 --fast --window 3 --prefill-chunk 2048 --kv-hot 512
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
for f in 1 0; do
  export FLASHRT_LINEAR_MULTI=$f
  run plain_32k_$f --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin $T1 --teacher
  run spec1_32k_$f --n-prompt 32768 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --teacher
  run spec1_245k_$f --n-prompt 245760 --gen 128 --windows 6 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --mtp $D --spec 1 --draft-vocab $V $T1 --teacher
done
echo done > $O/DONE
