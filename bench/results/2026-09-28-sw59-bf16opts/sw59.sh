#!/bin/bash
# sw59: the layer-end hc combine deferred into the next layer's mix (default; exact), and the
# options FLASHRT_MOE_YD16 (per-slot expert outputs in BF16) and FLASHRT_HC_GATE16 (hc gate in
# BF16): prefill at 32K (q8 KV, automatic chunks), one run each; then the KLD gate with both
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw59; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for arm in "default:0:0" "yd16:1:0" "gate16:0:1" "both:1:1"; do
  IFS=: read name yd gt <<< "$arm"
  wait_vram; FLASHRT_MOE_YD16=$yd FLASHRT_HC_GATE16=$gt $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/n32k_$name.txt 2>&1; echo "$name rc=$?"
done
cd $BENCH/kld
run() { name=$1; shift; wait_vram; FLASHRT_MOE_YD16=1 FLASHRT_HC_GATE16=1 timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run both-chunk1024-f16 --prefill-chunk 1024
run both-chunk1024-q8 --prefill-chunk 1024 --kv q8
echo done > $O/DONE
