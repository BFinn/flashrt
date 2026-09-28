#!/bin/bash
# sw57: moe_q2 with activation scales per 64 (FLASHRT_MOE_AB64=1) against per 32 (down kernel with
# resident activations). Prefill at 32K (q8 KV, automatic chunks), then the KLD gate with per-64
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw57; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for ab in 0 1; do
  wait_vram; FLASHRT_MOE_AB64=$ab $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/n32k_ab64_$ab.txt 2>&1; echo "32k $ab rc=$?"
done
cd $BENCH/kld
run() { name=$1; shift; wait_vram; FLASHRT_MOE_AB64=1 timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run ab64-chunk1024-f16 --prefill-chunk 1024
run ab64-chunk1024-q8 --prefill-chunk 1024 --kv q8
echo done > $O/DONE
