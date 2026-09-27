#!/bin/bash
# speed work 11: adaptive expert cache. 2K: 2 runs static, 2 runs adaptive (one with a trace); fast-path KLD (adaptive).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw11; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for r in 1 2; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --static-cache > $O/static_2k_r$r.txt 2>&1; echo "static r$r rc=$?"; done
wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --trace $O/trace_2k.i16 > $O/adapt_2k_r1.txt 2>&1; echo "adapt r1 rc=$?"
wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 > $O/adapt_2k_r2.txt 2>&1; echo "adapt r2 rc=$?"
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast > $O/kl8k-fast-adaptive.log 2>&1); echo "kld rc=$?"
echo done > $O/DONE
