#!/bin/bash
# speed work 18: Q8_0 KV cache. Fast KLD (q8 KV throughout), 3 runs at 2K, 32K and 245K from the fp16 states
# (converted to q8 on load; speed only), 3 windows each.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw18; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --kv q8 > $O/kl8k-fast-q8.log 2>&1); echo "kld rc=$?"
for r in 1 2 3; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --kv q8 > $O/q8_2k_r$r.txt 2>&1; echo "2k r$r rc=$?"; done
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 3 --load-state $BENCH/state-32k.bin --kv q8 > $O/q8_32k.txt 2>&1; echo "32k rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --load-state $BENCH/state-245k.bin --kv q8 > $O/q8_245k.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
