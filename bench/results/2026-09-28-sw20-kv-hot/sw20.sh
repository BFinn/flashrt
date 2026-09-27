#!/bin/bash
# KV compression 2: host-resident q8 KV with a GPU hot set. Exactness vs plain q8 (2K, tiny hot set;
# fast KLD with 512 blocks), then 245K and 32K from the saved states with 4096 blocks.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw20; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --kv-hot 256 > $O/hot256_2k.txt 2>&1; echo "2k hot rc=$?"
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --kv-hot 512 > $O/kl8k-fast-hot512.log 2>&1); echo "kld rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --load-state $BENCH/state-245k.bin --kv-hot 4096 > $O/hot4096_245k.txt 2>&1; echo "245k rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 3 --load-state $BENCH/state-32k.bin --kv-hot 4096 > $O/hot4096_32k.txt 2>&1; echo "32k rc=$?"
echo done > $O/DONE
