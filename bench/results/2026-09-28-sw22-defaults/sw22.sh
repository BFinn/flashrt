#!/bin/bash
# sw22: warp-aggregated select histogram, host-resident token embedding, swap budget 32 by default.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw22; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $B/fr_parity $M $BENCH/parity/long2216.frd qsa > $O/par_qsa_long.txt 2>&1; echo "qsa rc=$?"
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --kv-hot 512 > $O/kl8k-fast-hot512.log 2>&1); echo "kld rc=$?"
for r in 1 2 3; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 > $O/db_2k_r$r.txt 2>&1; echo "2k r$r rc=$?"; done
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 3 --load-state $BENCH/state-32k.bin --kv-hot 4096 > $O/hot_32k.txt 2>&1; echo "32k rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 6 --load-state $BENCH/state-245k-q8.bin --kv-hot 4096 > $O/hot_245k.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
