#!/bin/bash
# sw106: why did the fast-path KLD rise with sw104's cache settings (--fast 0.008602 -> 0.009204, the
# hit rate unchanged at 91%, swaps 55K -> 159K)? A race between uploads and reads would make runs
# differ; every fast-path KLD so far repeats exactly (sw94). The new defaults twice, each setting
# alone, and the old settings: fr_kld used CachePolicyConfig's defaults, budget 8 and seed 1 (must
# give 0.008602 and 55,410 swaps again, sw101).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw106; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
k() { local name=$1; shift; wait_vram
      (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast "$@" > $O/kld-$name.log 2>&1)
      echo "$name $(grep -h 'KLD mean' $O/kld-$name.log | awk '{print $3, $5, $7}') $(grep -hoE 'hit rate [0-9.]+%' $O/kld-$name.log) $(grep -hoE '[0-9]+ swaps' $O/kld-$name.log)"; }
k new-a
k new-b
k old --cache-seed-scale 1 --swap-budget 8   # fr_kld used the struct defaults: budget 8
k seed1-b64 --cache-seed-scale 1 --swap-budget 64
k seed003-b8 --cache-seed-scale 0.03 --swap-budget 8
echo done > $O/DONE
