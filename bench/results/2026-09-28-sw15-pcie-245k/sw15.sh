#!/bin/bash
# speed work 15: PCIe misses. 2K A/B (pcie 0.5 vs 0, 2 runs each); fast KLD with pcie;
# 245,760 prefill saving the state, decode 3 windows (pcie 0.5); then the saved state with pcie 0.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw15; mkdir -p $O
ST=$BENCH/state-245k.bin
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for r in 1 2; do
  wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 > $O/pcie05_2k_r$r.txt 2>&1; echo "pcie05 r$r rc=$?"
  wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 --pcie-frac 0 > $O/pcie0_2k_r$r.txt 2>&1; echo "pcie0 r$r rc=$?"
done
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast > $O/kl8k-fast-pcie.log 2>&1); echo "kld rc=$?"
wait_vram; timeout 7200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --save-state $ST > $O/pcie05_245k.txt 2>&1; echo "245k save rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --load-state $ST --pcie-frac 0 > $O/pcie0_245k.txt 2>&1; echo "245k pcie0 rc=$?"
wait_vram; timeout 1200 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 128 --windows 3 --load-state $ST > $O/pcie05_245k_load.txt 2>&1; echo "245k pcie05 load rc=$?"
echo done > $O/DONE
