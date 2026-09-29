#!/bin/bash
# sw94b: the bisect (sw94) put the fast-path KLD move at a97bac1, the prefill routing kernel with
# a warp per token, whose softmax sums in another order (p may differ in the last bit). On the
# current build, FLASHRT_ROUTE_WARP=0 (the block kernel) must give back sw78's 0.008931 and 12,021
# swaps exactly if nothing else moved it; the plain --fast configuration shows the size of the
# same perturbation there (0.008688 with the warp kernel, sw92).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw94; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
k() {   # label, env, fr_kld args...
  local label=$1 env=$2; shift 2
  wait_vram
  (cd $BENCH/kld && env $env timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast "$@" > $O/kld-$label.log 2>&1)
  echo "$label rc=$? $(grep -h 'KLD mean' $O/kld-$label.log | awk '{print $3}') $(grep -ho '[0-9]* swaps' $O/kld-$label.log)"
}
k head-win3-block FLASHRT_ROUTE_WARP=0 --window 3 --prefill-chunk 2048 --kv-hot 512
k head-fast-block FLASHRT_ROUTE_WARP=0
k head-fast-warp FLASHRT_ROUTE_WARP=1
echo done > $O/DONE-b
