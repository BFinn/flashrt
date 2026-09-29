#!/bin/bash
# sw94: which commit moved the fast-path KLD (window 3, hot set 512, chunks of 2048) from
# 0.008931 (sw78) to 0.009124 (sw92)? Builds fr_kld at each code commit in between, in a
# separate clone, and runs the same KLD command. The runs are deterministic (sw92 reproduced
# 0.009124 and its swap count exactly at 8e6d6b5).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
O=$BENCH/sw94; S=$O/src; mkdir -p $O
[ -d $S ] || git clone -q $FLASHRT $S
COMMITS=${COMMITS:-"b165391 a5c31ff 99303f7 b4373fa a97bac1 795d2c5 e8d5f96 d45c3fd d2da2b0 55c143f 19696d7 0d38d92"}
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for c in $COMMITS; do
  (cd $S && git checkout -q $c) || { echo "$c checkout failed"; continue; }
  if [ ! -d $S/build ]; then
    cmake -S $S -B $S/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc > $O/cmake.log 2>&1
  fi
  cmake --build $S/build --target fr_kld > $O/build-$c.log 2>&1 || { echo "$c build failed"; continue; }
  wait_vram
  (cd $BENCH/kld && timeout 2400 $S/build/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast \
     --window 3 --prefill-chunk 2048 --kv-hot 512 > $O/kld-$c.log 2>&1)
  echo "$c rc=$? $(grep -h 'KLD mean' $O/kld-$c.log | awk '{print $3}') $(grep -ho '[0-9]* swaps' $O/kld-$c.log)"
done
echo done > $O/DONE
