#!/bin/bash
# speed work 4: KLD gate after the kernel changes, then 3 fr_bench runs at 2K and an nsys profile.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw4; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; (cd $BENCH/kld && $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 > $O/kl8k-flashrt.log 2>&1); echo "kld rc=$?"
for r in 1 2 3; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 128 > $O/db_2k_r$r.txt 2>&1; echo "db r$r rc=$?"; done
wait_vram
cd $O && /usr/local/cuda-12.9/bin/nsys profile -o decode_2k --force-overwrite true --capture-range=cudaProfilerApi --trace=cuda,osrt \
  $B/fr_bench $M --ids $I --n-prompt 2048 --gen 64 > prof.txt 2>&1
/usr/local/cuda-12.9/bin/nsys stats -q --report cuda_gpu_kern_sum,cuda_api_sum --format csv -o decode_2k decode_2k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
