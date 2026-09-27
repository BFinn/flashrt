#!/bin/bash
# speed work 14: dp4a MoE hit kernels (planar cache), indexer scoring/select at depth, k_hc_down 16 rows.
# Unit tests, fast-path KL gate, 3 runs at 2K (adaptive), nsys.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw14; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $B/test_moe_hits > $O/test_moe_hits.txt 2>&1; echo "test_moe_hits rc=$?"
$B/test_cpu_pool > $O/test_cpu_pool.txt 2>&1; echo "test_cpu_pool rc=$?"
wait_vram; $B/fr_parity $M $BENCH/parity/long2216.frd qsa > $O/par_qsa_long.txt 2>&1; echo "qsa long rc=$?"
wait_vram; (cd $BENCH/kld && timeout 1800 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast > $O/kl8k-fast.log 2>&1); echo "kld rc=$?"
for r in 1 2 3; do wait_vram; timeout 600 $B/fr_bench $M --ids $I --n-prompt 2048 --gen 256 > $O/db_2k_r$r.txt 2>&1; echo "db r$r rc=$?"; done
wait_vram
cd $O && /usr/local/cuda-12.9/bin/nsys profile -o decode_2k --force-overwrite true --capture-range=cudaProfilerApi --trace=cuda,osrt \
  $B/fr_bench $M --ids $I --n-prompt 2048 --gen 64 > prof.txt 2>&1
/usr/local/cuda-12.9/bin/nsys stats -q --report cuda_gpu_kern_sum,cuda_api_sum --format csv -o decode_2k decode_2k.nsys-rep > /dev/null 2>&1
echo done > $O/DONE
