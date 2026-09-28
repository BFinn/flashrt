#!/bin/bash
# sw69: prefill with Q3_K matrices multiplied as Q8_0 (default) against Q3_K (FLASHRT_Q3_Q8=0):
# 32K and 64K, q8 KV, automatic chunks, one run each; then the chunk-logit KLD
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw69b; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for n in 32768 65536; do
  for q in 0 1; do
    wait_vram; FLASHRT_Q3_Q8=$q $B/fr_bench $M --ids $I --n-prompt $n --gen 8 --prefill-chunk auto --kv q8 > $O/n${n}_q$q.txt 2>&1; echo "$n $q rc=$?"
  done
done
wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --prefill-chunk 1024 > $O/chunk1024-f16.log 2>&1); echo "kld rc=$?"
echo done > $O/DONE
