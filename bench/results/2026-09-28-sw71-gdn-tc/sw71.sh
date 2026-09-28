#!/bin/bash
# sw71: the chunked GDN on fp16 tensor cores (FLASHRT_GDN_CHUNK=1) against the column kernel:
# the chunk-logit KLD first (fp16 KV, 1,024-token chunks), then 32K and 64K prefill (q8 KV,
# automatic chunks), one run each
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw71; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; (cd $BENCH/kld && FLASHRT_GDN_CHUNK=1 timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --prefill-chunk 1024 > $O/kld-chunk1.log 2>&1); echo "kld rc=$?"
for n in 32768 65536; do
  for c in 0 1; do
    wait_vram; FLASHRT_GDN_CHUNK=$c $B/fr_bench $M --ids $I --n-prompt $n --gen 8 --prefill-chunk auto --kv q8 > $O/n${n}_c$c.txt 2>&1; echo "$n $c rc=$?"
  done
done
echo done > $O/DONE
