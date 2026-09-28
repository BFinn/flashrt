#!/bin/bash
# sw49: tensor-core indexer scores (FLASHRT_IDX_TC) and the 8-warp column GDN: speed at 32K and
# 64K (q8 KV, chunks of 8,192), then the KLD gate (logits from prefill chunks, fp16 and q8 KV)
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw49; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for n in 32768 65536; do
  for idx in 0 1; do
    wait_vram; FLASHRT_IDX_TC=$idx $B/fr_bench $M --ids $I --n-prompt $n --gen 8 --prefill-chunk 8192 --kv q8 > $O/n${n}_idx$idx.txt 2>&1
  done
done
cd $BENCH/kld
run() { name=$1; shift; wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run chunk1024-f16 --prefill-chunk 1024
run chunk1024-q8 --prefill-chunk 1024 --kv q8
echo done > $O/DONE
