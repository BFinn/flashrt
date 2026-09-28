#!/bin/bash
# sw47: KLD gate for the prefill kernels (tensor-core attention, column GDN), logits straight
# from prefill chunks: fp16 KV, q8 KV, host KV + hot set (the VRAM mirror); and the fast decode
# path after a chunked prefill. First the column GDN's speed: 32K q8, chunks of 8,192, on and off
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw47; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
for col in 0 1; do
  wait_vram; FLASHRT_GDN_COL=$col $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n32k_q8_col$col.txt 2>&1
done
cd $BENCH/kld
run() { name=$1; shift; wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run chunk1024-f16 --prefill-chunk 1024
run chunk1024-q8 --prefill-chunk 1024 --kv q8
run chunk1024-hot512 --prefill-chunk 1024 --kv-hot 512
run fast-chunk2048-hot512 --fast --prefill-chunk 2048 --kv-hot 512
echo done > $O/DONE
