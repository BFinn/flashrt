#!/bin/bash
# sw46: tensor-core prefill attention against the split-K FP32 kernel (FLASHRT_ATTN_TC=0):
# 8K fp16 KV (greedy tokens after), 32K q8 in VRAM, chunks of 8,192
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw46; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for tc in 0 1; do
  wait_vram; FLASHRT_ATTN_TC=$tc $B/fr_bench $M --ids $I --n-prompt 8192 --gen 24 --prefill-chunk 4096 > $O/n8k_tc$tc.txt 2>&1
done
for tc in 0 1; do
  wait_vram; FLASHRT_ATTN_TC=$tc $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n32k_q8_tc$tc.txt 2>&1
done
echo done > $O/DONE
