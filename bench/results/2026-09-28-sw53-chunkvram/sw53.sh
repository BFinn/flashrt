#!/bin/bash
# sw53: VRAM held by the chunk path per chunk size (32K, q8 KV: 8,192 / 12,288 / 16,384), and
# 245K q8 in VRAM with chunks of 12,288; one run each
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw53; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for c in 8192 12288 16384; do
  wait_vram; $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk $c --kv q8 > $O/n32k_c$c.txt 2>&1; echo "$c rc=$?"
done
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 12288 --kv q8 > $O/n245k_c12288.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
