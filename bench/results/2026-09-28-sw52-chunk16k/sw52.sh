#!/bin/bash
# sw52: chunks of 16,384 against 8,192 (fuller expert MMQ tiles: about 320 tokens per expert),
# q8 KV at 32K and 64K, and 245K q8 in VRAM; one run each
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw52; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for n in 32768 65536; do
  wait_vram; $B/fr_bench $M --ids $I --n-prompt $n --gen 8 --prefill-chunk 16384 --kv q8 > $O/n${n}_c16k.txt 2>&1; echo "$n rc=$?"
done
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 16384 --kv q8 > $O/n245k_c16k.txt 2>&1; echo "245k rc=$?"
echo done > $O/DONE
