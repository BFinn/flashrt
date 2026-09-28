#!/bin/bash
# sw54: automatic prefill chunk length (ForwardRef::pick_chunk from free VRAM): fr_bench
# --prefill-chunk auto at 32K and 245K (q8 in VRAM; host KV + hot set with the mirror), then the
# engine (auto is its default) through engine_smoke.py with a 32K prompt, MTP spec 1
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw54; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk auto --kv q8 > $O/n32k_auto.txt 2>&1; echo "32k rc=$?"
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk auto --kv q8 > $O/n245k_q8_auto.txt 2>&1; echo "245k q8 rc=$?"
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk auto --kv-hot 4096 > $O/n245k_hot_auto.txt 2>&1; echo "245k hot rc=$?"
wait_vram; timeout 1800 python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $I --n 32000 --gen 96 -- \
  --mtp $D --spec 1 --draft-vocab $BENCH/mtp-vocab/ranks.txt --ctx 65536 --cache-prior $BENCH/cache-prior-calib32k.bin > $O/engine_smoke.txt 2>&1; echo "engine rc=$?"
echo done > $O/DONE
