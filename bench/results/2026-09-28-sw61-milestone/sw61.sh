#!/bin/bash
# sw61: milestone with every default of the prefill work (moe_q2 per 32, BF16 expert outputs and
# hc gate, hc fusions, automatic chunks): prefill at 32K, 64K (q8 KV), 245K (q8 KV; host KV + hot
# set 4096), one run each; the fast-path KLD after a chunked prefill; the engine with a 32K prompt
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw61; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for n in 32768 65536; do
  wait_vram; $B/fr_bench $M --ids $I --n-prompt $n --gen 16 --prefill-chunk auto --kv q8 > $O/n$n.txt 2>&1; echo "$n rc=$?"
done
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk auto --kv q8 > $O/n245k_q8.txt 2>&1; echo "245k q8 rc=$?"
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk auto --kv-hot 4096 > $O/n245k_hot.txt 2>&1; echo "245k hot rc=$?"
wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --fast --prefill-chunk 2048 --kv-hot 512 > $O/fast-chunk2048-hot512.log 2>&1); echo "fast kld rc=$?"
wait_vram; timeout 1800 python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $I --n 32000 --gen 96 -- \
  --mtp $D --spec 1 --draft-vocab $BENCH/mtp-vocab/ranks.txt --ctx 65536 --cache-prior $BENCH/cache-prior-calib32k.bin > $O/engine_smoke.txt 2>&1; echo "engine rc=$?"
echo done > $O/DONE
