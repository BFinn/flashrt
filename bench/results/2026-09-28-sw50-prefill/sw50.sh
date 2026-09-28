#!/bin/bash
# sw50: GDN column kernel v3 (two columns per thread, 3-stage tiles), arena registered at load:
# prefill at 32K, 64K (q8 KV) and 245K (q8 in VRAM; host KV + hot set with the VRAM mirror),
# chunks of 8,192; then the KLD gate
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw50; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for col in 0 1; do
  wait_vram; FLASHRT_GDN_COL=$col $B/fr_bench $M --ids $I --n-prompt 32768 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n32k_col$col.txt 2>&1
done
wait_vram; $B/fr_bench $M --ids $I --n-prompt 65536 --gen 8 --prefill-chunk 8192 --kv q8 > $O/n64k.txt 2>&1
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 8192 --kv q8 > $O/n245k_q8.txt 2>&1; echo "q8 rc=$?"
wait_vram; timeout 3000 $B/fr_bench $M --ids $I --n-prompt 245760 --gen 16 --prefill-chunk 8192 --kv-hot 4096 > $O/n245k_hot.txt 2>&1; echo "hot rc=$?"
cd $BENCH/kld
run() { name=$1; shift; wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 "$@" > $O/$name.log 2>&1; echo "$name rc=$?"; }
run chunk1024-f16 --prefill-chunk 1024
run chunk1024-q8 --prefill-chunk 1024 --kv q8
run fast-chunk2048-hot512 --fast --prefill-chunk 2048 --kv-hot 512
echo done > $O/DONE
