#!/bin/bash
# sw44: KLD with host KV + hot set through prefill chunks (the VRAM mirror): chunk logits, and chunked prefill then the fast decode path
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$FLASHRT/build; O=$BENCH/sw44; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cd $BENCH/kld

wait_vram; timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --fast --prefill-chunk 2048 --kv-hot 512 > $O/fast-chunk2048-hot512.log 2>&1; echo "fast chunk hot rc=$?"
echo done > $O/DONE
