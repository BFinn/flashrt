#!/bin/bash
# Rebuilds the KLD gate's base, $BENCH/kld/kl8k-f16.bin (4.1 GB of logits): llama.cpp's reference
# (the patched tree, llama.cpp-flashnext.patch on ec9281505) over wikitext-2 test (raw), 8,192-token
# chunks, 2 chunks, FP16 KV, batches of 16, experts on the CPU (as bench/results/2026-09-27-p1-kld:
# PPL 2.5239 +- 0.0509 there). fr_kld reads it: see docs/engine.md, "KLD gate".
# wiki.test.raw: sha256 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08.
set -eu
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
C=$DATA/wikitext-2-raw/wiki.test.raw
B=$LLAMA_CPP/build/bin
mkdir -p $BENCH/kld && cd $BENCH/kld
LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -f $C -c 8192 --chunks 2 --no-warmup -lm dio -ngl 99 -ot "ffn_.*_exps=CPU" -fa on -t 12 \
  -b 16 -ub 16 -ctk f16 -ctv f16 --kl-divergence-base kl8k-f16.bin > kl8k-ref.log 2>&1
grep "Final estimate" kl8k-ref.log
