#!/bin/bash
# sw79: why a CPU miss costs ~97 us in decode against 32 us standalone (p1-moe-cpu): plain 32K,
# teacher-forced, P2 conditions from the saved state, 2 windows of 128 tokens per arm; the miss
# server's per-layer host time by miss count is in each log. Arms: default (8 workers, adaptive
# cache, swap budget 8), no expert swaps (--static-cache), 6 and 11 workers, workers never sleep
# (--spin-us 100000); then the standalone bench_moe_cpu for the same box state
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
S32=$BENCH/state-32k-q8-mtp.bin
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
B=$FLASHRT/build; O=$BENCH/sw79; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M --ids $I --n-prompt 32768 --gen 128 --windows 2 --kv-hot 4096 --load-state $S32 $T1 --teacher "$@" > $O/$name.txt 2>&1; echo "$name rc=$?"; }
run default
run static --static-cache
run w6 --workers 6
run w11 --workers 11
run spin --spin-us 100000
$B/bench_moe_cpu > $O/bench_moe_cpu.txt 2>&1; echo "bench_moe_cpu rc=$?"
echo done > $O/DONE
