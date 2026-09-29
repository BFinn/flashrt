#!/bin/bash
# sw89: the expert cache on window 9's protocol (sw87, sw88: the cache filled from the prompt's
# routing misses what the model then generates; hit rate 66% against 93% on wikitext). Plain
# decode, greedy, flashrt_depthbench (the window 9 protocol), one run each:
#   prior   --cache-prior (the calibration routing prior, as the engine's deployment uses)
#   swap32  --swap-budget 32 (faster adaptation during decode; default 8)
#   both
#   spec    both, with --mtp --spec 2, greedy
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
P=$BENCH/cache-prior-calib32k.bin
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $FLASHRT/build/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
O=$BENCH/sw89; mkdir -p $O; cd $O
$DB --label sw89-prior --log $O/prior.log -- --cache-prior $P 2>&1 | tee -a $O/sw89.out
$DB --label sw89-swap32 --log $O/swap32.log -- --swap-budget 32 2>&1 | tee -a $O/sw89.out
$DB --label sw89-both --log $O/both.log -- --cache-prior $P --swap-budget 32 2>&1 | tee -a $O/sw89.out
$DB --label sw89-spec --log $O/spec.log -- --cache-prior $P --swap-budget 32 --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw89.out
echo done > $O/DONE
