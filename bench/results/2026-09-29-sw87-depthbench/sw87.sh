#!/bin/bash
# sw87: flashrt on the reference runs' protocol (window 9, bench/results/2026-09-27-w9-validation):
# the same token ids (strata-ids.json: 1K / 32K / 134K / 250K), one growing conversation with
# prefix reuse, 384 generated tokens per depth without stopping at end-of-sequence, a fresh
# engine per run, arms interleaved, 3 runs each:
#   P  flashrt-engine without the MTP head, greedy (against llama.cpp, arm L)
#   G  --mtp --spec 2 (argmax drafts), greedy (against Strata, arm G)
#   S  --mtp --spec 2, temperature 1.0, top_p 0.95, top_k 20 (sampled drafts; against Strata, arm S)
# Engine defaults otherwise (262K context, q8 host KV + hot set 4,096, reserve 512 MiB), no cache prior.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/strata-ids.json
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $FLASHRT/build/flashrt-engine --model $M --ids $IDS --gen 384"
O=$BENCH/sw87; mkdir -p $O; cd $O
P(){ $DB --label sw87-P-plain-r$1 --log $O/P$1.log 2>&1 | tee -a $O/sw87.out; }
G(){ $DB --label sw87-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw87.out; }
S(){ $DB --label sw87-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw87.out; }
echo START $(date +%T)
P 1; G 1; S 1
G 2; S 2; P 2
S 3; P 3; G 3
echo DONE $(date +%T)
echo done > $O/DONE
