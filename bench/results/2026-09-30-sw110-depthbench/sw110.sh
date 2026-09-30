#!/bin/bash
# sw110: window 9's protocol as sw96, on the final build of the cache warm-up work (sw104-sw109): each depth's
# prompt keeps the previous one's text but for its last 19 tokens, so flashrt can reuse up to the
# checkpoint before the previous prompt's tail, as Strata and llama.cpp reused prefixes.
# The same token ids (strata-ids.json: 1K / 32K / 134K / 250K), one growing conversation, 384
# generated tokens per depth without stopping at end-of-sequence, a fresh engine per run, arms
# interleaved in rotating order, 5 runs each. Every row records reused tokens, prompt time, draft
# acceptance, the decode's expert-cache hit rate, and the GPU clock, temperature and power.
#   P  no MTP head, greedy (against llama.cpp, arm L)
#   G  --mtp --spec 2 (argmax drafts), greedy (against Strata, arm G)
#   S  --mtp --spec 2, temperature 1.0, top_p 0.95, top_k 20 (sampled drafts; against Strata, arm S)
# Engine defaults otherwise: 262K context, q8 host KV + hot set 4,096, reserve 512 MiB; the expert
# cache: admit 1, margin 1.2, seed scale 0.03, 64 uploads per step committed at the next step; 8
# host checkpoints with the adaptive tail; the head's prompt pass in chunk calls; no cache prior.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/strata-ids.json
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $FLASHRT/build/flashrt-engine --model $M --ids $IDS --gen 384"
O=$BENCH/sw110; mkdir -p $O; cd $O
P(){ $DB --label sw110-P-plain-r$1 --log $O/P$1.log 2>&1 | tee -a $O/sw110.out; }
G(){ $DB --label sw110-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw110.out; }
S(){ $DB --label sw110-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw110.out; }
echo START $(date +%T)
P 1; G 1; S 1
G 2; S 2; P 2
S 3; P 3; G 3
P 4; G 4; S 4
G 5; S 5; P 5
echo DONE $(date +%T)
echo done > $O/DONE
