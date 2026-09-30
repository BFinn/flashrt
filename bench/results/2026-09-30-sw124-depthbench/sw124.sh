#!/bin/bash
# sw124: window 9's protocol (sw110, sw119) after P-5's kernels: the QSA selection on a cluster
# (sw121) and the parallel CLOCK (sw122). Arms as sw110: P no head, G head greedy, S head at
# temperature 1.0; 5 runs each, interleaved.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
O=$BENCH/sw124; mkdir -p $O; cd $O
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $FLASHRT/build/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
P(){ $DB --label sw124-P-plain-r$1 --log $O/P$1.log 2>&1 | tee -a $O/sw124.out; }
G(){ $DB --label sw124-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw124.out; }
S(){ $DB --label sw124-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw124.out; }
P 1; G 1; S 1; G 2; S 2; P 2; S 3; P 3; G 3; P 4; G 4; S 4; G 5; S 5; P 5
echo ALLDONE
