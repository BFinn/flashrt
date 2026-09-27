#!/bin/bash
# Window 9: background validation, interleaved, full depth.
#   L   = llama.cpp dev (block selection + pooled-key cache), no MTP, 48 slots   x4
#   G   = Strata tuned (reserve 1024, pcie-frac 0.35, 8 workers), MTP, greedy    x4
#   S   = Strata tuned + sampling temperature 1.0 top_p 0.95 top_k 20            x4
#   S06 = Strata tuned + sampling temperature 0.6                                 x2
cd $BENCH
trap "systemctl --user start flashnext-262k-server; echo RESTARTED_LIVE \$(date +%T); python3 w9_summary.py w9.out > w9-summary.txt; echo SUMMARY_WRITTEN" EXIT
systemctl --user stop flashnext-262k-server; sleep 5
DEV=$LLAMA_CPP/build/bin
CFG=$STRATA/strata-q2_0.json
TUNE=(--vram-reserve-mib 1024 --set=--pcie-frac=0.35 --set=--pool-workers=8)
L(){ python3 depthbench.py --gen 384 --depths 1000,32000,131000,245000 --label W9-L-llamacpp-r$1 --bin $DEV --env LLAMA_MOE_CACHE_ADMIT=3 --env LLAMA_MOE_CACHE_WINDOW=32 -- --no-warmup -lm dio -ngl 99 -ot "ffn_.*_exps=CPU" -fa on -ctk q8_0 -ctv q8_0 -c 262144 -b 4096 -ub 2048 -t 12 -tb 12 --jinja --parallel 1 --moe-expert-cache 48 --moe-expert-cache-inserts 2 2>&1 | grep -E "depth|died|SUMMARY"; }
G(){ python3 strata_depthbench.py run --cfg $CFG --ids strata-ids.json --label W9-G-strata-greedy-r$1 "${TUNE[@]}" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"; }
S(){ python3 strata_depthbench.py run --cfg $CFG --ids strata-ids.json --label W9-S-strata-t1.0-r$1 "${TUNE[@]}" --sampling "temperature=1.0 top_p=0.95 top_k=20" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"; }
S06(){ python3 strata_depthbench.py run --cfg $CFG --ids strata-ids.json --label W9-S06-strata-t0.6-r$1 "${TUNE[@]}" --sampling "temperature=0.6 top_p=0.95 top_k=20" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"; }
echo START $(date +%T)
L 1; G 1; S 1
G 2; S 2; L 2
S 3; L 3; G 3
L 4; G 4; S 4
S06 1; S06 2
echo W9DONE $(date +%T)
