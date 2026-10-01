#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# A record of a run on the development box (its systemd units, services and paths), not a portable
# script: bench/run.sh is the entry point for another machine.
# P0 window B on the target box. Window A's depth prompts turned out to be 134 distinct
# tokens repeated, so this window re-measures on natural text (wikitext-2, raw):
#   1. h2dbw with concurrent CPU readers (the arm that crashed in window A)
#   2. routing + QSA traces on wikitext at 32K and 131K
#   3. llama.cpp prefill vs ubatch at 32K with pinned (non-mmap) weights
#   4. Strata tuned on wikitext at 1K/32K/131K with --stats, then --prefill 4096 / 8192 at 32K
#
#   systemd-run --user --unit=bench-p0b --collect bash -c "$FLASHRT/bench/p0/window_b.sh > $BENCH/p0b.out 2>&1"
set -u
source "$(dirname "$0")/lib.sh"
OUT=${OUT:-$BENCH/p0b-$(date +%Y%m%d-%H%M)}
GEN=${GEN:-2048}
SDB="python3 $FLASHRT/bench/strata_depthbench.py"
CFG=$STRATA/strata-q2_0.json
TUNE=(--vram-reserve-mib 1024 --set=--pcie-frac=0.35 --set=--pool-workers=8)
mkdir -p "$OUT" && cd "$OUT" || exit 1

# ---- 0. natural-text prompts (vocabulary only; no GPU)
step tokenize wikitext
cat $DATA/wikitext-2-raw/wiki.test.raw $DATA/wikitext-2-raw/wiki.valid.raw > wiki.txt
"$FR/route_trace" --model "$MODEL" --text wiki.txt --n-prompt 140000 --tokenize-only --out wiki || exit 1
python3 - <<'EOF'
ids = list(map(int, open("wiki.prompt_ids.txt").read().split()))
import json
json.dump([{"depth": d, "ids": ids[:d]} for d in (1000, 32000, 131000)], open("wiki-ids.json", "w"))
for d in (32000, 131000):
    open(f"wiki_{d}.txt", "w").write(" ".join(map(str, ids[:d])))
tri = len(set(zip(ids, ids[1:], ids[2:]))) / len(ids)
print(f"wiki prompt: {len(ids)} tokens, {len(set(ids))} distinct, {tri:.3f} unique trigrams per token")
EOF

take_gpu
echo "out=$OUT"

# ---- 1. host->device bandwidth with CPU readers competing for DRAM
step h2dbw concurrent
{
    "$FR/h2dbw" --pages thp --cpu-threads 6
    "$FR/h2dbw" --pages thp --cpu-threads 12
} | tee h2dbw_concurrent.txt

# ---- 2. routing and QSA traces on natural text
step route_trace wikitext
wait_vram
"$FR/route_trace" --model "$MODEL" --ids wiki_32000.txt --gen $GEN --out wtrace_d32000 --probs --prefill-trace --qsa \
    --temp 1.0 --top-p 0.95 --top-k 20 --seed 42
wait_vram
"$FR/route_trace" --model "$MODEL" --ids wiki_131000.txt --gen $GEN --out wtrace_d131000 --prefill-trace --qsa \
    --temp 1.0 --top-p 0.95 --top-k 20 --seed 42

# ---- 3. llama.cpp prefill vs ubatch at 32K, weights read into pinned memory (no mmap)
step llama-bench ubatch sweep, no mmap
wait_vram
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -ot "ffn_.*_exps=CPU" -fa 1 -ctk q8_0 -ctv q8_0 -t 12 -mmp 0 \
    -p 32768 -n 0 -b 8192 -ub 2048,4096,8192 -r 1 -o md | tee ubatch_nommap.md

# ---- 4. Strata on natural text: the reference numbers, its stage breakdown, bigger prefill chunks
step strata wikitext
wait_vram
$SDB run --cfg $CFG --ids wiki-ids.json --out-dir "$OUT" --label P0B-strata-wiki-greedy "${TUNE[@]}" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"
wait_vram
$SDB run --cfg $CFG --ids wiki-ids.json --out-dir "$OUT" --label P0B-strata-wiki-stats --n-depths 2 "${TUNE[@]}" --set=--stats 2>&1 | grep -E "depth|ERR|exited|SUMMARY"
for c in 4096 8192; do
    wait_vram
    $SDB run --cfg $CFG --ids wiki-ids.json --out-dir "$OUT" --label P0B-strata-wiki-prefill$c --n-depths 2 "${TUNE[@]}" --set=--prefill=$c 2>&1 | grep -E "depth|ERR|exited|SUMMARY"
done

echo "P0B_DONE $(date +%T)"
