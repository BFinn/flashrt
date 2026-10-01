#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# A record of a run on the development box (its systemd units, services and paths), not a portable
# script: bench/run.sh is the entry point for another machine.
# P0 window C on the target box, all on natural text (wikitext-2):
#   1. Strata's own time breakdown at 32K: `--stats` and `--gpu-stages` (per-stage GPU
#      table), in generate mode; then PLE read depth (--ple-inflight) and --ple-io mmap
#   2. llama.cpp prefill vs ubatch at 32K with direct-I/O loading (pinned, no mmap)
#   3. the reference baselines on wikitext at 1K / 32K / 131K / 245K, interleaved:
#      Strata tuned greedy x2, Strata tuned t=1.0 x2, llama.cpp dev tree x1
#
#   systemd-run --user --unit=bench-p0c --collect bash -c "$FLASHRT/bench/p0/window_c.sh > $BENCH/p0c.out 2>&1"
set -u
source "$(dirname "$0")/lib.sh"
OUT=${OUT:-$BENCH/p0c-$(date +%Y%m%d-%H%M)}
SDB="python3 $FLASHRT/bench/strata_depthbench.py"
DB="python3 $FLASHRT/bench/depthbench.py"
CFG=$STRATA/strata-q2_0.json
TUNE=(--vram-reserve-mib 1024 --set=--pcie-frac=0.35 --set=--pool-workers=8)
mkdir -p "$OUT" && cd "$OUT" || exit 1

# ---- 0. natural-text prompts to 245K (vocabulary only; no GPU)
step tokenize wikitext
cat $DATA/wikitext-2-raw/wiki.test.raw $DATA/wikitext-2-raw/wiki.valid.raw > wiki.txt
"$FR/route_trace" --model "$MODEL" --text wiki.txt --n-prompt 250000 --tokenize-only --out wiki 2>&1 | grep tokenized || exit 1
python3 - <<'EOF'
import json
ids = list(map(int, open("wiki.prompt_ids.txt").read().split()))
json.dump([{"depth": d, "ids": ids[:d]} for d in (1000, 32000, 131000, 245000)], open("wiki-ids.json", "w"))
open("wiki_32000.txt", "w").write(" ".join(map(str, ids[:32000])))
print(f"wiki prompt: {len(ids)} tokens")
EOF

take_gpu
echo "out=$OUT"

# ---- 1. Strata generate mode on the 32K prompt: its own breakdowns
mapfile -t SARGS < <(python3 -c "import json; print('\n'.join(json.load(open('$CFG'))['args']))")
strata_gen() {
    local label=$1; shift
    wait_vram
    ( cd $STRATA && LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 timeout 900 \
        systemd-run --user --scope --quiet -p MemoryMax=56G -p MemorySwapMax=0 -- \
        engine/strata "${SARGS[@]}" --tokens-file "$OUT/wiki_32000.txt" --max-new 384 \
        --vram-reserve-mib 1024 --pcie-frac 0.35 --pool-workers 8 "$@" ) > "$label.log" 2>&1
    echo "$label: exit $? ($(wc -l < "$label.log") log lines)"
}
step strata generate breakdowns
if [ "${SKIP_DONE:-0}" != 1 ]; then
    strata_gen gen-stats --stats
fi
# --gpu-only-full was OOM-killed at MemoryMax=56G in the first attempt; not retried.
strata_gen gen-stages --gpu-stages
# prefill spent 21.6 of 47.4 s blocked on PLE (n-gram table) SSD reads at 64 in flight
strata_gen gen-ple-inflight256 --stats --ple-inflight 256
strata_gen gen-ple-inflight1024 --stats --ple-inflight 1024
strata_gen gen-ple-mmap --stats --ple-io mmap

# ---- 2. llama.cpp prefill vs ubatch at 32K, direct-I/O load (pinned host weights)
step llama-bench ubatch sweep, -lm dio
wait_vram
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -ot "ffn_.*_exps=CPU" -fa 1 -ctk q8_0 -ctv q8_0 -t 12 -lm dio \
    -p 32768 -n 0 -b 8192 -ub 2048,4096,8192 -r 1 -o md | tee ubatch_dio.md

# ---- 3. baselines on natural text, interleaved
S()  { wait_vram; $SDB run --cfg $CFG --ids wiki-ids.json --out-dir "$OUT" --label P0C-strata-greedy-r$1 "${TUNE[@]}" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"; }
ST() { wait_vram; $SDB run --cfg $CFG --ids wiki-ids.json --out-dir "$OUT" --label P0C-strata-t1.0-r$1 "${TUNE[@]}" \
         --sampling "temperature=1.0 top_p=0.95 top_k=20" 2>&1 | grep -E "depth|ERR|exited|SUMMARY"; }
L()  { wait_vram; $DB --label P0C-llamacpp-r$1 --bin $LLAMA --ids wiki-ids.json --gen 384 --out "$OUT" \
         --env LLAMA_MOE_CACHE_ADMIT=3 --env LLAMA_MOE_CACHE_WINDOW=32 -- --no-warmup -lm dio -ngl 99 \
         -ot "ffn_.*_exps=CPU" -fa on -ctk q8_0 -ctv q8_0 -c 262144 -b 4096 -ub 2048 -t 12 -tb 12 --jinja \
         --parallel 1 --moe-expert-cache 48 --moe-expert-cache-inserts 2 2>&1 | grep -E "depth|died|SUMMARY"; }
step baselines
S 1; L 1; ST 1; S 2; ST 2

echo "P0C_DONE $(date +%T)"
