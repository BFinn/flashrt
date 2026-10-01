#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# A record of a run on the development box (its systemd units, services and paths), not a portable
# script: bench/run.sh is the entry point for another machine.
# HISTORICAL (window A, 2026-09-27). Do not rerun as is: it restarts flashnext-262k-server on
# exit, and no live service should run now. New windows source bench/p0/lib.sh instead.
# P0 window A on the target box: bandwidth probes, routing traces, llama.cpp prefill ubatch
# sweep. Stops the live server for the duration and restarts it on exit.
#
#   systemd-run --user --unit=bench-p0a --collect bash -c "$FLASHRT/bench/p0/window_a.sh > $BENCH/p0a.out 2>&1"
#
# Needs: flashrt built with -DFLASHRT_LLAMA_DIR=$LLAMA_CPP (for route_trace).
set -u
OUT=${OUT:-$BENCH/p0a-$(date +%Y%m%d-%H%M)}
FR=$FLASHRT/build
LLAMA=$LLAMA_CPP/build/bin
MODEL=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
IDS=$BENCH/strata-ids.json
LIVE=flashnext-262k-server
GEN=${GEN:-2048}

mkdir -p "$OUT" && cd "$OUT" || exit 1
step() { echo; echo "== $* ($(date +%T))"; }

# ---- the live server: refuse if it is busy, stop it, restart it on any exit
# LIVE_URL: the live llama-server's address; LIVE_KEY_FILE: a file holding its API key
key=$(cat "$LIVE_KEY_FILE")
if curl -s -m 5 -H "Authorization: Bearer $key" "$LIVE_URL/slots" | grep -q '"is_processing":true'; then
    echo "live server is processing a request; not starting"
    exit 1
fi
restore() { systemctl --user start $LIVE; echo "RESTARTED_LIVE $(date +%T) $(systemctl --user is-active $LIVE)"; }
trap restore EXIT
systemctl --user stop $LIVE
wait_vram() {
    for _ in $(seq 120); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 600 ] && return 0
        sleep 1
    done
    echo "VRAM still in use after 120 s"; exit 1
}
wait_vram
echo "START $(date +%T) out=$OUT"

# ---- 1. host DRAM read bandwidth
step membw
cat /sys/kernel/mm/transparent_hugepage/enabled /proc/sys/vm/nr_hugepages
for pages in 4k thp; do
    for t in 1 6 12; do "$FR/membw" --threads $t --pages $pages --gb 8 --seconds 5; done
done | tee membw.txt

# ---- 2. host->device bandwidth, alone and with CPU readers competing for DRAM
step h2dbw
{
    "$FR/h2dbw" --pages 4k
    "$FR/h2dbw" --pages thp
    "$FR/h2dbw" --pages thp --cpu-threads 6
    "$FR/h2dbw" --pages thp --cpu-threads 12
} | tee h2dbw.txt

# ---- 3. routing traces at 1K and 32K depth, sampled like the target (t=1.0, top_p 0.95, top_k 20)
step route_trace
for d in 1000 32000; do
    python3 -c "import json,sys; d=[x for x in json.load(open('$IDS')) if x['depth']==$d][0]; print(' '.join(map(str,d['ids'])))" > ids_$d.txt
    wait_vram
    qsa=""; [ $d -ge 32000 ] && qsa=--qsa        # QSA selection is only non-trivial beyond ~2K cells
    "$FR/route_trace" --model "$MODEL" --ids ids_$d.txt --gen $GEN --out trace_d$d --probs --prefill-trace $qsa \
        --temp 1.0 --top-p 0.95 --top-k 20 --seed 42
done

# ---- 4. llama.cpp prefill vs ubatch at 32K (is prefill transfer-bound, and do bigger chunks help?)
step llama-bench ubatch sweep
wait_vram
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -ot "ffn_.*_exps=CPU" -fa 1 -ctk q8_0 -ctv q8_0 -t 12 \
    -p 32768 -n 0 -b 8192 -ub 2048,4096,8192 -r 1 -o md | tee ubatch.md

echo "P0A_DONE $(date +%T)"
