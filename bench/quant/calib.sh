#!/usr/bin/env bash
# Calibration window for the per-expert quant search (docs/research/dynamic-quant.md).
# Needs the GPU and nearly all RAM. Run it as its own unit, when the owner frees the box:
#
#   systemd-run --user --unit=bench-quant-calib --collect \
#       bash -c "$FLASHRT/bench/quant/calib.sh > $BENCH/quant-calib/calib.out 2>&1"
#
# Each step is skipped when its output exists, so a rerun resumes. Steps:
#   0  llama-imatrix, built in its own build dir (CPU, about 15 min); the main build is untouched
#   1  calibration and evaluation text (calib_data.py), if not prepared beforehand
#   2  per-domain imatrix with the Q2_0 model: per-expert input sums of squares and routing counts
#   3  per-domain decode routing traces (route_trace), for cache_sim replay with new blob sizes
#   4  super-expert profile: max |ffn_moe_down| per (layer, expert) on a mixed 1K-token prompt
#   5  IQ3_S reference logits (KLD base) for the search and hold-out sets
#   6  Q2_0 baseline KLD against both references: llama-perplexity, and flashrt's fr_kld if built
# The IQ3_S step keeps its 50 GB of experts in the page cache (mmap), so it needs about 57 GB
# available: nothing else heavy may run in the window.
set -uo pipefail
source $FLASHRT/bench/p0/lib.sh           # LLAMA, FR, MODEL, step, need_ram, take_gpu

D=$BENCH/quant-calib
DATA=$D/data
IQ3S=$DATA/flashnext-gsq-iq3_s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
LLAMA_SRC=$LLAMA_CPP
IMB=$LLAMA_SRC/build-imatrix
DOMAINS="general chat code agentic multilingual"
CTX=2048
SEARCH_CHUNKS=16
HOLDOUT_CHUNKS=8
export HF_HOME=$D/hf-cache                      # ~/.cache/huggingface is a dangling link
mkdir -p "$D/trace" "$D/kld"

# The downloads share the USB disk the IQ3_S reference is read from: pause them for the window.
DL_PAUSED=""
if systemctl --user is-active -q dl-iq3s; then echo "the IQ3_S download is still running; not starting"; exit 1; fi
if [ ! -e "$IQ3S" ]; then echo "no IQ3_S link at $IQ3S (download not finished?); not starting"; exit 1; fi
resume_dl() {
    [ -n "$DL_PAUSED" ] || return 0
    systemd-run --user --unit=dl-bf16 --collect -p Nice=19 -p IOSchedulingClass=idle \
        bash -c "$BENCH/quant-dl/run_bf16.sh >> $BENCH/quant-dl/bf16.log 2>&1" && echo "RESUMED dl-bf16 $(date +%T)"
}
if systemctl --user is-active -q dl-bf16; then systemctl --user stop dl-bf16; DL_PAUSED=1; echo "PAUSED dl-bf16"; fi

NEED_RAM_GB=40 take_gpu
prev_trap=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/")
trap "resume_dl; $prev_trap" EXIT

LOAD=(-ngl 99 -ot "ffn_.*_exps=CPU" -fa on -t 12)

step "0 llama-imatrix"
if [ ! -x "$IMB/bin/llama-imatrix" ]; then
    cmake -S "$LLAMA_SRC" -B "$IMB" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DLLAMA_CURL=OFF >/dev/null &&
        cmake --build "$IMB" --target llama-imatrix -j 20 | tail -2 || { echo "imatrix build failed"; exit 1; }
fi

step "1 calibration data"
if [ ! -f "$DATA/manifest.json" ]; then
    nice -n 10 python3 $FLASHRT/bench/quant/calib_data.py "$DATA" || { echo "data prep failed"; exit 1; }
fi
cat "$DATA/manifest.json"

step "2 imatrix per domain (Q2_0)"
for d in $DOMAINS; do
    out=$D/imatrix-$d.gguf
    [ -f "$out" ] && { echo "have $out"; continue; }
    wait_vram
    t0=$(date +%s)
    "$IMB/bin/llama-imatrix" -m "$MODEL" -f "$DATA/calib-$d.txt" -o "$out.tmp.gguf" "${LOAD[@]}" -lm dio \
        -c $CTX -b $CTX -ub $CTX --parse-special --no-ppl 2>&1 | grep -E -i "error|chunks|tokeniz|saving|stored" | tail -6
    [ -f "$out.tmp.gguf" ] && mv "$out.tmp.gguf" "$out" && echo "$d: $(( $(date +%s) - t0 )) s"
done

step "3 decode routing traces"
for d in $DOMAINS; do
    [ -f "$D/trace/$d.decode_topk.npy" ] && { echo "have trace $d"; continue; }
    wait_vram
    "$FR/route_trace" --model "$MODEL" --text "$DATA/calib-$d.txt" --n-prompt 8192 --gen 1024 --probs \
        --temp 1.0 --top-k 20 --top-p 0.95 --seed 1 --ctx 10240 --out "$D/trace/$d" 2>&1 | tail -2
done

step "4 super-expert profile"
if [ ! -f "$D/superexperts.json" ]; then
    mix=$D/trace/mix.txt
    for d in $DOMAINS; do head -c 1200 "$DATA/calib-$d.txt"; echo; done > "$mix"
    "$FR/route_trace" --model "$MODEL" --text "$mix" --n-prompt 1024 --tokenize-only --out "$D/trace/mix" 2>&1 | tail -1
    wait_vram
    "$FR/ref_dump" --model "$MODEL" --ids "$D/trace/mix.prompt_ids.txt" --n-prompt 1024 --gen 1 --ctx 2048 \
        --capture ffn_moe_down,ffn_moe_topk --out "$D/trace/mix.frd" 2>&1 | tail -1
    python3 $FLASHRT/bench/quant/superexp.py "$D/trace/mix.frd" > "$D/superexperts.json" &&
        rm -f "$D/trace/mix.frd"
    python3 -c "import json; t=json.load(open('$D/superexperts.json'))['top'][:10]; [print(r) for r in t]"
fi

step "5 IQ3_S reference logits"
for set in search holdout; do
    base=$D/kld/ref-$set.kld
    [ -f "$base" ] && { echo "have $base"; continue; }
    [ $set = search ] && chunks=$SEARCH_CHUNKS || chunks=$HOLDOUT_CHUNKS
    wait_vram
    need_ram 57
    # Sequential read of the USB copy fills the page cache once, before random access starts.
    dd if="$(readlink -f "$IQ3S")" of=/dev/null bs=16M status=none
    "$LLAMA/llama-perplexity" -m "$IQ3S" -f "$DATA/eval-$set.txt" --kl-divergence-base "$base.tmp" \
        "${LOAD[@]}" -c $CTX -b $CTX -ub $CTX --chunks $chunks 2>&1 | grep -E "Final estimate|error|chunks" | tail -3
    [ -f "$base.tmp" ] && mv "$base.tmp" "$base"
done

step "6 Q2_0 baseline KLD"
for set in search holdout; do
    [ -f "$D/kld/q2_0-$set.txt" ] && { echo "have q2_0-$set"; continue; }
    wait_vram
    "$LLAMA/llama-perplexity" -m "$MODEL" --kl-divergence-base "$D/kld/ref-$set.kld" --kl-divergence \
        "${LOAD[@]}" -lm dio -c $CTX -b $CTX -ub $CTX 2>&1 | sed -n '/KL divergence statistics/,$p' > "$D/kld/q2_0-$set.txt"
    head -12 "$D/kld/q2_0-$set.txt"
done
if [ -x "$FR/fr_kld" ] && [ ! -f "$D/kld/fr-q2_0-search.txt" ]; then
    wait_vram
    "$FR/fr_kld" "$MODEL" "$D/kld/ref-search.kld" --ctx $CTX --chunks $SEARCH_CHUNKS > "$D/kld/fr-q2_0-search.txt" 2>&1
    tail -4 "$D/kld/fr-q2_0-search.txt"
fi

step "done"
du -sh "$D"/imatrix-*.gguf "$D"/kld/* "$D"/trace 2>/dev/null
