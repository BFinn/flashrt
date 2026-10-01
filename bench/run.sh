#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One-command reproduction on another machine, in plain bash (phase 6, X-2). The scripts in
# bench/results run as systemd units on the development box; this one needs none of that.
#
#   MODELS=/path/to/models bench/run.sh [preflight|build|test|decode|window9|kld|server|all]
#
# MODELS  the directory with Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf and the MTP
#         draft head mtp-Flash-Next-Q8_0-noembd.gguf (required)
# BENCH   scratch for outputs (default ./bench-out)
# DATA    for `kld` only: the directory holding wikitext-2-raw/wiki.test.raw
# LLAMA_CPP  for `kld` only, when $BENCH/kld/kl8k-f16.bin is missing: a llama.cpp tree at
#         ec92815 with bench/reference/llama.cpp-flashnext.patch applied and built
# RUNS    window 9's runs per arm (default 1; the published tables use 5)
# CUDACXX nvcc (default /usr/local/cuda/bin/nvcc)
#
# `all` is preflight, build, test, decode, window9 and server (about an hour, most of it window
# 9's prefills). Each step prints what it measured and the published value to compare against.
set -u
cd "$(dirname "$0")/.." || exit 1
FR=$PWD
: "${MODELS:?set MODELS to the directory with the model GGUF shards and the MTP head}"
BENCH=${BENCH:-$FR/bench-out}
RUNS=${RUNS:-1}
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$FR/bench/reference/mtp-vocab-ranks.txt
IDS=$FR/bench/reference/w9-ids.json
mkdir -p "$BENCH"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"   # rustup's default install

say() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*"; }

# the previous loader's VRAM can take a few seconds to come back
wait_vram() {
    for _ in $(seq 150); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 600 ] && return 0
        sleep 2
    done
    echo "VRAM still in use: stop other GPU processes first"; exit 1
}

preflight() {
    say "preflight: what this machine has, against what flashrt was built and measured on"
    local ok=1
    if ! command -v nvidia-smi > /dev/null; then echo "no nvidia-smi: an NVIDIA GPU and driver are required"; return 1; fi
    IFS=, read -r name mem cap < <(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1)
    echo "GPU: $name,${mem} MiB, compute capability${cap} (measured: RTX 5080 16 GB, 12.0)"
    if awk -v c="$cap" 'BEGIN { exit !(c + 0 < 9.0) }'; then echo "  needs compute capability 9.0 or newer (clusters, PDL)"; ok=0; fi
    [ "${mem// /}" -lt 15000 ] && { echo "  needs 16 GB of VRAM"; ok=0; }
    awk -v c="$cap" 'BEGIN { exit !(c + 0 != 12.0) }' && warn "built and tuned for 12.0; set CMAKE_CUDA_ARCHITECTURES for this GPU (untested)"
    local ram_gb dev rota
    ram_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
    echo "RAM: ${ram_gb} GB (measured: 64 GB; the experts take ~55 GB of host RAM)"
    [ "$ram_gb" -lt 60 ] && { echo "  needs 64 GB"; ok=0; }
    if grep -q avx512_vnni /proc/cpuinfo; then echo "CPU: AVX-512 VNNI (the CPU expert kernels' path)"
    else warn "no AVX-512 VNNI: CPU misses take the scalar path, and decode will be much slower"; fi
    echo "CPU threads: $(nproc) (measured: 12 cores / 24 threads; the miss pool uses 8 workers)"
    for f in "$M" "${M/00001-of/00002-of}" "$D"; do [ -f "$f" ] || { echo "missing: $f"; ok=0; }; done
    if [ -f "$M" ]; then
        dev=$(findmnt -no SOURCE --target "$MODELS" 2>/dev/null)
        rota=$(lsblk -no ROTA "$dev" 2>/dev/null | head -1 | tr -d ' ')
        echo "model storage: $dev (rotational: ${rota:-unknown}); the n-gram (PLE) rows are read per token with O_DIRECT, so use an NVMe drive"
        [ "${rota:-0}" = 1 ] && warn "a rotational disk will stall every decode step"
    fi
    local missing=""
    for t in cmake ninja g++ cargo python3 curl "${CUDACXX:-/usr/local/cuda/bin/nvcc}"; do command -v "$t" > /dev/null || missing="$missing $t"; done
    [ -n "$missing" ] && { echo "missing tools:$missing (CUDA 12.8+, CMake 3.28+, Ninja, GCC 12+, Rust 1.82+)"; ok=0; }
    [ $ok = 1 ] && echo "preflight: ok" || { echo "preflight: FAILED"; return 1; }
}

build() {
    say "build"
    local nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120}" -DFLASHRT_NATIVE=ON \
          -DCMAKE_CUDA_COMPILER="$nvcc" > "$BENCH/build.log" 2>&1 &&
        cmake --build build >> "$BENCH/build.log" 2>&1 &&
        cargo build --release --manifest-path server/Cargo.toml >> "$BENCH/build.log" 2>&1 ||
        { echo "build failed: see $BENCH/build.log"; return 1; }
    echo "built (log: $BENCH/build.log)"
}

tests() {
    say "tests (the CUDA kernels against CPU references; two of them load the model)"
    (cd build && FLASHRT_TEST_MODEL="$M" ctest --output-on-failure) > "$BENCH/ctest.log" 2>&1
    local rc=$?
    grep -E "tests passed|tests failed" "$BENCH/ctest.log"
    [ $rc = 0 ] || echo "see $BENCH/ctest.log"
    return $rc
}

decode() {
    say "decode: fr_bench on window 9's 32K prompt, fresh prefill, 256 tokens, greedy (the reference box, 2026-10-01: prefill 5,958 tok/s; decode 95.0 plain, 97.4 with --spec 2)"
    python3 - "$IDS" "$BENCH/w9-32k.ids" << 'EOF'
import json, sys
item = next(i for i in json.load(open(sys.argv[1])) if 30000 < len(i["ids"]) < 40000)
open(sys.argv[2], "w").write(" ".join(map(str, item["ids"])))
EOF
    local n
    n=$(wc -w < "$BENCH/w9-32k.ids")
    local C=(--ids "$BENCH/w9-32k.ids" --n-prompt "$n" --gen 256 --prefill-chunk auto --kv q8 --kv-hot 4096)
    wait_vram
    build/fr_bench "$M" "${C[@]}" > "$BENCH/decode-plain.txt" 2>&1
    grep -hE "^prefill:|^decode:|hit rate" "$BENCH/decode-plain.txt"
    wait_vram
    build/fr_bench "$M" "${C[@]}" --mtp "$D" --spec 2 --draft-vocab "$V" > "$BENCH/decode-spec2.txt" 2>&1
    grep -hE "^decode:|hit rate|speculative" "$BENCH/decode-spec2.txt"
}

window9() {
    say "window 9 (the reference engines' prompts, 1K / 32K / 134K / 250K in one conversation, 384 tokens each), $RUNS run(s) per arm"
    local O=$BENCH/window9 DB=(python3 bench/flashrt_depthbench.py --no-scope --engine build/flashrt-engine --model "$M" --ids "$IDS" --gen 384)
    mkdir -p "$O"
    : > "$O/w9.out"
    for r in $(seq "$RUNS"); do
        wait_vram; "${DB[@]}" --label run-P-plain-r$r --log "$O/P$r.log" | tee -a "$O/w9.out"
        wait_vram; "${DB[@]}" --label run-G-spec2-greedy-r$r --log "$O/G$r.log" -- --mtp "$D" --spec 2 --draft-vocab "$V" | tee -a "$O/w9.out"
        wait_vram; "${DB[@]}" --label run-S-spec2-t1.0-r$r --log "$O/S$r.log" --sampling "temperature=1.0 top_p=0.95 top_k=20" \
            -- --mtp "$D" --spec 2 --draft-vocab "$V" | tee -a "$O/w9.out"
    done
    python3 bench/depthsum.py "$O/w9.out" | tee "$O/summary.txt" | head -6
    echo "published (sw128, n=5): P 103.5 / 95.5 / 93.9 / 91.9, G 127.9 / 111.3 / 103.7 / 104.9, S 118.8 / 95.5 / 106.7 / 106.4"
}

kld() {
    say "KLD gate: the fast path against llama.cpp's FP16-KV reference (published band 0.0087-0.0092)"
    if [ ! -f "$BENCH/kld/kl8k-f16.bin" ]; then
        : "${LLAMA_CPP:?the KLD base is missing: set LLAMA_CPP (and DATA) to build it, see bench/reference/README.md}"
        : "${DATA:?set DATA to the directory holding wikitext-2-raw/}"
        wait_vram; MODELS=$MODELS BENCH=$BENCH DATA=$DATA LLAMA_CPP=$LLAMA_CPP bash bench/reference/kld-base.sh || return 1
    fi
    wait_vram
    (cd "$BENCH/kld" && "$FR/build/fr_kld" "$M" kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast) > "$BENCH/kld-fast.log" 2>&1
    grep -h "KLD mean\|same top" "$BENCH/kld-fast.log"
}

server() {
    say "server: flashrt-server with the engine, then bench/server_smoke.py against it"
    local port=${PORT:-8090}
    wait_vram
    server/target/release/flashrt-server --model "$M" --port "$port" --engine build/flashrt-engine --engine-arg "$M" \
        --engine-arg --mtp --engine-arg "$D" --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg "$V" \
        --engine-arg --ctx --engine-arg 65536 > "$BENCH/server.log" 2>&1 &
    local pid=$! ok=0
    for _ in $(seq 450); do
        curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && { ok=1; break; }
        kill -0 $pid 2> /dev/null || break
        sleep 2
    done
    local rc=1
    [ $ok = 1 ] && { python3 bench/server_smoke.py --url "http://127.0.0.1:$port" | tail -3; rc=${PIPESTATUS[0]}; }
    [ $ok = 1 ] || echo "the server did not come up: see $BENCH/server.log"
    kill $pid 2> /dev/null; wait $pid 2> /dev/null
    return $rc
}

case "${1:-all}" in
    preflight) preflight ;;
    build) build ;;
    test) tests ;;
    decode) decode ;;
    window9) window9 ;;
    kld) kld ;;
    server) server ;;
    all) preflight && build && tests && decode && window9 && server ;;
    *) sed -n 3,20p "$0"; exit 2 ;;
esac
