#!/bin/bash
# sw111: the cache's fill weighs the prompt's last 16 tokens. sw99's simulator under the current
# policy: adding their routing at 2x the prompt's total lifted window 9's first 64 tokens from 67% to
# 77% hits (84.1 -> 85.9% overall) and cost wikitext 0.2 points. Short agent turns are mostly such
# first tokens. (1) Teacher-forced as sw104 (fr_bench, --cache-tail-weight 0 / 0.5 / 2, rotating,
# 3 runs each). (2) The agentic session through the server, weight 0 against 2, 2 runs each.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw111; mkdir -p $O; T=$BENCH/sw100
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
C="--gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096"
W=(0 0.5 2)
for r in 1 2 3; do
  for w in "${W[@]}"; do
    run w9-w$w-r$r --ids $T/w9_teacher.ids --n-prompt 32793 $C --cache-tail-weight $w
    run wiki-w$w-r$r --ids $T/wiki_teacher.ids --n-prompt 32768 $C --cache-tail-weight $w
    run w9mtp-w$w-r$r --ids $T/w9_teacher.ids --n-prompt 32793 $C --mtp $D --spec 2 --draft-vocab $V --cache-tail-weight $w
  done
  W=("${W[@]:1}" "${W[0]}")
done
start() {   # unit, log, tail weight
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 --engine-arg --cache-tail-weight --engine-arg $3 > $2 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && return 0; sleep 2; done
  echo "$1 not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
for arm in "w0 0 1" "w2 2 1" "w2 2 2" "w0 0 2"; do
  set -- $arm
  start fr-server-sw111-$1-$3 $O/server-$1-$3.log $2 || continue
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw111-$1-r$3 --temperature 0 2>&1 | tee -a $O/agent.out
  stop fr-server-sw111-$1-$3
done
echo done > $O/DONE
