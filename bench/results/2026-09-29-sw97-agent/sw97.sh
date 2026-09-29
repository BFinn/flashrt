#!/bin/bash
# sw97: the server end to end on the current build (server_smoke.py, now with the timings check),
# then the agentic workload (bench/agent_trace.py, phase 2 of docs/improvement-plan.md): a
# 12-turn review session over this repository with read_file / list_dir tools, reasoning on,
# 512 tokens per turn. A fresh server (and engine) per run, arms interleaved, 3 runs each:
#   g  greedy
#   t  temperature 0.6, top_p 0.95, top_k 20 (Qwen's thinking-mode settings; sampled drafts)
# Engine: --mtp --spec 2 --draft-vocab, --ctx 131072, defaults otherwise (8 host checkpoints,
# swap budget 32, no cache prior).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw97; mkdir -p $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
start() {   # unit name, log
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 > $2 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && { echo "$1 ready after $((i * 2)) s"; return 0; }; sleep 2; done
  echo "$1 not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
if start fr-server-sw97s $O/server-smoke.log; then
  python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/smoke.txt 2>&1; echo "smoke rc=$?"
  stop fr-server-sw97s
fi
run() {   # arm, run, agent_trace args...
  local arm=$1 r=$2; shift 2
  start fr-server-sw97-$arm$r $O/server-$arm$r.log || return
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw97-$arm-r$r "$@" 2>&1 | tee -a $O/sw97.out
  stop fr-server-sw97-$arm$r
}
g() { run g $1 --temperature 0; }
t() { run t $1 --temperature 0.6; }
g 1; t 1
t 2; g 2
g 3; t 3
echo done > $O/DONE
