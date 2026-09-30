#!/bin/bash
# sw105: the expert cache's new defaults from sw104 (seed scale 0.03, swap budget 64) end to end.
# (0) The KLD gate: the fast path's value depends on what the cache holds (sw94).
# (1) Window 9's protocol, MTP greedy, n = 3 (against sw102, the same build otherwise).
# (2) The agentic session (bench/agent_trace.py, as sw97, greedy): the old settings (OLD) against
#     the new defaults, 2 runs each, interleaved. Each tool result refills the cache, so the
#     warm-up repeats every turn.
set -u
OLD="--cache-seed-scale 1 --swap-budget 32"
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw105; mkdir -p $O; cd $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for k in "fast:--fast" "win3:--fast --window 3 --prefill-chunk 2048 --kv-hot 512"; do
  wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 ${k#*:} > $O/kld-${k%%:*}.log 2>&1)
  echo "kld ${k%%:*} $(grep -h 'KLD mean' $O/kld-${k%%:*}.log | awk '{print $3}') $(grep -hoE '[0-9]+ swaps' $O/kld-${k%%:*}.log)" | tee -a $O/sw105-steps.out
done
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
for r in 1 2 3; do $DB --label sw105-G-spec2-greedy-r$r --log $O/G$r.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw105.out; done
start() {   # unit, log, extra engine flags
  local unit=$1 log=$2; shift 2
  local extra=""; for a in "$@"; do extra="$extra --engine-arg $a"; done
  wait_vram
  systemd-run --user --unit=$unit --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 $extra > $log 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && return 0; sleep 2; done
  echo "$unit not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
agent() {   # arm, run, extra engine flags
  local arm=$1 r=$2; shift 2
  start fr-server-sw105-$arm$r $O/server-$arm$r.log "$@" || return
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw105-$arm-r$r --temperature 0 2>&1 | tee -a $O/agent.out
  stop fr-server-sw105-$arm$r
}
agent old 1 $OLD; agent new 1
agent new 2; agent old 2 $OLD
echo done > $O/DONE
