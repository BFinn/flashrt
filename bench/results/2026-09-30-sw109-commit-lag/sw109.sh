#!/bin/bash
# sw109: uploads commit at the next step (lag 1, waiting for them there) instead of two steps later
# (sw107). sw108's agent turns 0-1 (the same prompts and lengths in every run) hit 1-2 points lower
# than the timing-dependent build, which often committed a step earlier. The wait overlaps the token
# just enqueued. (1) The KLD twice with lag 1 (it must repeat). (2) Teacher-forced as sw104, lag 1
# against FLASHRT_COMMIT_LAG=2, alternating, 3 runs each. (3) The agentic session twice with lag 1.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw109; mkdir -p $O; T=$BENCH/sw100
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
k() { local name=$1; shift; wait_vram
      (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast "$@" > $O/kld-$name.log 2>&1)
      echo "$name $(grep -h 'KLD mean' $O/kld-$name.log | awk '{print $3, $5, $7}') $(grep -hoE 'hit rate [0-9.]+%' $O/kld-$name.log) $(grep -hoE '[0-9]+ swaps' $O/kld-$name.log)"; }
k fast-a; k fast-b; k win3 --window 3 --prefill-chunk 2048 --kv-hot 512
run() { local name=$1 lag=$2; shift 2; wait_vram; FLASHRT_COMMIT_LAG=$lag timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-32) $(grep -hoE 'swaps [0-9]+' $O/$name.txt | tail -1)"; }
C="--gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096"
for r in 1 2 3; do
  for lag in 1 2; do
    run w9-lag$lag-r$r $lag --ids $T/w9_teacher.ids --n-prompt 32793 $C
    run wiki-lag$lag-r$r $lag --ids $T/wiki_teacher.ids --n-prompt 32768 $C
    run w9mtp-lag$lag-r$r $lag --ids $T/w9_teacher.ids --n-prompt 32793 $C --mtp $D --spec 2 --draft-vocab $V
  done
done
start() {
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 > $2 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && return 0; sleep 2; done
  echo "$1 not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
for r in 1 2; do
  start fr-server-sw109-a$r $O/server-a$r.log || continue
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw109-lag1-r$r --temperature 0 2>&1 | tee -a $O/agent.out
  stop fr-server-sw109-a$r
done
echo done > $O/DONE
