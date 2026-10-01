#!/bin/bash
# sw119: the headline numbers after the expert-cache leak fix (sw118).
# (1) The agent session (bench/agent_trace.py, greedy), the engine before the fix (built from
#     749cb02^, exported, not a git worktree) against after, 2 runs each, alternating: agent runs read
#     this repository's source, so both arms run on the same state.
# (2) Window 9's protocol as sw110 (the same script's arms), the head greedy (G) and temperature 1.0
#     (S), 5 runs each, interleaved; the plain arm (P, no windows, so no leak) 2 runs as a control.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw119; mkdir -p $O; cd $O
URL=http://127.0.0.1:8090
# the engine before the fix
if [ ! -x $O/old/build/flashrt-engine ]; then
  rm -rf $O/old && mkdir -p $O/old && git -C $FLASHRT archive 749cb02^ | tar -x -C $O/old
  cmake -S $O/old -B $O/old/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DFLASHRT_NATIVE=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc > $O/old-build.log 2>&1
  cmake --build $O/old/build --target flashrt-engine >> $O/old-build.log 2>&1 || { echo "old build failed"; exit 1; }
fi
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
start() {   # unit, engine, log
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $2 \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 > $3 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && return 0; sleep 2; done
  echo "$1 not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
agent() {   # arm, engine, run
  start fr-server-sw119-$1$3 $2 $O/server-$1$3.log || return
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw119-$1-r$3 --temperature 0 2>&1 | tee -a $O/agent.out
  stop fr-server-sw119-$1$3
}
agent old $O/old/build/flashrt-engine 1; agent new $FLASHRT/build/flashrt-engine 1
agent new $FLASHRT/build/flashrt-engine 2; agent old $O/old/build/flashrt-engine 2
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $FLASHRT/build/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
P(){ $DB --label sw119-P-plain-r$1 --log $O/P$1.log 2>&1 | tee -a $O/sw119.out; }
G(){ $DB --label sw119-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw119.out; }
Sa(){ $DB --label sw119-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw119.out; }
P 1; G 1; Sa 1; G 2; Sa 2; P 2; Sa 3; G 3; G 4; Sa 4; Sa 5; G 5
echo ALLDONE
