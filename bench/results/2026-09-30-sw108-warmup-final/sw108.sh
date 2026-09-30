#!/bin/bash
# sw108: the expert cache's final warm-up policy end to end (seed scale 0.03, budget 64 uploads
# started per step, commits two steps after issue: deterministic; sw104-sw107). Against sw102
# (the same engine before the warm-up work) and sw105's old-settings agent runs.
# (1) Window 9's protocol with the head: greedy and temperature 1.0, n = 3 each, interleaved.
# (2) The agentic session (agent_trace.py, greedy), 2 runs. (3) server_smoke.py.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw108; mkdir -p $O; cd $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
G(){ $DB --label sw108-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw108.out; }
Sa(){ $DB --label sw108-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw108.out; }
G 1; Sa 1; Sa 2; G 2; G 3; Sa 3
start() {   # unit, log
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --ctx --engine-arg 131072 > $2 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && return 0; sleep 2; done
  echo "$1 not ready"; return 1
}
stop() { systemctl --user stop $1; for i in $(seq 30); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
for r in 1 2; do
  start fr-server-sw108-a$r $O/server-a$r.log || continue
  python3 $FLASHRT/bench/agent_trace.py --repo $FLASHRT --url $URL --label sw108-new-r$r --temperature 0 2>&1 | tee -a $O/agent.out
  [ $r = 2 ] && { python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/smoke.txt 2>&1; echo "smoke rc=$? $(tail -1 $O/smoke.txt)"; }
  stop fr-server-sw108-a$r
done
echo done > $O/DONE
