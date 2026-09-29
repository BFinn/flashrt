#!/bin/bash
# sw93: phase 1 of docs/improvement-plan.md, the server end to end (as sw86): server_smoke.py
# (with the new sampling-limit checks), then a graceful stop (SIGTERM: the server asks the engine
# to quit). Second run: the engine is killed under a running server; /health and requests must
# answer 503 at once, and the server must exit by itself.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
P=$BENCH/cache-prior-calib32k.bin
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw93; mkdir -p $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
start() {   # unit name, log
  wait_vram
  systemd-run --user --unit=$1 --collect -p MemoryMax=56G bash -c "$S --model $M --port 8090 --engine $B/flashrt-engine \
    --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 1 --engine-arg --draft-vocab --engine-arg $V \
    --engine-arg --cache-prior --engine-arg $P --engine-arg --ctx --engine-arg 65536 > $2 2>&1"
  for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && { echo "$1 ready after $((i * 2)) s"; return 0; }; sleep 2; done
  echo "$1 not ready"; return 1
}
# run 1: smoke, then a graceful stop
if start fr-server-sw93a $O/server-a.log; then
  python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/smoke.txt 2>&1; echo "smoke rc=$?"
  systemctl --user stop fr-server-sw93a
  grep -E "shutting down|engine exited|killing" $O/server-a.log | sed 's/^/  /'
fi
# run 2: kill the engine under the server
if start fr-server-sw93b $O/server-b.log; then
  e=$(pgrep -x flashrt-engine); echo "killing engine pid $e"; kill -9 $e
  sleep 1
  echo "health: $(curl -s -o /dev/null -w '%{http_code}' $URL/health) $(curl -s $URL/health)"
  res=$(curl -s -o $O/after-kill.json -w '%{http_code} in %{time_total} s' -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}' $URL/v1/chat/completions)
  echo "request after the kill: HTTP $res: $(cat $O/after-kill.json)"
  for i in $(seq 20); do systemctl --user is-active --quiet fr-server-sw93b || break; sleep 1; done
  systemctl --user is-active --quiet fr-server-sw93b && { echo "server still running after 20 s"; systemctl --user stop fr-server-sw93b; } \
    || echo "server exited by itself within $i s"
  grep -E "engine exited|engine is down" $O/server-b.log | sed 's/^/  /'
fi
echo done > $O/DONE
