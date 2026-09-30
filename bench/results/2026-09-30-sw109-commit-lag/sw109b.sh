#!/bin/bash
# sw109b: server_smoke.py on the final build (commit lag 1 fixed, its toggle removed).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw109; mkdir -p $O
URL=http://127.0.0.1:8090
for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && break; sleep 2; done
systemd-run --user --unit=fr-server-sw109s --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V > $O/server-smoke.log 2>&1"
for i in $(seq 300); do curl -sf $URL/v1/models > /dev/null && break; sleep 2; done
python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/smoke.txt 2>&1; echo "smoke rc=$? $(tail -1 $O/smoke.txt)"
systemctl --user stop fr-server-sw109s
