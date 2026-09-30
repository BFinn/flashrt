#!/bin/bash
# sw117: why flashrt-server's decode slows over many short independent requests (sw116: 131 tok/s on
# the first GSM8K item, ~55 after ~150). The same server and items (the first N, default 200), with
# bench/gsm8k_eval.py keeping each response's timings: the expert cache's hits and misses and the
# drafts proposed and kept, per request.
set -u
N=${N:-200}
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw117; mkdir -p $O
DATA=$BENCH/gsm8k/test.jsonl
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
ready() { for i in $(seq 600); do curl -sf $1 > /dev/null && return 0; sleep 2; done; echo "$1 not ready"; return 1; }
stop() { systemctl --user stop $1; for i in $(seq 60); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
wait_vram
systemd-run --user --unit=fr-server-sw117 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
  --engine-arg --ctx --engine-arg 16384 ${EXTRA:-} > $O/flashrt-server${TAG:-}.log 2>&1"
if ready http://127.0.0.1:8090/v1/models; then
  python3 $FLASHRT/bench/gsm8k_eval.py --data $DATA --n $N --url http://127.0.0.1:8090 --out $O/flashrt${TAG:-}.jsonl --label flashrt${TAG:-}
fi
stop fr-server-sw117
echo ALLDONE
