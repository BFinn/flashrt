#!/bin/bash
# sw86: the server end to end on the current engine (sampled drafts at temperature > 0): the
# offline tokenizer check, then flashrt-server + flashrt-engine (--spec 1, 64K context, cache
# prior) as a temporary unit on port 8090, bench/server_smoke.py against it, then stopped
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
P=$BENCH/cache-prior-calib32k.bin
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw86; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram
systemd-run --user --unit=fr-server-sw86 --collect -p MemoryMax=56G bash -c "$S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 1 --engine-arg --draft-vocab --engine-arg $V \
  --engine-arg --cache-prior --engine-arg $P --engine-arg --ctx --engine-arg 65536 > $O/server.log 2>&1"
ok=0
for i in $(seq 450); do
  if curl -sf http://127.0.0.1:8090/v1/models > /dev/null; then ok=1; break; fi
  sleep 2
done
echo "server ready=$ok after $((i * 2)) s"
[ $ok = 1 ] && python3 $FLASHRT/bench/server_smoke.py --url http://127.0.0.1:8090 > $O/smoke.txt 2>&1; echo "smoke rc=$?"
systemctl --user stop fr-server-sw86
echo done > $O/DONE
