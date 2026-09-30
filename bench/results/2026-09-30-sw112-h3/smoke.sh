#!/bin/bash
# sw112: the state and server checks on the build without the toggles (after fingerprint.sh):
# engine_smoke --reuse (exact restores) and --faults with the head, then server_smoke.py.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw112/smoke; mkdir -p $O; cd $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$? $(grep -hE 'check\(s\) failed|checks passed|passed' $O/$label.txt | tail -1)"
}
smoke reuse-mtp "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
smoke faults-mtp "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
wait_vram
systemd-run --user --unit=fr-server-sw112 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
  --engine-arg --ctx --engine-arg 131072 > $O/server.log 2>&1"
ok=0; for i in $(seq 450); do curl -sf $URL/v1/models > /dev/null && { ok=1; break; }; sleep 2; done
if [ $ok = 1 ]; then
  python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/server_smoke.txt 2>&1; echo "server_smoke rc=$? $(tail -1 $O/server_smoke.txt)"
else echo "server not ready"; fi
systemctl --user stop fr-server-sw112
echo done > $O/DONE
