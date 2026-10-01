#!/bin/bash
# sw128: P-5 closed (the select cluster, the parallel CLOCK, the argmax cluster, fp16 pooled keys).
# First the server check (the pooled keys' VRAM changed): flashrt-server + flashrt-engine as a
# temporary unit, bench/server_smoke.py against it (as sw86). Then window 9's protocol as sw124:
# P no head, G head greedy, S head at temperature 1.0; 5 runs each, interleaved.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
PR=$BENCH/cache-prior-calib32k.bin
B=$FLASHRT/build; SV=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw128; mkdir -p $O; cd $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram
systemd-run --user --unit=fr-server-sw128 --collect -p MemoryMax=56G bash -c "$SV --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
  --engine-arg --cache-prior --engine-arg $PR --engine-arg --ctx --engine-arg 65536 > $O/server.log 2>&1"
ok=0
for i in $(seq 450); do
  if curl -sf http://127.0.0.1:8090/v1/models > /dev/null; then ok=1; break; fi
  sleep 2
done
echo "server ready=$ok after $((i * 2)) s"
[ $ok = 1 ] && python3 $FLASHRT/bench/server_smoke.py --url http://127.0.0.1:8090 > $O/smoke.txt 2>&1; echo "smoke rc=$?"
systemctl --user stop fr-server-sw128
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
P(){ wait_vram; $DB --label sw128-P-plain-r$1 --log $O/P$1.log 2>&1 | tee -a $O/sw128.out; }
G(){ wait_vram; $DB --label sw128-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw128.out; }
S(){ wait_vram; $DB --label sw128-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw128.out; }
P 1; G 1; S 1; G 2; S 2; P 2; S 3; P 3; G 3; P 4; G 4; S 4; G 5; S 5; P 5
echo ALLDONE
