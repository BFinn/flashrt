#!/bin/bash
# sw132: checks after the known issues' fixes: the expert histogram and the Q3_K embedding decode
# shared (outputs must not change: sw112's fingerprint against sw131's), the model load's cleanup,
# the CUDA tests' checks, and the server's special-token handling and API shapes (bench/server_smoke.py
# against a server with an API key; engine_smoke's default mode).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$FLASHRT/bench/reference/mtp-vocab-ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw132; mkdir -p $O; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cmake --build build > $O/build.log 2>&1 && (source ~/.cargo/env; cargo build --release --locked --manifest-path server/Cargo.toml >> $O/build.log 2>&1) \
  || { echo "build failed"; exit 1; }
echo "build: $(grep -c warning $O/build.log) warnings"
(cd build && FLASHRT_TEST_MODEL=$M ctest > $O/ctest.txt 2>&1); echo "ctest: $(grep -E 'tests passed|tests failed' $O/ctest.txt)"
if [ -z "${SKIP_FP:-}" ]; then
  bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw132 > $O/fp-new.txt 2>&1
  diff $BENCH/sw131/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
fi
wait_vram
python3 bench/engine_smoke.py $B/flashrt-engine $M --ids $I --n 4096 --gen 32 -- --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 \
  --mtp $D --spec 2 --draft-vocab $V > $O/engine-default.txt 2>&1
echo "engine_smoke rc=$? $(grep -c '^PASS' $O/engine-default.txt) pass"
wait_vram
KEY=sw132-$(date +%s)
systemd-run --user --unit=fr-server-sw132 --collect -p MemoryMax=56G bash -c "$FLASHRT/server/target/release/flashrt-server --model $M --port 8090 \
  --api-key $KEY --engine $B/flashrt-engine --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 \
  --engine-arg --draft-vocab --engine-arg $V --engine-arg --ctx --engine-arg 65536 > $O/server.log 2>&1"
ok=0
for i in $(seq 450); do curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:8090/v1/models > /dev/null && { ok=1; break; }; sleep 2; done
echo "server ready=$ok"
[ $ok = 1 ] && python3 bench/server_smoke.py --url http://127.0.0.1:8090 --key $KEY > $O/smoke.txt 2>&1; echo "smoke rc=$? $(grep -c '^ok' $O/smoke.txt) ok, $(grep -c '^FAIL' $O/smoke.txt) fail"
systemctl --user stop fr-server-sw132
echo ALLDONE
