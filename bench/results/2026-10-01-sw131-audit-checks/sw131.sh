#!/bin/bash
# sw131: the checks after the code audit's fixes (error paths, state files, graph capture, quit,
# the server's request checks and disconnect handling). Build; sw112's fingerprint (must equal
# sw130's: the fixes change no output); the 245K state loaded teacher-forced, plain and --spec 2
# (state_file rewritten: its swaps and hits must equal sw126's runs exactly); engine_smoke in its
# three modes; the server with an API key and bench/server_smoke.py against it.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$FLASHRT/bench/reference/mtp-vocab-ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw131; mkdir -p $O; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
cmake --build build > $O/build.log 2>&1 && (source ~/.cargo/env; cargo build --release --locked --manifest-path server/Cargo.toml >> $O/build.log 2>&1) \
  || { echo "build failed"; exit 1; }
echo "build: $(grep -c warning $O/build.log) warnings"
if [ -z "${SKIP_FP:-}" ]; then
  bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw131 > $O/fp-new.txt 2>&1
  diff $BENCH/sw130/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
fi
S245="--ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin --teacher --gen 256 --windows 1"
wait_vram; $B/fr_bench $M $S245 > $O/state-plain.txt 2>&1
echo "state plain: $(grep -hoE 'swaps [0-9]+' $O/state-plain.txt | tail -1), $(grep -h 'hit rate' $O/state-plain.txt | cut -c1-60) (sw126: swaps 5864, 111665 hits)"
wait_vram; $B/fr_bench $M $S245 --mtp $D --spec 2 --draft-vocab $V > $O/state-spec2.txt 2>&1
echo "state spec2: $(grep -hoE 'swaps [0-9]+' $O/state-spec2.txt | tail -1), $(grep -h 'hit rate' $O/state-spec2.txt | cut -c1-60) (sw126: swaps 5071, 182814 hits)"
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 bench/engine_smoke.py $B/flashrt-engine $M --ids $I $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$? $(grep -cE '^PASS' $O/$label.txt) pass, $(grep -cE '^FAIL' $O/$label.txt) fail"
}
smoke default "--n 4096 --gen 32" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke default-mtp "--n 4096 --gen 32" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
smoke reuse-mtp "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
smoke faults-plain "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512
smoke faults-mtp "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
wait_vram
KEY=sw131-$(date +%s)
systemd-run --user --unit=fr-server-sw131 --collect -p MemoryMax=56G bash -c "$FLASHRT/server/target/release/flashrt-server --model $M --port 8090 \
  --api-key $KEY --engine $B/flashrt-engine --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 \
  --engine-arg --draft-vocab --engine-arg $V --engine-arg --ctx --engine-arg 65536 > $O/server.log 2>&1"
ok=0
for i in $(seq 450); do curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:8090/v1/models > /dev/null && { ok=1; break; }; sleep 2; done
echo "server ready=$ok"
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/v1/models); echo "without the key: HTTP $code (want 401)"
[ $ok = 1 ] && python3 bench/server_smoke.py --url http://127.0.0.1:8090 --key $KEY > $O/smoke.txt 2>&1; echo "smoke rc=$?"
systemctl --user stop fr-server-sw131
echo ALLDONE
