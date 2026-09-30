#!/bin/bash
# sw118: the expert cache's slot leak (sw117) fixed: a key missed by several tokens of a window could
# be admitted twice in one step, and the first slot was orphaned for good. Run from the checkout with
# the fix pulled but not built; the current fr_bench is kept as the old arm.
# (1) GSM8K requests with --cache-check (no orphaned slots expected); (2) KLD gate; (3) teacher-forced
# A/B, old and new interleaved; (4) GSM8K decode speed over 200 requests (against sw117's run);
# (5) engine_smoke and server_smoke.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
T=$BENCH/sw100
O=$BENCH/sw118; mkdir -p $O; cd $FLASHRT
cp build/fr_bench $O/fr_bench.old
cmake --build build 2>&1 | grep -E 'error|warning|FAILED' | head -20
(cd build && FLASHRT_TEST_MODEL=$M ctest --output-on-failure 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
# (1)
N=100 TAG=-check-fixed EXTRA="--engine-arg --cache-check" bash bench/results/2026-09-30-sw117-request-decay/sw117.sh > /dev/null 2>&1
grep "cache check" $BENCH/sw117/flashrt-server-check-fixed.log | awk 'NR<=3 || NR%20==0' | sed 's/.*cache check: /check: /'
# (2)
k() { local name=$1; shift; wait_vram
      (cd $BENCH/kld && timeout 2400 $FLASHRT/build/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 "$@" > $O/kld-$name.log 2>&1)
      echo "kld $name $(grep -h 'KLD mean' $O/kld-$name.log | awk '{print $3}') $(grep -h 'same top' $O/kld-$name.log | awk '{print $4}') $(grep -hoE '[0-9]+ swaps' $O/kld-$name.log)"; }
k fast --fast
k win3 --fast --window 3 --prefill-chunk 2048 --kv-hot 512
# (3)
C="--prefill-chunk auto --kv q8 --kv-hot 4096"
ab() { local arm=$1 name=$2; shift 2
  local B=$FLASHRT/build/fr_bench; [ $arm = old ] && B=$O/fr_bench.old
  wait_vram; $B $M "$@" > $O/$name-$arm-$(date +%s).txt 2>&1
  local f=$(ls -t $O/$name-$arm-*.txt | head -1)
  echo "$arm $name $(grep -h '^decode:' $f | tail -1 | cut -c1-80) | $(grep -h 'expert cache hit rate' $f | cut -c1-30)"; }
for r in 1 2 3; do
  for arm in old new; do
    [ $r = 2 ] && arm=$([ $arm = old ] && echo new || echo old)
    ab $arm w9 --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C
    ab $arm w9mtp --ids $T/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher $C --mtp $D --spec 2 --draft-vocab $V
    ab $arm wiki8k --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n-prompt 8192 --gen 256 --teacher $C
  done
done
# (4)
N=200 TAG=-fixed bash bench/results/2026-09-30-sw117-request-decay/sw117.sh | tail -2
# (5)
mv $BENCH/sw112/smoke $BENCH/sw112/smoke-sw113 2>/dev/null
bash bench/results/2026-09-30-sw112-h3/smoke.sh
mv $BENCH/sw112/smoke $O/smoke
echo ALLDONE
