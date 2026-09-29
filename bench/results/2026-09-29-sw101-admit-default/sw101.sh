#!/bin/bash
# sw101: admit 1 / margin 1.2 as the expert cache's default (sw100). (1) The KLD gate on the fast
# path, whose value depends on the cache's content (sw94): plain --fast and window 3 / hot set 512.
# (2) The MTP arms, where the hit rate is lowest: fr_bench teacher-forced on window 9's generation
# with --spec 2, old and new admission, 2 runs each. (3) Window 9's protocol, arms G and S, n = 3
# (against sw96's n = 5 with the old admission).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; O=$BENCH/sw101; mkdir -p $O; cd $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast > $O/kld-fast.log 2>&1); echo "kld fast $(grep -h 'KLD mean' $O/kld-fast.log | awk '{print $3}')"
wait_vram; (cd $BENCH/kld && timeout 2400 $B/fr_kld $M kl8k-f16.bin --ctx 8192 --chunks 2 --batch 64 --fast --window 3 --prefill-chunk 2048 --kv-hot 512 > $O/kld-fast-win3-hot512.log 2>&1); echo "kld win3 $(grep -h 'KLD mean' $O/kld-fast-win3-hot512.log | awk '{print $3}')"
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -hoE '[0-9.]+ tok/s' $O/$name.txt | tail -1) $(grep -h 'hit rate' $O/$name.txt | tail -1 | cut -c1-40)"; }
for r in 1 2; do
  for a in "2 1.5" "1 1.2"; do
    set -- $a
    run w9-spec2-a$1-r$r --ids $BENCH/sw100/w9_teacher.ids --n-prompt 32793 --gen 320 --teacher --prefill-chunk auto --kv q8 --kv-hot 4096 \
      --mtp $D --spec 2 --draft-vocab $V --cache-admit $1 --cache-margin $2
  done
done
IDS=$BENCH/strata-ids.json
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $IDS --gen 384"
G(){ $DB --label sw101-G-spec2-greedy-r$1 --log $O/G$1.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw101.out; }
S(){ $DB --label sw101-S-spec2-t1.0-r$1 --log $O/S$1.log --sampling "temperature=1.0 top_p=0.95 top_k=20" -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw101.out; }
G 1; S 1; S 2; G 2; G 3; S 3
echo done > $O/DONE
