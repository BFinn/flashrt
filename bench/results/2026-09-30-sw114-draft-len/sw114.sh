#!/bin/bash
# sw114: data for a per-round draft length (P-3). Every speculative round logged (fr_bench
# --round-log: drafts verified and kept, draft and verify ms, each draft's probability under the
# head), for --spec 1, 2, 3, greedy and temperature 1.0 (sampled drafts), wikitext from the saved
# 32K and 245K states and window 9's 32K prompt; plain runs for the no-draft rate. 6 windows of
# 128 tokens (sampled text: every arm generates its own).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
T1="--temp 1.0 --top-k 20 --top-p 0.95 --seed 1"
S32="--ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin"
S245="--ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin"
W9="--ids $BENCH/sw100/w9_teacher.ids --n-prompt 32793 --prefill-chunk auto --kv q8 --kv-hot 4096"
B=$FLASHRT/build; O=$BENCH/sw114; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" > $O/$name.txt 2>&1
        echo "$name rc=$? $(grep -h '^decode:' $O/$name.txt | tail -1 | cut -c1-90) | $(grep -h 'tokens per round' $O/$name.txt | cut -c1-70)"; }
G="--gen 128 --windows 6"
for ctx in 32 245; do
  S=S$ctx; S=${!S}
  run plain_${ctx}k_greedy $S $G
  run plain_${ctx}k_t1 $S $G $T1
  for k in 1 2 3; do
    run spec${k}_${ctx}k_greedy $S $G --mtp $D --spec $k --draft-vocab $V --round-log $O/spec${k}_${ctx}k_greedy.rounds
    run spec${k}_${ctx}k_t1 $S $G --mtp $D --spec $k --draft-vocab $V $T1 --round-log $O/spec${k}_${ctx}k_t1.rounds
  done
done
run plain_w9_greedy $W9 $G
run plain_w9_t1 $W9 $G $T1
for k in 1 2 3; do
  run spec${k}_w9_greedy $W9 $G --mtp $D --spec $k --draft-vocab $V --round-log $O/spec${k}_w9_greedy.rounds
  run spec${k}_w9_t1 $W9 $G --mtp $D --spec $k --draft-vocab $V $T1 --round-log $O/spec${k}_w9_t1.rounds
done
echo ALLDONE
