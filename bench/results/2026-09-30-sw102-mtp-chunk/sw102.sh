#!/bin/bash
# sw102: P-1, step 2. During a chunked prefill the MTP head catches up in calls of 1,024 rows, its
# MoE as grouped expert GEMMs over the call, instead of batches of 64 with 8-row mat-vec slices.
# (1) A/B with FLASHRT_MTP_CHUNK=1/0, alternating, 2 runs each (as sw98): a cold 131,072-token
# wikitext prompt with the head (--spec 2), then 256 tokens at temperature 1.0 for acceptance.
# (2) engine_smoke --reuse and --faults with the head (a failure mid-prefill must free the chunk
# buffers). (3) Window 9's protocol, MTP greedy, n = 3 (against sw101).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw102; mkdir -p $O; cd $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for arm in 1 0 1 0; do
  wait_vram
  FLASHRT_MTP_CHUNK=$arm python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS --n 131072 --gen 256 --temp 1.0 -- \
    --ctx 140000 --mtp $D --spec 2 --draft-vocab $V > $O/chunk$arm-$(date +%s).txt 2>&1
  echo "chunk $arm rc=$? $(grep -h '^r1:' $(ls -t $O/chunk$arm-*.txt | head -1) | cut -c1-160)"
done
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$? $(grep -hE 'check\(s\) failed' $O/$label.txt)"
}
smoke reuse-mtp "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
smoke faults-mtp "--faults --n 2048 --gen 32" --ctx 32768 --prefill-chunk 512 --mtp $D --spec 2 --draft-vocab $V
DB="python3 $FLASHRT/bench/flashrt_depthbench.py --engine $B/flashrt-engine --model $M --ids $BENCH/strata-ids.json --gen 384"
for r in 1 2 3; do $DB --label sw102-G-spec2-greedy-r$r --log $O/G$r.log -- --mtp $D --spec 2 --draft-vocab $V 2>&1 | tee -a $O/sw102.out; done
echo done > $O/DONE
