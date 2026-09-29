#!/bin/bash
# sw95e: (1) engine_smoke --reuse with every restore on the cold runs' chunk grid (N = 8,217: the
# tail checkpoint at 8,192): all three cases must be bit-exact. (2) The segmentation control for
# sw95d's off-grid tail case (a restore at 8,960 differed from the cold run by KL 0.16 at the first
# generated position, 0.008 with FLASHRT_GDN_CHUNK=0): the same cold prompt (sw95d's, 11,231
# tokens) prefilled in chunks of 2,048 and of 1,792. Similar KL means the off-grid difference is
# the chunked prefill's rounding.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
IDS=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw95; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
smoke() {   # label, mode args, engine args...
  local label=$1 mode=$2; shift 2
  wait_vram
  python3 $FLASHRT/bench/engine_smoke.py $B/flashrt-engine $M --ids $IDS $mode -- "$@" > $O/$label.txt 2> $O/$label.log
  echo "$label rc=$?"
}
smoke reuse-plain-e "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048
smoke reuse-mtp-e "--reuse --n 9000 --gen 16" --ctx 32768 --prefill-chunk 2048 --ckpt-interval 2048 --mtp $D --spec 2 --draft-vocab $V
for c in 2048 1792; do   # sw95d's tail prompt (N = 8,985 plus 2,246 inserted) cold, two chunk lengths
  wait_vram
  python3 $FLASHRT/bench/results/2026-09-29-sw95-ckpt/first_top.py $B/flashrt-engine $M --ids $IDS --n 8985 --insert 2246 -- \
    --ctx 32768 --prefill-chunk $c >> $O/segmentation.jsonl 2>> $O/segmentation.log
done
echo done > $O/DONE-e
