#!/bin/bash
# sw115: text for a prompt-lookup draft study (P-4). Greedy decoding saving every generated token
# (fr_bench --save-tokens), 6 windows of 256, from the saved 32K and 245K wikitext states and window
# 9's 32K prompt, with the MTP head drafting 3 tokens from the true streams at every position
# (--mtp --draft 3 without --spec; --round-log keeps them). lookup.py then asks, at every
# position, what an n-gram match in the earlier sequence would have drafted, what the head did,
# and how much of each the target produced.
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw115; mkdir -p $O
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1; shift; wait_vram; timeout 2400 $B/fr_bench $M "$@" --gen 256 --windows 6 --save-tokens $O/$name.gen \
          --mtp $D --draft 3 --draft-vocab $V --round-log $O/$name.drafts > $O/$name.txt 2>&1
        echo "$name rc=$? $(wc -w < $O/$name.gen) tokens"; }
run wiki32k --ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin
run wiki245k --ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin
run w9 --ids $BENCH/sw100/w9_teacher.ids --n-prompt 32793 --prefill-chunk auto --kv q8 --kv-hot 4096
echo ALLDONE
