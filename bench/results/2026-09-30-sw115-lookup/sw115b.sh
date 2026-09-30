#!/bin/bash
# sw115b: teacher-forced texts for the lookup study (sw115's greedy text degenerates: it loops or
# repeats the end-of-turn token once the answer ends). The head drafts 3 tokens at every position
# of a natural continuation: the wikitext file after 32K and 245K, window 9's 384-token reference
# answer, and a code edit (make_code_edit.py: moe_fast.cu back with two identifiers renamed).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; SV=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw115b; mkdir -p $O
source ~/.cargo/env
cargo build --release --manifest-path $FLASHRT/server/Cargo.toml 2>&1 | tail -1
printf '<|im_end|>' > $O/eot.txt; echo "end of turn: $($SV --model $M --tokenize $O/eot.txt)"
NC=$(python3 $FLASHRT/bench/results/2026-09-30-sw115-lookup/make_code_edit.py $SV $M $FLASHRT/arch/qwen4exp/moe_fast.cu $O/code.ids)
echo "code edit: prompt and answer tokens $NC"
NP=${NC%% *}
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
run() { local name=$1 gen=$2 win=$3; shift 3; wait_vram
        timeout 2400 $B/fr_bench $M "$@" --teacher --gen $gen --windows $win --save-tokens $O/$name.gen \
          --mtp $D --draft 3 --draft-vocab $V --round-log $O/$name.drafts > $O/$name.txt 2>&1
        echo "$name rc=$? $(wc -w < $O/$name.gen) tokens"; }
run wiki32k 256 6 --ids $I --n-prompt 32768 --kv-hot 4096 --load-state $BENCH/state-32k-q8-mtp.bin
run wiki245k 256 6 --ids $I --n-prompt 245760 --kv-hot 4096 --load-state $BENCH/state-245k-q8-mtp.bin
run w9 320 1 --ids $BENCH/sw100/w9_teacher.ids --n-prompt 32793 --prefill-chunk auto --kv q8 --kv-hot 4096
run code 256 6 --ids $O/code.ids --n-prompt $NP --prefill-chunk auto --kv q8 --kv-hot 4096
echo ALLDONE
