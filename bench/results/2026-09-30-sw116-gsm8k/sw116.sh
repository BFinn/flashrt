#!/bin/bash
# sw116: GSM8K quality check, flashrt against llama.cpp on the same GGUF (bench/gsm8k_eval.py:
# greedy, thinking off, one request at a time, max 1024 tokens). The first N test items (default
# 500). flashrt-server as deployed (the MTP head, --spec 2: greedy argmax drafts are exact);
# llama-server as the reference (llama.cpp's CPU expert path, -ot exps=CPU, fp16 KV, no expert
# cache), both through the OpenAI chat API with the GGUF's template. Temporary units only.
set -u
N=${N:-500}
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw116; mkdir -p $O $BENCH/gsm8k
DATA=$BENCH/gsm8k/test.jsonl
[ -s $DATA ] || curl -sfL -o $DATA https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
echo "data: $(wc -l < $DATA) items, sha256 $(sha256sum $DATA | cut -c1-16)"
echo "flashrt $(git -C $FLASHRT rev-parse --short HEAD); llama.cpp $(git -C $LLAMA_CPP rev-parse --short HEAD)$(git -C $LLAMA_CPP diff --quiet || echo ' (uncommitted changes)')"
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
ready() { for i in $(seq 600); do curl -sf $1 > /dev/null && return 0; sleep 2; done; echo "$1 not ready"; return 1; }
stop() { systemctl --user stop $1; for i in $(seq 60); do systemctl --user is-active --quiet $1 || return 0; sleep 1; done; }
EV="python3 $FLASHRT/bench/gsm8k_eval.py --data $DATA --n $N"

# flashrt
wait_vram
systemd-run --user --unit=fr-server-sw116 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V \
  --engine-arg --ctx --engine-arg 16384 > $O/flashrt-server.log 2>&1"
if ready http://127.0.0.1:8090/v1/models; then $EV --url http://127.0.0.1:8090 --out $O/flashrt.jsonl --label flashrt; fi
stop fr-server-sw116

# llama.cpp
wait_vram
cat > $O/llama.sh <<EOS
exec $LLAMA_CPP/build/bin/llama-server -m $M --host 127.0.0.1 --port 8299 --no-warmup -ngl 99 -ot 'ffn_.*_exps=CPU' -fa on \
  -c 8192 -t 12 -tb 12 --jinja --parallel 1
EOS
systemd-run --user --unit=llama-sw116 --collect -p MemoryMax=56G -p MemorySwapMax=0 bash -c "bash $O/llama.sh > $O/llama-server.log 2>&1"
if ready http://127.0.0.1:8299/health; then $EV --url http://127.0.0.1:8299 --out $O/llama.jsonl --label llama.cpp; fi
stop llama-sw116

python3 $FLASHRT/bench/gsm8k_eval.py --compare $O/flashrt.jsonl $O/llama.jsonl
echo ALLDONE
