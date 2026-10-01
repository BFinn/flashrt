#!/bin/bash
# sw129: phase 6's checks on the development box. bench/run.sh as another machine would run it
# (plain bash; wrapped in a capped unit only because of this box's rules), its KLD step against the
# existing base, then the engine's fault checks (the done event gained fields) and the server's
# unit tests and lints.
set -u
mkdir -p $BENCH/runsh/kld && ln -sf $BENCH/kld/kl8k-f16.bin $BENCH/runsh/kld/kl8k-f16.bin
cd $FLASHRT
MODELS=$MODELS BENCH=$BENCH/runsh CUDACXX=/usr/local/cuda-12.9/bin/nvcc bench/run.sh all > $BENCH/runsh/all.out 2>&1
MODELS=$MODELS BENCH=$BENCH/runsh bench/run.sh kld > $BENCH/runsh/kld.out 2>&1
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf; D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
for a in plain mtp; do
  X=; [ $a = mtp ] && X="--mtp $D --spec 2 --draft-vocab bench/reference/mtp-vocab-ranks.txt"
  python3 bench/engine_smoke.py build/flashrt-engine $M --ids $BENCH/p0c-20260927/wiki.prompt_ids.txt --n 2048 --gen 32 --faults -- \
    --ctx 32768 --prefill-chunk 512 $X > $BENCH/phase6/faults-$a.txt 2> $BENCH/phase6/faults-$a.log
done
(source ~/.cargo/env; cd server && cargo test --release --locked && cargo clippy --release --all-targets --locked -- -D warnings)
