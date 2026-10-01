#!/bin/bash
# sw127: the QSA selection for prefill sub-batches (1 CTA per token below 98K positions, 4 above;
# FLASHRT_SELECT_CL1=0 keeps sw121's 8-CTA cluster for them too). sw124's window 9 run showed prefill
# 9.5-12% slower than sw119's; sw121 had measured decode only. Same binary, toggle rotated: prefill
# of 32K, 128K and 245K tokens (q8 host KV, hot set 4096, automatic chunks); then the fingerprint
# (the selection's output does not depend on the cluster size; it must equal sw126's new build).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
I=$BENCH/p0c-20260927/wiki.prompt_ids.txt
B=$FLASHRT/build; O=$BENCH/sw127; mkdir -p $O/runs; cd $FLASHRT
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
for order in ${ORDERS:-0,1 1,0}; do
  for m in ${order//,/ }; do
    for n in 32768 131072 245760; do
      wait_vram; f=$O/runs/n$n-cl$m-$(date +%s).txt
      FLASHRT_SELECT_CL1=$m $B/fr_bench $M --ids $I --n-prompt $n --gen 8 --prefill-chunk auto --kv q8 --kv-hot 4096 > $f 2>&1
      echo "cl1=$m $n $(grep -h '^prefill:' $f | cut -c1-90)"
    done
  done
done
[ -n "${SKIP_FP:-}" ] || bash bench/results/2026-09-30-sw112-h3/fingerprint.sh sw127 > $O/fp-new.txt 2>&1
diff $BENCH/sw126/fp-new.txt $O/fp-new.txt && echo FINGERPRINT-IDENTICAL || echo FINGERPRINT-DIFFERS
echo ALLDONE
