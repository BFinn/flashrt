#!/bin/bash
# sw103: the server end to end after the head's chunk buffers (sw102) changed the prefill's VRAM
# budget: server_smoke.py against a server with the head, then a long prompt through the server
# (a ~95K-token chat message: the chunked prefill with the head's chunk set, then decode).
set -u
M=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
D=$MODELS/mtp-Flash-Next-Q8_0-noembd.gguf
V=$BENCH/mtp-vocab/ranks.txt
B=$FLASHRT/build; S=$FLASHRT/server/target/release/flashrt-server; O=$BENCH/sw103; mkdir -p $O
URL=http://127.0.0.1:8090
wait_vram() { for i in $(seq 300); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 600 ] && return; sleep 2; done; echo "VRAM busy"; exit 1; }
wait_vram
systemd-run --user --unit=fr-server-sw103 --collect -p MemoryMax=56G -p KillMode=mixed bash -c "exec $S --model $M --port 8090 --engine $B/flashrt-engine \
  --engine-arg $M --engine-arg --mtp --engine-arg $D --engine-arg --spec --engine-arg 2 --engine-arg --draft-vocab --engine-arg $V > $O/server.log 2>&1"
for i in $(seq 300); do curl -sf $URL/v1/models > /dev/null && { echo "ready after $((i * 2)) s"; break; }; sleep 2; done
python3 $FLASHRT/bench/server_smoke.py --url $URL > $O/smoke.txt 2>&1; echo "smoke rc=$? $(tail -1 $O/smoke.txt)"
python3 - > $O/long.txt 2>&1 <<PY
import json, time, urllib.error, urllib.request
text = open("$BENCH/p0b-20260927/wiki.txt").read() * 2   # ~95K tokens of text
body = {"messages": [{"role": "user", "content": text + "\n\nSummarise the text above in three sentences."}], "max_tokens": 256,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("$URL/v1/chat/completions", data=json.dumps(body).encode(),
                                          headers={"Content-Type": "application/json"}), timeout=1800).read())
except urllib.error.HTTPError as e:
    raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:500]}")
print(json.dumps({"wall_s": round(time.time() - t, 1), "usage": r["usage"], "timings": r.get("timings"),
                  "content": r["choices"][0]["message"]["content"][:400]}, indent=1))
PY
echo "long rc=$?"; head -c 1200 $O/long.txt
systemctl --user stop fr-server-sw103
echo done > $O/DONE
