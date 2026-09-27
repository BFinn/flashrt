# Shared by the measurement-window scripts: take the GPU from the live server and give it
# back on exit. Source it, then call `take_gpu`.
LIVE=flashnext-262k-server
FR=$FLASHRT/build
LLAMA=$LLAMA_CPP/build/bin
MODEL=$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

step() { echo; echo "== $* ($(date +%T))"; }

# A killed loader with ~35 GB of pinned host memory took over 2 minutes to release it
# (window C), so wait up to 10 minutes.
wait_vram() {
    for _ in $(seq 600); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 600 ] && return 0
        sleep 1
    done
    echo "VRAM still in use after 600 s"; exit 1
}

take_gpu() {
    local key
    key=$(cat ~/.config/llama/api-key)
    if curl -s -m 5 -H "Authorization: Bearer $key" http://$TAILNET_HOST:8082/slots | grep -q '"is_processing":true'; then
        echo "live server is processing a request; not starting"
        exit 1
    fi
    trap 'systemctl --user start $LIVE; echo "RESTARTED_LIVE $(date +%T) $(systemctl --user is-active $LIVE)"' EXIT
    systemctl --user stop $LIVE
    wait_vram
    echo "START $(date +%T)"
}
