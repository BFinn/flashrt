# Shared by the measurement-window scripts: take the GPU and RAM, and give the live server
# back on exit if it was running. Source it, then call `take_gpu`.
LIVE_UNITS="strata-server flashnext-262k-server"   # none should run since 2026-09-27
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

# Refuses when free RAM is short: a job that pushes the user slice's memory pressure past
# systemd-oomd's limit (60% for user-1000.slice, 50% for user@1000.service, over 20 s) gets
# the whole user manager killed, as on 2026-09-27 16:10.
need_ram() {
    local want_gb=$1 avail_gb
    avail_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
    if [ "$avail_gb" -lt "$want_gb" ]; then
        echo "need ${want_gb} GB of available RAM, have ${avail_gb} GB; not starting"
        exit 1
    fi
}

# Takes the GPU and RAM for a measurement window. Since 2026-09-27 no live service runs on
# the box. If one of the live units is active anyway, stop it for the window and start that
# same unit again on exit; otherwise start nothing.
take_gpu() {
    local u active=""
    for u in $LIVE_UNITS; do systemctl --user is-active -q $u && active=$u; done
    if [ -z "$active" ] && { pgrep -x llama-server >/dev/null || pgrep -x strata >/dev/null; }; then
        echo "another model loader is running (llama-server or strata); not starting"
        exit 1
    fi
    if [ -n "$active" ]; then
        trap "systemctl --user start $active; echo \"RESTARTED_LIVE $active \$(date +%T)\"" EXIT
        systemctl --user stop "$active"
    fi
    wait_vram
    need_ram "${NEED_RAM_GB:-40}"
    echo "START $(date +%T)"
}
