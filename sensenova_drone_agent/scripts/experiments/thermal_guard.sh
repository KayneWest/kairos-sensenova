#!/usr/bin/env bash
# Thermal/power guard for sda-* docker compute jobs (no root required).
# Polls both GPUs every INTERVAL seconds. If ANY GPU is >= PAUSE_C (or draws
# >= PAUSE_W watts), it `docker pause`s every running sda-* container —
# freezing the cgroup halts kernel submission within milliseconds. Containers
# are unpaused once ALL GPUs are <= RESUME_C and <= RESUME_W.
# Touches ONLY containers whose name starts with sda- ; never other
# processes or another machine's jobs. An exit trap guarantees nothing is
# left frozen. Events log to LOG.
#
# Usage:  setsid nohup bash thermal_guard.sh >/dev/null 2>&1 &
#         (setsid detaches it from the launching session so it survives)
# Stop:   touch /tmp/sda_thermal_guard_stop
set -u
PAUSE_C="${PAUSE_C:-80}"
RESUME_C="${RESUME_C:-68}"
PAUSE_W="${PAUSE_W:-520}"     # ~per-GPU watts; RTX 5090 TGP is ~575W
RESUME_W="${RESUME_W:-420}"
INTERVAL="${INTERVAL:-15}"
LOG="${LOG:-/home/mkrzus/kairos-sensenova/sensenova_drone_agent/output/thermal_guard.log}"
STOPFILE=/tmp/sda_thermal_guard_stop
PIDFILE=/tmp/sda_thermal_guard.pid

echo $$ > "$PIDFILE"
paused=0
log() { echo "$(date -Is) $*" >> "$LOG"; }

unpause_all() {
  local c
  for c in $(docker ps --filter status=paused --format '{{.Names}}' | grep '^sda-' || true); do
    docker unpause "$c" >/dev/null 2>&1 && log "UNPAUSE $c"
  done
  paused=0
}
trap 'log "guard exiting - unpausing everything"; unpause_all; rm -f "$PIDFILE"' EXIT

log "guard started pid=$$ pause>=${PAUSE_C}C/${PAUSE_W}W resume<=${RESUME_C}C/${RESUME_W}W interval=${INTERVAL}s"
while true; do
  [ -f "$STOPFILE" ] && { log "stopfile seen - exiting"; rm -f "$STOPFILE"; exit 0; }
  read -r maxt maxw < <(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' 'BEGIN{t=0;w=0} {if($1>t)t=$1; if($2>w)w=$2} END{print t, w}') || { sleep "$INTERVAL"; continue; }
  maxt=${maxt%%.*}; maxw=${maxw%%.*}
  if [ "$paused" = 0 ] && { [ "${maxt:-0}" -ge "$PAUSE_C" ] || [ "${maxw:-0}" -ge "$PAUSE_W" ]; }; then
    log "TRIP temp=${maxt}C power=${maxw}W - pausing sda-* containers"
    for c in $(docker ps --filter status=running --format '{{.Names}}' | grep '^sda-' || true); do
      docker pause "$c" >/dev/null 2>&1 && log "PAUSE $c"
    done
    paused=1
  elif [ "$paused" = 1 ] && [ "${maxt:-99}" -le "$RESUME_C" ] && [ "${maxw:-999}" -le "$RESUME_W" ]; then
    log "COOL temp=${maxt}C power=${maxw}W - resuming"
    unpause_all
  fi
  sleep "$INTERVAL"
done
