#!/usr/bin/env bash
# Thermal/power guard for sda-* docker compute jobs (no root required).
# Polls both GPUs every INTERVAL seconds. Trips (docker pause every running
# sda-* container — freezing the cgroup halts kernel submission within
# milliseconds) when ANY GPU is >= PAUSE_C, or when TOTAL power across all
# GPUs is >= PAUSE_SUM_W. Total (not per-GPU) power is the trigger because
# the machine shutdowns happened under combined dual-GPU load; one GPU at
# its ~575W TGP alone is fine. Each trip dimension resumes independently
# with its own hysteresis, so a temp trip doesn't wait on power or vice
# versa. Touches ONLY containers whose name starts with sda- ; never other
# processes or another machine's jobs. An exit trap guarantees nothing is
# left frozen. Events log to LOG.
#
# Usage:  setsid nohup bash thermal_guard.sh >/dev/null 2>&1 &
#         (setsid detaches it from the launching session so it survives)
# Stop:   touch /tmp/sda_thermal_guard_stop
set -u
PAUSE_C="${PAUSE_C:-80}"
RESUME_C="${RESUME_C:-68}"
PAUSE_SUM_W="${PAUSE_SUM_W:-950}"   # total across GPUs; 2x 5090 flat-out is ~1150W
RESUME_SUM_W="${RESUME_SUM_W:-780}"
INTERVAL="${INTERVAL:-15}"
LOG="${LOG:-/home/mkrzus/kairos-sensenova/sensenova_drone_agent/output/thermal_guard.log}"
STOPFILE=/tmp/sda_thermal_guard_stop
PIDFILE=/tmp/sda_thermal_guard.pid

echo $$ > "$PIDFILE"
paused=0
trip_temp=0
trip_power=0
log() { echo "$(date -Is) $*" >> "$LOG"; }

unpause_all() {
  local c
  for c in $(docker ps --filter status=paused --format '{{.Names}}' | grep '^sda-' || true); do
    docker unpause "$c" >/dev/null 2>&1 && log "UNPAUSE $c"
  done
  paused=0
  trip_temp=0
  trip_power=0
}
trap 'log "guard exiting - unpausing everything"; unpause_all; rm -f "$PIDFILE"' EXIT

log "guard started pid=$$ pause>=${PAUSE_C}C/sum${PAUSE_SUM_W}W resume<=${RESUME_C}C/sum${RESUME_SUM_W}W interval=${INTERVAL}s"
while true; do
  [ -f "$STOPFILE" ] && { log "stopfile seen - exiting"; rm -f "$STOPFILE"; exit 0; }
  read -r maxt sumw < <(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' 'BEGIN{t=0;s=0} {if($1>t)t=$1; s+=$2} END{print t, s}') || { sleep "$INTERVAL"; continue; }
  maxt=${maxt%%.*}; sumw=${sumw%%.*}
  if [ "$paused" = 0 ] && { [ "${maxt:-0}" -ge "$PAUSE_C" ] || [ "${sumw:-0}" -ge "$PAUSE_SUM_W" ]; }; then
    [ "${maxt:-0}" -ge "$PAUSE_C" ] && trip_temp=1
    [ "${sumw:-0}" -ge "$PAUSE_SUM_W" ] && trip_power=1
    log "TRIP temp=${maxt}C sum_power=${sumw}W (temp_trip=${trip_temp} power_trip=${trip_power}) - pausing sda-* containers"
    for c in $(docker ps --filter status=running --format '{{.Names}}' | grep '^sda-' || true); do
      docker pause "$c" >/dev/null 2>&1 && log "PAUSE $c"
    done
    paused=1
  elif [ "$paused" = 1 ]; then
    # Each dimension resumes with hysteresis only if it tripped; otherwise
    # it just needs to be below its pause line.
    temp_ok=0; power_ok=0
    if [ "$trip_temp" = 1 ]; then [ "${maxt:-99}" -le "$RESUME_C" ] && temp_ok=1
    else [ "${maxt:-99}" -lt "$PAUSE_C" ] && temp_ok=1; fi
    if [ "$trip_power" = 1 ]; then [ "${sumw:-9999}" -le "$RESUME_SUM_W" ] && power_ok=1
    else [ "${sumw:-9999}" -lt "$PAUSE_SUM_W" ] && power_ok=1; fi
    if [ "$temp_ok" = 1 ] && [ "$power_ok" = 1 ]; then
      log "COOL temp=${maxt}C sum_power=${sumw}W - resuming"
      unpause_all
    fi
  fi
  sleep "$INTERVAL"
done
