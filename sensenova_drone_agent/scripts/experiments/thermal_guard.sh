#!/usr/bin/env bash
# Thermal/power guard for sda-* docker compute jobs (no root required).
#
# Polls both GPUs every INTERVAL seconds and `docker pause`s every running
# sda-* container (cgroup freeze halts kernel submission within
# milliseconds; CUDA contexts survive) on any of three rules:
#   1. Instant peak temp: ANY GPU >= PAUSE_C (resume when <= RESUME_C).
#   2. Instant total power: sum across GPUs >= PAUSE_SUM_W (resume
#      <= RESUME_SUM_W). Total, not per-GPU: the machine shutdowns happened
#      under combined dual-GPU load; one GPU at its ~575W TGP alone is fine.
#   3. Sustained rules (lessons transplanted from the ~/optimal-z guard on
#      this same box, which held 79C / 588W sustained without ever hitting
#      an instant trip): 10-min rolling avg max-temp >= SUST_C forces a
#      >=180s cooldown (chassis heat-soak); 5-min rolling avg total power
#      >= SUST_SUM_W forces a >=90s cooldown.
# Each dimension resumes independently with its own hysteresis; forced
# cooldowns must also expire before resume. Touches ONLY containers whose
# name starts with sda-. An exit trap guarantees nothing is left frozen.
# Note: a paused container keeps its VRAM — this guard sheds heat/power,
# it never frees the card.
#
# Persistence: run under the user systemd unit sda-thermal-guard.service
# (Restart=on-failure + loginctl linger -> survives reboots, session
# teardowns, and crashes). Manual fallback:
#   setsid nohup bash thermal_guard.sh >/dev/null 2>&1 &
# Stop: systemctl --user stop sda-thermal-guard   (or touch $STOPFILE)
set -u
PAUSE_C="${PAUSE_C:-80}"
RESUME_C="${RESUME_C:-68}"
PAUSE_SUM_W="${PAUSE_SUM_W:-950}"     # instant total across GPUs
RESUME_SUM_W="${RESUME_SUM_W:-780}"
SUST_C="${SUST_C:-72}"                # 10-min avg max-temp trip
SUST_C_COOLDOWN="${SUST_C_COOLDOWN:-180}"
SUST_SUM_W="${SUST_SUM_W:-850}"       # 5-min avg total-power trip
SUST_W_COOLDOWN="${SUST_W_COOLDOWN:-90}"
INTERVAL="${INTERVAL:-15}"
LOG="${LOG:-/home/mkrzus/kairos-sensenova/sensenova_drone_agent/output/thermal_guard.log}"
STOPFILE=/tmp/sda_thermal_guard_stop
PIDFILE=/tmp/sda_thermal_guard.pid

TEMP_WIN=$(( 600 / INTERVAL ))        # samples in 10 min
POW_WIN=$(( 300 / INTERVAL ))         # samples in 5 min

echo $$ > "$PIDFILE"
paused=0
trip_temp=0
trip_power=0
cooldown_until=0
ti=0; pi=0
declare -a tbuf pbuf
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

pause_all() {
  local c
  for c in $(docker ps --filter status=running --format '{{.Names}}' | grep '^sda-' || true); do
    docker pause "$c" >/dev/null 2>&1 && log "PAUSE $c"
  done
  paused=1
}

buf_avg() {  # $1=buffer name  $2=count
  local -n buf=$1
  local n=$2 s=0 i
  [ "$n" -eq 0 ] && { echo 0; return; }
  for ((i = 0; i < n; i++)); do s=$((s + buf[i])); done
  echo $((s / n))
}

log "guard started pid=$$ inst>=${PAUSE_C}C/sum${PAUSE_SUM_W}W sust>=${SUST_C}Cavg10m/${SUST_SUM_W}Wavg5m resume<=${RESUME_C}C/sum${RESUME_SUM_W}W interval=${INTERVAL}s"
while true; do
  [ -f "$STOPFILE" ] && { log "stopfile seen - exiting"; rm -f "$STOPFILE"; exit 0; }
  read -r maxt sumw < <(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' 'BEGIN{t=0;s=0} {if($1>t)t=$1; s+=$2} END{print t, s}') || { sleep "$INTERVAL"; continue; }
  maxt=${maxt%%.*}; sumw=${sumw%%.*}
  maxt=${maxt:-0}; sumw=${sumw:-0}

  tbuf[$((ti % TEMP_WIN))]=$maxt; ti=$((ti + 1))
  pbuf[$((pi % POW_WIN))]=$sumw; pi=$((pi + 1))
  tn=$(( ti < TEMP_WIN ? ti : TEMP_WIN ))
  pn=$(( pi < POW_WIN ? pi : POW_WIN ))
  avg_t=$(buf_avg tbuf "$tn")
  avg_w=$(buf_avg pbuf "$pn")
  now=$(date +%s)

  sust_t=0; sust_w=0
  [ "$tn" -ge "$TEMP_WIN" ] && [ "$avg_t" -ge "$SUST_C" ] && sust_t=1
  [ "$pn" -ge "$POW_WIN" ] && [ "$avg_w" -ge "$SUST_SUM_W" ] && sust_w=1

  if [ "$paused" = 0 ]; then
    if [ "$maxt" -ge "$PAUSE_C" ] || [ "$sumw" -ge "$PAUSE_SUM_W" ] || [ "$sust_t" = 1 ] || [ "$sust_w" = 1 ]; then
      [ "$maxt" -ge "$PAUSE_C" ] || [ "$sust_t" = 1 ] && trip_temp=1
      [ "$sumw" -ge "$PAUSE_SUM_W" ] || [ "$sust_w" = 1 ] && trip_power=1
      [ "$sust_t" = 1 ] && [ $((now + SUST_C_COOLDOWN)) -gt "$cooldown_until" ] && cooldown_until=$((now + SUST_C_COOLDOWN))
      [ "$sust_w" = 1 ] && [ $((now + SUST_W_COOLDOWN)) -gt "$cooldown_until" ] && cooldown_until=$((now + SUST_W_COOLDOWN))
      log "TRIP temp=${maxt}C sum_power=${sumw}W avg10m=${avg_t}C avg5m=${avg_w}W (temp=${trip_temp} power=${trip_power} sust_t=${sust_t} sust_w=${sust_w}) - pausing sda-*"
      pause_all
    fi
  else
    # Per-dimension resume: hysteresis where that dimension tripped, else
    # just below its instant pause line; sustained averages must have
    # decayed below their thresholds; forced cooldowns must have expired.
    temp_ok=0; power_ok=0
    if [ "$trip_temp" = 1 ]; then [ "$maxt" -le "$RESUME_C" ] && [ "$avg_t" -lt "$SUST_C" ] && temp_ok=1
    else [ "$maxt" -lt "$PAUSE_C" ] && temp_ok=1; fi
    if [ "$trip_power" = 1 ]; then [ "$sumw" -le "$RESUME_SUM_W" ] && [ "$avg_w" -lt "$SUST_SUM_W" ] && power_ok=1
    else [ "$sumw" -lt "$PAUSE_SUM_W" ] && power_ok=1; fi
    if [ "$now" -ge "$cooldown_until" ] && [ "$temp_ok" = 1 ] && [ "$power_ok" = 1 ]; then
      log "COOL temp=${maxt}C sum_power=${sumw}W avg10m=${avg_t}C avg5m=${avg_w}W - resuming"
      unpause_all
    fi
  fi
  sleep "$INTERVAL"
done
