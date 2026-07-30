#!/usr/bin/env bash

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
cd "${ROOT}" || exit 1

wait_for_names() {
  local pattern="$1"
  while true; do
    names="$(docker ps --format '{{.Names}}')"
    if grep -Eq "${pattern}" <<< "${names}"; then
      sleep 60
    else
      break
    fi
  done
}

echo "watcher-start $(date -Is)"
wait_for_names 'sda-native-zeroaware-soaronly_strict_repl_seed_2026060(1|2)$'

echo "launch-0304 $(date -Is)"
SEEDS_CSV=20260603,20260604 \
GPUS_CSV=0,1 \
WAIT_FOR_BATCH=0 \
bash sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_soar_only_regular_lr_repeats.sh

wait_for_names 'sda-native-zeroaware-soaronly_strict_repl_seed_2026060(3|4)$'

echo "launch-05 $(date -Is)"
SEEDS_CSV=20260605 \
GPUS_CSV=0 \
WAIT_FOR_BATCH=0 \
bash sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_soar_only_regular_lr_repeats.sh

echo "watcher-done $(date -Is)"
