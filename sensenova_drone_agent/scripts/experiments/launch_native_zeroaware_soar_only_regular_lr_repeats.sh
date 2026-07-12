#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
SEEDS_CSV="${SEEDS_CSV:-20260601,20260602,20260603,20260604,20260605}"
GPUS_CSV="${GPUS_CSV:-0,1}"
RUN_PREFIX="${RUN_PREFIX:-soaronly_strict_repl}"
WAIT_FOR_BATCH="${WAIT_FOR_BATCH:-1}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds provided via SEEDS_CSV" >&2
  exit 1
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs provided via GPUS_CSV" >&2
  exit 1
fi

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local variant="${RUN_PREFIX}_seed_${seed}"
  env \
    DATA_SOURCES=soar \
    RUN_ID="native_closedloop_zeroaware_${variant}" \
    NAME="sda-native-zeroaware-${variant}" \
    TRAIN_BALANCE_SPEC="${TRAIN_BALANCE_SPEC:-soar_game_positive=0.50,soar_game_active=0.50}" \
    SOURCE_EVAL_SOURCES="${SOURCE_EVAL_SOURCES:-all,soar}" \
    SOURCE_EVAL_BATCHES="${SOURCE_EVAL_BATCHES:-48}" \
    SOURCE_GATE_HARD_SOURCES="${SOURCE_GATE_HARD_SOURCES:-all,soar}" \
    SOURCE_GATE_SOFT_SOURCES="${SOURCE_GATE_SOFT_SOURCES:-}" \
    BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc_zero_causal_gate_source_aware}" \
    CAUSAL_POLICY_MIN_MARGIN="${CAUSAL_POLICY_MIN_MARGIN:-0.002}" \
    CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.5}" \
    CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:-0.002}" \
    REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-2.5}" \
    REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,zero,zero,shuffle}" \
    EVAL_BATCHES="${EVAL_BATCHES:-128}" \
    IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-2e-5}" \
    bash "${ROOT}/sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_repeat.sh" "${seed}" "device=${gpu}"
  local state
  state="$(docker inspect "${variant/#/sda-native-zeroaware-}" --format '{{.State.Status}}' 2>/dev/null || true)"
  if [[ "${state}" == "created" ]]; then
    docker start "${variant/#/sda-native-zeroaware-}" >/dev/null
  fi
}

batch_names=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]//[[:space:]]/}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]//[[:space:]]/}"
  variant="${RUN_PREFIX}_seed_${seed}"
  name="sda-native-zeroaware-${variant}"
  launch_seed "${seed}" "${gpu}"
  batch_names+=("${name}")

  if [[ "${WAIT_FOR_BATCH}" == "1" && "${#batch_names[@]}" -ge "${#GPUS[@]}" ]]; then
    echo "Waiting for batch: ${batch_names[*]}"
    docker wait "${batch_names[@]}" >/dev/null
    batch_names=()
  fi
done

if [[ "${WAIT_FOR_BATCH}" == "1" && "${#batch_names[@]}" -gt 0 ]]; then
  echo "Waiting for final batch: ${batch_names[*]}"
  docker wait "${batch_names[@]}" >/dev/null
fi

echo "Launched ${#SEEDS[@]} SOAR-only regular-LR repeat runs: ${SEEDS_CSV}"
