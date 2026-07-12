#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
SEED_A="${SEED_A:-20260534}"
SEED_B="${SEED_B:-20260535}"

launch_variant() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  shift 3
  env \
  RUN_ID="native_closedloop_zeroaware_${variant}_seed_${seed}" \
  NAME="sda-native-zeroaware-${variant}-seed-${seed}" \
  SOURCE_EVAL_SOURCES="${SOURCE_EVAL_SOURCES:-all,soar,droid}" \
  SOURCE_EVAL_BATCHES="${SOURCE_EVAL_BATCHES:-32}" \
  SOURCE_GATE_HARD_SOURCES="${SOURCE_GATE_HARD_SOURCES:-all,soar}" \
  SOURCE_GATE_SOFT_SOURCES="${SOURCE_GATE_SOFT_SOURCES:-droid}" \
  SOURCE_GATE_SOFT_MIN_MARGIN="${SOURCE_GATE_SOFT_MIN_MARGIN:--0.005}" \
  BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc_zero_causal_gate_source_aware}" \
  "$@" \
  bash "${ROOT}/sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_repeat.sh" "${seed}" "device=${gpu}"
}

case "${VARIANT_SET:-causal_shortfall}" in
  causal_shortfall)
    launch_variant "sourceaware_shortfall" "${SEED_A}" 0 \
      CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.5}" \
      CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:-0.002}" \
      REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-2.5}" \
      REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,zero,zero,shuffle}"
    launch_variant "sourceaware_low_lr_shortfall" "${SEED_B}" 1 \
      CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.5}" \
      CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:-0.002}" \
      IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-1e-5}" \
      REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-2.5}" \
      REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,zero,zero,shuffle}"
    ;;
  sourceaware_only)
    launch_variant "sourceaware_only" "${SEED_A}" 0
    launch_variant "sourceaware_only" "${SEED_B}" 1
    ;;
  *)
    echo "Unknown VARIANT_SET=${VARIANT_SET}" >&2
    exit 2
    ;;
esac
