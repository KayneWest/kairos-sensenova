#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
SEED_A="${SEED_A:-20260536}"
SEED_B="${SEED_B:-20260537}"

launch_soar_only() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  shift 3
  env \
    DATA_SOURCES=soar \
    RUN_ID="native_closedloop_zeroaware_${variant}_seed_${seed}" \
    NAME="sda-native-zeroaware-${variant}-seed-${seed}" \
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
    "$@" \
    bash "${ROOT}/sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_repeat.sh" "${seed}" "device=${gpu}"
}

launch_soar_only "soaronly_strict" "${SEED_A}" 0
launch_soar_only "soaronly_strict_low_lr" "${SEED_B}" 1 \
  IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-1e-5}"
