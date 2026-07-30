#!/usr/bin/env bash
set -Eeuo pipefail

SEED_VALUE="${1:?usage: $0 <seed> [gpu_selector]}"
GPU_VALUE="${2:-${GPU_SELECTOR:-all}}"

RUN_ID="${RUN_ID:-native_closedloop_zeroaware_postrl_seed_${SEED_VALUE}}"
NAME="${NAME:-sda-soar-native-zeroaware-seed-${SEED_VALUE}}"

RUN_ID="${RUN_ID}" \
NAME="${NAME}" \
GPU_SELECTOR="${GPU_VALUE}" \
SOURCE_RUN="${SOURCE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1}" \
TOKENIZER_CKPT="${TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1/tokenizer_ckpts/latest.pt}" \
DYNAMICS_CKPT="${DYNAMICS_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1/dynamics_ckpts/final_step_0375000.pt}" \
TASKS_JSON="${TASKS_JSON:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1/tasks_all_data.json}" \
RESIDUAL_ADAPTER_CKPT=off \
BC_STEPS="${BC_STEPS:-2400}" \
IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-800}" \
TRAIN_VALUE_DURING_IMAGINATION=1 \
TRAIN_BALANCE_SPEC="${TRAIN_BALANCE_SPEC:-hf_robot_active=0.80,soar_game_positive=0.10,soar_game_active=0.10}" \
BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc_zero_causal_gate}" \
MIN_IMAGINATION_SELECTION_UPDATE="${MIN_IMAGINATION_SELECTION_UPDATE:-100}" \
CAUSAL_POLICY_MIN_MARGIN="${CAUSAL_POLICY_MIN_MARGIN:-0.002}" \
CAUSAL_POLICY_MODE="${CAUSAL_POLICY_MODE:-advantage_gate}" \
CAUSAL_POLICY_NEGATIVE_MODES="${CAUSAL_POLICY_NEGATIVE_MODES:-zero,shuffle}" \
CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.0}" \
CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:--1.0}" \
SOURCE_EVAL_SOURCES="${SOURCE_EVAL_SOURCES:-}" \
SOURCE_EVAL_BATCHES="${SOURCE_EVAL_BATCHES:-0}" \
SOURCE_GATE_HARD_SOURCES="${SOURCE_GATE_HARD_SOURCES:-all,soar}" \
SOURCE_GATE_SOFT_SOURCES="${SOURCE_GATE_SOFT_SOURCES:-droid}" \
SOURCE_GATE_SOFT_MIN_MARGIN="${SOURCE_GATE_SOFT_MIN_MARGIN:--0.005}" \
REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-2.0}" \
REWARD_CONTRAST_MARGIN="${REWARD_CONTRAST_MARGIN:-0.05}" \
REWARD_CONTRAST_HORIZON="${REWARD_CONTRAST_HORIZON:-4}" \
REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,zero,shuffle}" \
EVAL_CAUSAL_DYNAMICS=1 \
EVAL_BATCHES="${EVAL_BATCHES:-96}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-2}" \
ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:--1}" \
ACTION_DIM="${ACTION_DIM:-49}" \
RAW_ACTION_DIM="${RAW_ACTION_DIM:-12}" \
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}" \
ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}" \
IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-2e-5}" \
LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.10}" \
SEED="${SEED_VALUE}" \
TRAIN_BALANCE_SEED="${TRAIN_BALANCE_SEED:-${SEED_VALUE}}" \
EVAL_SEED="${EVAL_SEED:-20260528}" \
SPLIT_SEED="${SPLIT_SEED:-20260527}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
bash sensenova_drone_agent/scripts/experiments/launch_soar_residual_adapter_imagination.sh
