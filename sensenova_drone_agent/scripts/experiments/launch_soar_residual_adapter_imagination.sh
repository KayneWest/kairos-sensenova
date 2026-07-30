#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-v1}"
NAME="${NAME:-sda-soar-residual-adapter-imagination-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/soar_residual_adapter_imagination_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"

mkdir -p "${OUT}"
printf '%s\n' "${NAME}" > "${OUT}/container_name.txt"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus "${GPU_SELECTOR:-all}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e RUN_ID="${RUN_ID}" \
  -e OUT="/workspace/sensenova_drone_agent/output/soar_residual_adapter_imagination_${RUN_ID}" \
  -e SOURCE_RUN="${SOURCE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1}" \
  -e TASKS_JSON="${TASKS_JSON:-}" \
  -e TOKENIZER_CKPT="${TOKENIZER_CKPT:-}" \
  -e DYNAMICS_CKPT="${DYNAMICS_CKPT:-}" \
  -e DATA_SOURCES="${DATA_SOURCES:-soar,droid}" \
  -e SOAR_ROOT="${SOAR_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}" \
  -e DROID_ROOT="${DROID_ROOT:-/workspace/sensenova_drone_agent/data/robotics/hf_action_exports/droid_lerobot_dreamer4}" \
  -e RESIDUAL_ADAPTER_CKPT="${RESIDUAL_ADAPTER_CKPT:-/workspace/sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt}" \
  -e IMAGINATION_MODE="${IMAGINATION_MODE:-train}" \
  -e BC_STEPS="${BC_STEPS:-1200}" \
  -e IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-400}" \
  -e SELECT_BEST_IMAGINATION="${SELECT_BEST_IMAGINATION:-1}" \
  -e IMAGINATION_EVAL_EVERY="${IMAGINATION_EVAL_EVERY:-50}" \
  -e MIN_IMAGINATION_SELECTION_UPDATE="${MIN_IMAGINATION_SELECTION_UPDATE:-0}" \
  -e BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc}" \
  -e EVAL_CAUSAL_DYNAMICS="${EVAL_CAUSAL_DYNAMICS:-1}" \
  -e EVAL_BATCHES="${EVAL_BATCHES:-64}" \
  -e BATCH_SIZE="${BATCH_SIZE:-4}" \
  -e NUM_WORKERS="${NUM_WORKERS:-2}" \
  -e SEQ_LEN="${SEQ_LEN:-16}" \
  -e CTX_LEN="${CTX_LEN:-8}" \
  -e IMAGINATION_HORIZON="${IMAGINATION_HORIZON:-8}" \
  -e ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:--1}" \
  -e ACTION_DIM="${ACTION_DIM:-49}" \
  -e RAW_ACTION_DIM="${RAW_ACTION_DIM:-12}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}" \
  -e ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}" \
  -e TRAIN_SAMPLING_MODE="${TRAIN_SAMPLING_MODE:-dreamer4_reward_mixture}" \
  -e TRAIN_BALANCE_SPEC="${TRAIN_BALANCE_SPEC:-soar_game_positive=0.35,soar_game_active=0.25,hf_robot_active=0.40}" \
  -e TRAIN_BALANCED_SAMPLES="${TRAIN_BALANCED_SAMPLES:-0}" \
  -e TRAIN_BALANCE_RETURN_THRESHOLD="${TRAIN_BALANCE_RETURN_THRESHOLD:-0.0}" \
  -e TRAIN_BALANCE_SEED="${TRAIN_BALANCE_SEED:-0}" \
  -e TRAIN_ACTION_ACTIVE_THRESHOLD="${TRAIN_ACTION_ACTIVE_THRESHOLD:-0.0}" \
  -e TRAIN_MIN_ACTION_ACTIVE_STEPS="${TRAIN_MIN_ACTION_ACTIVE_STEPS:-1}" \
  -e LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
  -e IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-2e-5}" \
  -e TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-per_task}" \
  -e REWARD_CLIP="${REWARD_CLIP:-5.0}" \
  -e VALUE_CLIP="${VALUE_CLIP:-5.0}" \
  -e REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-0.5}" \
  -e REWARD_CONTRAST_MARGIN="${REWARD_CONTRAST_MARGIN:-0.05}" \
  -e REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,shuffle}" \
  -e REWARD_CONTRAST_HORIZON="${REWARD_CONTRAST_HORIZON:-1}" \
  -e CAUSAL_POLICY_MODE="${CAUSAL_POLICY_MODE:-advantage_gate}" \
  -e CAUSAL_POLICY_NEGATIVE_MODES="${CAUSAL_POLICY_NEGATIVE_MODES:-zero,shuffle}" \
  -e CAUSAL_POLICY_MIN_MARGIN="${CAUSAL_POLICY_MIN_MARGIN:-0.0}" \
  -e CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.0}" \
  -e CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:--1.0}" \
  -e SOURCE_EVAL_SOURCES="${SOURCE_EVAL_SOURCES:-}" \
  -e SOURCE_EVAL_BATCHES="${SOURCE_EVAL_BATCHES:-0}" \
  -e SOURCE_GATE_HARD_SOURCES="${SOURCE_GATE_HARD_SOURCES:-all,soar}" \
  -e SOURCE_GATE_SOFT_SOURCES="${SOURCE_GATE_SOFT_SOURCES:-droid}" \
  -e SOURCE_GATE_SOFT_MIN_MARGIN="${SOURCE_GATE_SOFT_MIN_MARGIN:--0.005}" \
  -e AUX_INVERSE_WEIGHT="${AUX_INVERSE_WEIGHT:-0.1}" \
  -e AUX_EFFECT_WEIGHT="${AUX_EFFECT_WEIGHT:-0.1}" \
  -e AUX_ACTION_EFFECT_MIN_NORM="${AUX_ACTION_EFFECT_MIN_NORM:-0.0}" \
  -e ADVANTAGE_MODE="${ADVANTAGE_MODE:-centered_sign}" \
  -e ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-bc_return}" \
  -e ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-2.0}" \
  -e PRIOR_WEIGHT="${PRIOR_WEIGHT:-1.0}" \
  -e PRIOR_HINGE_WEIGHT="${PRIOR_HINGE_WEIGHT:-25.0}" \
  -e PRIOR_HINGE_TARGET="${PRIOR_HINGE_TARGET:-0.008}" \
  -e MEAN_PRIOR_WEIGHT="${MEAN_PRIOR_WEIGHT:-10.0}" \
  -e MEAN_PRIOR_HINGE_WEIGHT="${MEAN_PRIOR_HINGE_WEIGHT:-100.0}" \
  -e MEAN_PRIOR_HINGE_TARGET="${MEAN_PRIOR_HINGE_TARGET:-0.004}" \
  -e VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.10}" \
  -e ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.0005}" \
  -e LOG_STD_INIT="${LOG_STD_INIT:--2.5}" \
  -e TRAIN_VALUE_DURING_IMAGINATION="${TRAIN_VALUE_DURING_IMAGINATION:-0}" \
  -e EVAL_HOLDOUT_FRACTION="${EVAL_HOLDOUT_FRACTION:-0.1}" \
  -e SPLIT_SEED="${SPLIT_SEED:-20260527}" \
  -e EVAL_SEED="${EVAL_SEED:-20260527}" \
  -e SEED="${SEED:-20260527}" \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/soar_residual_adapter_imagination_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
