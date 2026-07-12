#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-v1}"
NAME="${NAME:-sda-dreamer4-all-data-balanced-imagination-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_${RUN_ID}}"
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
  -e OUT="/workspace/sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_${RUN_ID}" \
  -e DREAMER_RAW="${DREAMER_RAW:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4}" \
  -e DREAMER_SHARDS="${DREAMER_SHARDS:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full}" \
  -e SOAR_ROOT="${SOAR_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}" \
  -e NATIVE_RUN="${NATIVE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_smoke}" \
  -e IMAGINATION_MODE="${IMAGINATION_MODE:-no_update}" \
  -e BC_STEPS="${BC_STEPS:-1200}" \
  -e IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-400}" \
  -e SELECT_BEST_IMAGINATION="${SELECT_BEST_IMAGINATION:-0}" \
  -e IMAGINATION_EVAL_EVERY="${IMAGINATION_EVAL_EVERY:-0}" \
  -e BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc}" \
  -e DETACH_POLICY_LOG_PROB="${DETACH_POLICY_LOG_PROB:-1}" \
  -e EVAL_BATCHES="${EVAL_BATCHES:-64}" \
  -e BATCH_SIZE="${BATCH_SIZE:-4}" \
  -e NUM_WORKERS="${NUM_WORKERS:-2}" \
  -e TRAIN_SAMPLING_MODE="${TRAIN_SAMPLING_MODE:-dreamer4_reward_mixture}" \
  -e TRAIN_BALANCE_SPEC="${TRAIN_BALANCE_SPEC:-hf_expert_positive=0.25,hf_mixed_positive=0.25,hf_mixed_zero=0.25,soar_game_positive=0.25}" \
  -e TRAIN_BALANCED_SAMPLES="${TRAIN_BALANCED_SAMPLES:-0}" \
  -e TRAIN_BALANCE_RETURN_THRESHOLD="${TRAIN_BALANCE_RETURN_THRESHOLD:-0.0}" \
  -e TRAIN_BALANCE_SEED="${TRAIN_BALANCE_SEED:-0}" \
  -e TRAIN_ACTION_ACTIVE_THRESHOLD="${TRAIN_ACTION_ACTIVE_THRESHOLD:-0.0}" \
  -e TRAIN_MIN_ACTION_ACTIVE_STEPS="${TRAIN_MIN_ACTION_ACTIVE_STEPS:-1}" \
  -e ACTION_DIM="${ACTION_DIM:-49}" \
  -e RAW_ACTION_DIM="${RAW_ACTION_DIM:-12}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}" \
  -e ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}" \
  -e LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
  -e IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-3e-5}" \
  -e TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-per_task}" \
  -e REWARD_CLIP="${REWARD_CLIP:-5.0}" \
  -e VALUE_CLIP="${VALUE_CLIP:-5.0}" \
  -e EVAL_HOLDOUT_FRACTION="${EVAL_HOLDOUT_FRACTION:-0.1}" \
  -e SPLIT_SEED="${SPLIT_SEED:-20260518}" \
  -e EVAL_SEED="${EVAL_SEED:-20260518}" \
  -e SEED="${SEED:-20260518}" \
  -e ADVANTAGE_MODE="${ADVANTAGE_MODE:-centered_sign}" \
  -e ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-bc_return}" \
  -e ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-2.0}" \
  -e IMAGINATION_DYNAMICS_ACTION_MODE="${IMAGINATION_DYNAMICS_ACTION_MODE:-policy}" \
  -e IMAGINATION_AGENT_ACTION_CONTEXT_MODE="${IMAGINATION_AGENT_ACTION_CONTEXT_MODE:-policy}" \
  -e REWARD_VALUE_ACTION_CONTEXT_MODE="${REWARD_VALUE_ACTION_CONTEXT_MODE:-policy}" \
  -e REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-0.0}" \
  -e REWARD_CONTRAST_MARGIN="${REWARD_CONTRAST_MARGIN:-0.05}" \
  -e REWARD_CONTRAST_START="${REWARD_CONTRAST_START:-0}" \
  -e REWARD_CONTRAST_EVERY="${REWARD_CONTRAST_EVERY:-1}" \
  -e REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,shuffle}" \
  -e REWARD_CONTRAST_HORIZON="${REWARD_CONTRAST_HORIZON:-1}" \
  -e REWARD_CONTRAST_POSITIVE_THRESHOLD="${REWARD_CONTRAST_POSITIVE_THRESHOLD:-0.0}" \
  -e REWARD_CONTRAST_MIN_ACTION_NORM="${REWARD_CONTRAST_MIN_ACTION_NORM:-0.0}" \
  -e CAUSAL_POLICY_MODE="${CAUSAL_POLICY_MODE:-off}" \
  -e CAUSAL_POLICY_NEGATIVE_MODES="${CAUSAL_POLICY_NEGATIVE_MODES:-zero,shuffle}" \
  -e CAUSAL_POLICY_MIN_MARGIN="${CAUSAL_POLICY_MIN_MARGIN:-0.0}" \
  -e EVAL_CAUSAL_DYNAMICS="${EVAL_CAUSAL_DYNAMICS:-0}" \
  -e AUX_INVERSE_WEIGHT="${AUX_INVERSE_WEIGHT:-0.0}" \
  -e AUX_EFFECT_WEIGHT="${AUX_EFFECT_WEIGHT:-0.0}" \
  -e AUX_ACTION_EFFECT_MIN_NORM="${AUX_ACTION_EFFECT_MIN_NORM:-0.0}" \
  -e PRIOR_WEIGHT="${PRIOR_WEIGHT:-1.0}" \
  -e PRIOR_HINGE_WEIGHT="${PRIOR_HINGE_WEIGHT:-25.0}" \
  -e PRIOR_HINGE_TARGET="${PRIOR_HINGE_TARGET:-0.008}" \
  -e MEAN_PRIOR_WEIGHT="${MEAN_PRIOR_WEIGHT:-10.0}" \
  -e MEAN_PRIOR_HINGE_WEIGHT="${MEAN_PRIOR_HINGE_WEIGHT:-100.0}" \
  -e MEAN_PRIOR_HINGE_TARGET="${MEAN_PRIOR_HINGE_TARGET:-0.004}" \
  -e VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.10}" \
  -e ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.0005}" \
  -e LOG_STD_INIT="${LOG_STD_INIT:--2.5}" \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/all_data_balanced_imagination_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
