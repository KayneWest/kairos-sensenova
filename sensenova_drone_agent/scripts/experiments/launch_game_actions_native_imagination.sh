#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-blocks_v1}"
NAME="${NAME:-sda-dreamer4-game-actions-imagination-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/dreamer4_game_actions_imagination_${RUN_ID}}"
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
  -e OUT="/workspace/sensenova_drone_agent/output/dreamer4_game_actions_imagination_${RUN_ID}" \
  -e DATA_ROOT="${DATA_ROOT:-/workspace/sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1}" \
  -e NATIVE_RUN="${NATIVE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_game_actions_native_blocks_v1}" \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e BC_STEPS="${BC_STEPS:-1200}" \
  -e IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-400}" \
  -e IMAGINATION_MODE="${IMAGINATION_MODE:-train}" \
  -e EVAL_BATCHES="${EVAL_BATCHES:-64}" \
  -e BATCH_SIZE="${BATCH_SIZE:-4}" \
  -e NUM_WORKERS="${NUM_WORKERS:-2}" \
  -e ACTION_DIM="${ACTION_DIM:-61}" \
  -e RAW_ACTION_DIM="${RAW_ACTION_DIM:-15}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}" \
  -e ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}" \
  -e REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-0}" \
  -e NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.0}" \
  -e MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-1}" \
  -e REWARD_FILTER_MODE="${REWARD_FILTER_MODE:-none}" \
  -e REWARD_SIGNAL_THRESHOLD="${REWARD_SIGNAL_THRESHOLD:-0.0}" \
  -e MIN_REWARD_SIGNAL_STEPS="${MIN_REWARD_SIGNAL_STEPS:-1}" \
  -e LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
  -e IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-3e-5}" \
  -e ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-bc_return}" \
  -e LOG_STD_INIT="${LOG_STD_INIT:--2.5}" \
  -e SEED="${SEED:-20260518}" \
  -e EVAL_SEED="${EVAL_SEED:-20260518}" \
  -e SPLIT_SEED="${SPLIT_SEED:-20260518}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/game_actions_native_imagination_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
