#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-blocks_v1}"
NAME="${NAME:-sda-dreamer4-game-actions-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/dreamer4_game_actions_native_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"

mkdir -p "${OUT}"
printf '%s\n' "${NAME}" > "${OUT}/container_name.txt"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e RUN_ID="${RUN_ID}" \
  -e OUT="/workspace/sensenova_drone_agent/output/dreamer4_game_actions_native_${RUN_ID}" \
  -e DATA_ROOT="${DATA_ROOT:-/workspace/sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1}" \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e DYNAMICS_STEPS="${DYNAMICS_STEPS:-20000}" \
  -e TOKENIZER_STEPS="${TOKENIZER_STEPS:-5000}" \
  -e SKIP_TOKENIZER="${SKIP_TOKENIZER:-1}" \
  -e ACTION_DIM="${ACTION_DIM:-61}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/game_actions_native_dreamer4_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
