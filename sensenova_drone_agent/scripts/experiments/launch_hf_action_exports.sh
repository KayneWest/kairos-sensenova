#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
NAME="${NAME:-sda-hf-action-export}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"

mkdir -p "${ROOT}/sensenova_drone_agent/logs/data_exports"
printf '%s\n' "${NAME}" > "${ROOT}/sensenova_drone_agent/logs/data_exports/hf_action_export.container_name"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e PYTHONUNBUFFERED=1 \
  -e FRAME_SIZE="${FRAME_SIZE:-128}" \
  -e FRAME_STRIDE="${FRAME_STRIDE:-2}" \
  -e SHARD_SIZE="${SHARD_SIZE:-2048}" \
  -e MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}" \
  -e REWARD_MODE="${REWARD_MODE:-zero}" \
  -e TASK_MODE="${TASK_MODE:-fixed}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/hf_action_export_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
