#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IMAGE="${IMAGE:-sensenova_drone_agent-pybullet-drones-gpu:local}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-dreamer4_soar_action_dynamics_continuation_v1}"
CONTAINER_NAME="${CONTAINER_NAME:-sda-${RUN_NAME}}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists. Remove it or set CONTAINER_NAME=..."
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU}" \
  --user "$(id -u):$(id -g)" \
  --shm-size 16g \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RUN_NAME="${RUN_NAME}" \
  -v "${ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/soar_action_conditioned_dynamics_continuation_payload.sh

echo "Started ${CONTAINER_NAME} on GPU ${GPU}."
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  tail -f ${ROOT}/sensenova_drone_agent/output/${RUN_NAME}/logs/payload.log"
