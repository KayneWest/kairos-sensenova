#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IMAGE="${IMAGE:-sensenova_drone_agent-pybullet-drones-gpu:local}"
GPU="${GPU:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-soar_dreamer4_native_v2_action_contrast}"

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
  -v "${ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/soar_native_action_contrast_v2_payload.sh

echo "Started ${CONTAINER_NAME} on GPU ${GPU}."
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  tail -f ${ROOT}/sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast/native_run.log"
