#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

GPU_ARGS=()
if [[ "${SENSENOVA_DOCKER_GPUS:-auto}" != "none" ]]; then
  if [[ "${SENSENOVA_DOCKER_GPUS:-auto}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
      GPU_ARGS=(--gpus all)
    fi
  else
    GPU_ARGS=(--gpus "${SENSENOVA_DOCKER_GPUS}")
  fi
fi

IMAGE="${SENSENOVA_PYBULLET_IMAGE:-sensenova_drone_agent-pybullet-drones:local}"
if [[ -z "${SENSENOVA_PYBULLET_IMAGE:-}" && "${#GPU_ARGS[@]}" -gt 0 ]]; then
  if docker image inspect sensenova_drone_agent-pybullet-drones-gpu:local >/dev/null 2>&1; then
    IMAGE="sensenova_drone_agent-pybullet-drones-gpu:local"
  fi
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e TORCH_HOME=/workspace/.cache/torch \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  python sensenova_drone_agent/scripts/train_pybullet_drones_imagination_policy.py "$@"
