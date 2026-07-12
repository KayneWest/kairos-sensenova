#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e TORCH_HOME=/workspace/.cache/torch \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones:local \
  python sensenova_drone_agent/scripts/eval_pybullet_drones_hover.py "$@"
