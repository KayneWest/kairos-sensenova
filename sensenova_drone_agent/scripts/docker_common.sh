#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
COMPOSE_CMD=(docker compose -f "${COMPOSE_FILE}")
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DEFAULT_PX4_IMAGE="sensenova_drone_agent-px4-source-sim:local"
DEFAULT_PX4_DEV_IMAGE="${PX4_DEV_IMAGE:-px4io/px4-dev:v1.17.0-rc2}"
DEFAULT_ROS_DISTRO="${ROS_DISTRO:-jazzy}"
DEFAULT_GZ_PARTITION="${GZ_PARTITION:-sensenova_drone_agent}"

compose() {
  env HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" "${COMPOSE_CMD[@]}" "$@"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but was not found on PATH."
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "docker is installed but not usable by the current user."
    exit 1
  fi
}

ensure_px4_checkout() {
  local px4_dir="${PROJECT_ROOT}/third_party/PX4-Autopilot"
  mkdir -p "${PROJECT_ROOT}/third_party"

  if [[ ! -d "${px4_dir}" ]]; then
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive "${px4_dir}"
  else
    git -C "${px4_dir}" submodule update --init --recursive
  fi
}
