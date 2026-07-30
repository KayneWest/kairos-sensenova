#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_PATH="${PROJECT_ROOT}/logs/prereqs/install_ros_gz_bridge.log"

mkdir -p "$(dirname "${LOG_PATH}")"

exec > >(tee "${LOG_PATH}") 2>&1

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/docker_common.sh"

require_docker
echo "Building Docker tools image with ROS_DISTRO=${DEFAULT_ROS_DISTRO}"
DOCKER_BUILDKIT=1 compose build sim
DOCKER_BUILDKIT=1 compose build tools
compose run --rm tools bash -lc 'echo "ROS_DISTRO=${ROS_DISTRO}"; ros2 pkg executables ros_gz_image | head -n 5; python3 --version'
echo "ROS bridge tooling image is ready."
