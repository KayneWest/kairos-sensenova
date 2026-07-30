#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD=""
HEADLESS=""
MODEL_POSE=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/launch_px4_gazebo_x500_depth.sh
  ./scripts/launch_px4_gazebo_x500_depth.sh --world walls
  ./scripts/launch_px4_gazebo_x500_depth.sh --world walls --pose 2,0,1.5,0,0,0
  ./scripts/launch_px4_gazebo_x500_depth.sh --world forest --pose 6,0,1.8,0,0,1.5708
  ./scripts/launch_px4_gazebo_x500_depth.sh --gui
EOF
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/docker_common.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--world requires a value"
        exit 1
      fi
      WORLD="$1"
      ;;
    --gui)
      HEADLESS="0"
      ;;
    --headless)
      HEADLESS="1"
      ;;
    --pose)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--pose requires a value"
        exit 1
      fi
      MODEL_POSE="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

require_docker

if [[ -z "${HEADLESS}" ]]; then
  if [[ -n "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
    HEADLESS="0"
  else
    HEADLESS="1"
  fi
fi

if [[ "${HEADLESS}" == "0" ]]; then
  echo "Launching with Gazebo GUI."
  echo "If X11 access fails, run: xhost +local:docker"
else
  echo "Launching headless."
fi

if [[ -n "${WORLD}" ]]; then
  exec env HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" "${COMPOSE_CMD[@]}" run --rm \
    -e PX4_SIM_MODEL=gz_x500_depth \
    -e PX4_GZ_WORLD="${WORLD}" \
    -e PX4_GZ_MODEL_POSE="${MODEL_POSE}" \
    -e HEADLESS="${HEADLESS}" \
    sim
else
  exec env HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" "${COMPOSE_CMD[@]}" run --rm \
    -e PX4_SIM_MODEL=gz_x500_depth \
    -e PX4_GZ_MODEL_POSE="${MODEL_POSE}" \
    -e HEADLESS="${HEADLESS}" \
    sim
fi
