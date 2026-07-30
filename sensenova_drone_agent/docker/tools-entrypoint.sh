#!/usr/bin/env bash
set -eo pipefail

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
fi

if [[ -d /opt/drone-sim-venv ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/drone-sim-venv/bin/activate
  set -u
fi

export GZ_PARTITION="${GZ_PARTITION:-sensenova_drone_agent}"
export PYTHONUNBUFFERED=1

if [[ -d /workspace ]]; then
  cd /workspace
fi

exec "$@"
