#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bridge_gazebo_camera_to_ros.sh --topic /DISCOVERED/GAZEBO/IMAGE/TOPIC
EOF
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/docker_common.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--topic requires a value"
        exit 1
      fi
      TOPIC="$1"
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

if [[ -z "${TOPIC}" ]]; then
  usage
  exit 1
fi

require_docker

echo "Bridging Gazebo topic: ${TOPIC}"
echo "Ensuring the long-lived tools container is running."
compose up -d tools

echo
echo "Now run:"
echo "docker compose -f ${PROJECT_ROOT}/docker-compose.yml exec tools bash -lc 'source /opt/ros/\${ROS_DISTRO}/setup.bash && ros2 topic list'"
echo "docker compose -f ${PROJECT_ROOT}/docker-compose.yml exec tools bash -lc 'source /opt/ros/\${ROS_DISTRO}/setup.bash && ros2 topic hz ${TOPIC}'"
echo "docker compose -f ${PROJECT_ROOT}/docker-compose.yml exec tools bash -lc 'source /opt/ros/\${ROS_DISTRO}/setup.bash && ros2 run rqt_image_view rqt_image_view ${TOPIC}'"
echo "docker compose -f ${PROJECT_ROOT}/docker-compose.yml stop tools"

exec env HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" "${COMPOSE_CMD[@]}" exec tools bash -lc \
  "source /opt/ros/\${ROS_DISTRO}/setup.bash && ros2 run ros_gz_bridge parameter_bridge '${TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image'"
