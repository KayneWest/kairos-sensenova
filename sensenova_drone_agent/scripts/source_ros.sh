#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SENSENOVA_DRONE_AGENT_PROJECT_ROOT="${PROJECT_ROOT}"
export SENSENOVA_DRONE_AGENT_DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"

echo "Docker-based workflow: ROS is available inside the 'tools' container."
echo "ROS_DISTRO=${ROS_DISTRO}"
echo "Compose file: ${SENSENOVA_DRONE_AGENT_DOCKER_COMPOSE_FILE}"
echo "Example shell:"
echo "docker compose -f ${SENSENOVA_DRONE_AGENT_DOCKER_COMPOSE_FILE} run --rm tools bash"
