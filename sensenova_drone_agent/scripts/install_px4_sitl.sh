#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_PATH="${PROJECT_ROOT}/logs/prereqs/px4_install.log"
DOC_PATH="${PROJECT_ROOT}/docs/PX4_GAZEBO_SETUP.md"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/docker_common.sh"

mkdir -p "${PROJECT_ROOT}/third_party" "$(dirname "${LOG_PATH}")"

exec > >(tee "${LOG_PATH}") 2>&1

echo "Starting Docker-based PX4 SITL preparation at $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "PX4 sim image: ${DEFAULT_PX4_IMAGE}"
echo "PX4 dev base image: ${DEFAULT_PX4_DEV_IMAGE}"

require_docker
ensure_px4_checkout

PX4_DIR="${PROJECT_ROOT}/third_party/PX4-Autopilot"
cd "${PX4_DIR}"
px4_commit="$(git rev-parse HEAD)"

docker pull "${DEFAULT_PX4_DEV_IMAGE}"
DOCKER_BUILDKIT=1 compose build sim

base_image_digest="$(docker image inspect "${DEFAULT_PX4_DEV_IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo 'digest unavailable')"
sim_image_id="$(docker image inspect "${DEFAULT_PX4_IMAGE}" --format '{{.Id}}' 2>/dev/null || echo 'image unavailable')"
base_os="$(docker run --rm --entrypoint /bin/bash "${DEFAULT_PX4_IMAGE}" -lc '. /etc/os-release && echo "${PRETTY_NAME}"')"
gazebo_version="$(docker run --rm --entrypoint /bin/bash "${DEFAULT_PX4_IMAGE}" -lc 'gz sim --version | head -n 1')"

cat > "${DOC_PATH}" <<EOF
# PX4/Gazebo Setup

## Install command used

\`docker pull ${DEFAULT_PX4_DEV_IMAGE}\`

\`docker compose -f ${PROJECT_ROOT}/docker-compose.yml build sim\`

## PX4 commit

\`${px4_commit}\`

## Gazebo version

${gazebo_version}

## Notes

- Docker sim image: \`${DEFAULT_PX4_IMAGE}\`
- Docker sim image id: \`${sim_image_id}\`
- Docker dev base image: \`${DEFAULT_PX4_DEV_IMAGE}\`
- Docker dev base digest: \`${base_image_digest}\`
- Container base OS: ${base_os}
- Full log: \`logs/prereqs/px4_install.log\`
- PX4 source is kept in \`third_party/PX4-Autopilot\` and is built inside the Docker sim container to avoid host package installs.

## Reboot required?

No. The Docker-based path does not modify host PX4/Gazebo packages.
EOF

echo
echo "PX4 source-build sim image is ready."
echo "PX4 install log written to ${LOG_PATH}"
echo "Setup summary written to ${DOC_PATH}"
