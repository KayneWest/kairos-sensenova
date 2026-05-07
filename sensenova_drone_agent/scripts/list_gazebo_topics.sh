#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/prereqs"
DOC_PATH="${PROJECT_ROOT}/docs/CAMERA_TOPIC_DISCOVERY.md"
ALL_TOPICS="${LOG_DIR}/gazebo_topics.txt"
CANDIDATE_TOPICS="${LOG_DIR}/gazebo_camera_topics.txt"
TOPIC_INFO="${LOG_DIR}/gazebo_camera_topic_info.txt"

mkdir -p "${LOG_DIR}"
: > "${ALL_TOPICS}"
: > "${CANDIDATE_TOPICS}"
: > "${TOPIC_INFO}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/docker_common.sh"

require_docker

compose run --rm tools bash -lc '
set -euo pipefail
cd /workspace
gz topic -l | tee logs/prereqs/gazebo_topics.txt
gz topic -l | grep -Ei "camera|image|depth|rgb|point|lidar" | tee logs/prereqs/gazebo_camera_topics.txt || true
if [[ -s logs/prereqs/gazebo_camera_topics.txt ]]; then
  while read -r topic; do
    [[ -z "${topic}" ]] && continue
    {
      echo "==== ${topic} ===="
      gz topic -i -t "${topic}" || true
      echo
    } | tee -a logs/prereqs/gazebo_camera_topic_info.txt
  done < logs/prereqs/gazebo_camera_topics.txt
fi
'

rgb_topic="$(grep -E '/.*/image$' "${CANDIDATE_TOPICS}" 2>/dev/null | grep -Ev '^/depth_camera($|/)' | head -n 1 || true)"
depth_topic="$(grep -Ei '^/depth_camera$|/depth/.*/image$|/depth' "${CANDIDATE_TOPICS}" 2>/dev/null | head -n 1 || true)"
camera_info_topic="$(grep -E '^/world/.*/camera_info$' "${CANDIDATE_TOPICS}" 2>/dev/null | head -n 1 || true)"

if [[ -z "${camera_info_topic}" ]]; then
  camera_info_topic="$(grep -E '/.*/camera_info$|^/camera_info$' "${CANDIDATE_TOPICS}" 2>/dev/null | head -n 1 || true)"
fi

cat > "${DOC_PATH}" <<EOF
# Camera Topic Discovery

## All Gazebo topics

Saved to \`logs/prereqs/gazebo_topics.txt\`

## Candidate image/depth topics

Saved to \`logs/prereqs/gazebo_camera_topics.txt\`

## Chosen RGB topic

${rgb_topic:-Not yet identified}

## Chosen depth topic

${depth_topic:-Not yet identified}

## Chosen camera info topic

${camera_info_topic:-Not yet identified}

## Notes

- Topic metadata saved to \`logs/prereqs/gazebo_camera_topic_info.txt\`
- Topics were discovered dynamically from the running Gazebo instance on this machine.
EOF

echo "Gazebo topic discovery complete."
echo "All topics: ${ALL_TOPICS}"
echo "Camera candidates: ${CANDIDATE_TOPICS}"
echo "Topic info: ${TOPIC_INFO}"
