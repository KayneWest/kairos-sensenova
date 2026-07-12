#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

docker build \
  -f sensenova_drone_agent/docker/Dockerfile.game_action_sources \
  -t sensenova_drone_agent-game-sources:local \
  .
