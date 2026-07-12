#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${DREAMER_IMAGE:-sensenova_drone_agent-dreamer:local}"
LOGDIR="${1:-sensenova_drone_agent/output/dreamer_world_model_latent_debug}"

cd "$REPO_ROOT"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build \
    -f sensenova_drone_agent/docker/Dockerfile.dreamer \
    -t "$IMAGE" .
fi

docker run --rm \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  "$IMAGE" \
  python -m dreamerv3.main \
    --configs sensenova_drone \
    --logdir "/workspace/$LOGDIR" \
    --run.steps "${DREAMER_STEPS:-20000}" \
    --jax.platform "${DREAMER_JAX_PLATFORM:-cpu}"
