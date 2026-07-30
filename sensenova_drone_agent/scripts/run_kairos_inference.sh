#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_FILE="${1:-sensenova_drone_agent/config/demo_real_i2v_480p.json}"
CONFIG_FILE="${2:-kairos/configs/kairos_4b_config_DMD.py}"
MASTER_PORT="${MASTER_PORT:-29558}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/kairos/third_party:${PYTHONPATH:-}"

torchrun --nnodes=1 --master_port "${MASTER_PORT}" --nproc-per-node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/sensenova_drone_agent/scripts/run_kairos_inference.py" \
    --input_file "${INPUT_FILE}" \
    --config_file "${CONFIG_FILE}"
