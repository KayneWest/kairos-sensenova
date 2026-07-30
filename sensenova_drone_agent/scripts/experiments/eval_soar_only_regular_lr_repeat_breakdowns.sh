#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"
RUN_GLOB="${RUN_GLOB:-${ROOT}/sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_soaronly_strict*_seed_*}"
SOURCES="${SOURCES:-all,soar}"
HORIZONS="${HORIZONS:-4,8,16}"
EVAL_BATCHES="${EVAL_BATCHES:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_SEED="${EVAL_SEED:-20260528}"
GPU_SELECTOR="${GPU_SELECTOR:-device=0}"
ONLY_GATE_PASS="${ONLY_GATE_PASS:-1}"

mapfile -t RUN_DIRS < <(find "$(dirname "${RUN_GLOB}")" -maxdepth 1 -type d -name "$(basename "${RUN_GLOB}")" | sort)

for run_dir in "${RUN_DIRS[@]}"; do
  summary="${run_dir}/summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "skip unfinished: ${run_dir}"
    continue
  fi
  if [[ "${ONLY_GATE_PASS}" == "1" ]]; then
    if ! python3 - "${summary}" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
best = summary.get("best_imagination_selection") or {}
raise SystemExit(0 if float(best.get("metric_value", -1e9)) > -1e5 else 1)
PY
    then
      echo "skip non-passing: ${run_dir}"
      continue
    fi
  fi
  echo "breakdown: ${run_dir}"
  docker run --rm \
    --gpus "${GPU_SELECTOR}" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/.docker-home \
    -e PYTHONPATH=/workspace/.pydeps:/workspace/dreamer4/dreamer4 \
    -v "${ROOT}:/workspace" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -w /workspace \
    "${IMAGE}" \
    python /workspace/sensenova_drone_agent/scripts/eval_soar_imagination_breakdown.py \
      --run-dir "/workspace/${run_dir#${ROOT}/}" \
      --sources "${SOURCES}" \
      --horizons "${HORIZONS}" \
      --eval-batches "${EVAL_BATCHES}" \
      --batch-size "${BATCH_SIZE}" \
      --eval-seed "${EVAL_SEED}" \
      --device cuda
done
