#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-audit_130k}"
NAME="${NAME:-sda-decision-quality-audit-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/decision_quality_audit_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"
CKPT="${CKPT:-/workspace/sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt}"

mkdir -p "${OUT}/logs"
printf '%s\n' "${NAME}" > "${OUT}/container_name.txt"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus "${GPU_SELECTOR:-device=1}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash -c "python /workspace/sensenova_drone_agent/scripts/eval_latent_imagination_decision_quality.py \
    --ckpt '${CKPT}' \
    --out-dir '/workspace/sensenova_drone_agent/output/decision_quality_audit_${RUN_ID}' \
    --source-names '${SOURCE_NAMES:-}' \
    --num-contexts '${NUM_CONTEXTS:-256}' \
    --batch-size '${BATCH_SIZE:-8}' \
    --num-workers '${NUM_WORKERS:-2}' \
    --num-sampled '${NUM_SAMPLED:-64}' \
    --k-sweep '${K_SWEEP:-1,4,8,16,32,64}' \
    --horizons '${HORIZONS:-4,8,16}' \
    --eval-chunk '${EVAL_CHUNK:-16}' \
    --seed '${SEED:-20260706}' \
    --device '${DEVICE:-cuda}' \
    2>&1 | tee -a '/workspace/sensenova_drone_agent/output/decision_quality_audit_${RUN_ID}/logs/payload.log'"

echo "Started ${NAME}"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
