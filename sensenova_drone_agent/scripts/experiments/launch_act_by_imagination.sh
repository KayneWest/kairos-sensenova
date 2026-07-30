#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-armE_seed2_final}"
NAME="${NAME:-sda-act-by-imagination-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/act_by_imagination_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"
CKPT="${CKPT:-/workspace/sensenova_drone_agent/output/latent_imagination_planner_all_data_v3_rankfix_armE_seed2/planner_ckpts/final.pt}"

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
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash -c "python /workspace/sensenova_drone_agent/scripts/eval_act_by_imagination.py \
    --ckpt '${CKPT}' \
    --out-dir '/workspace/sensenova_drone_agent/output/act_by_imagination_${RUN_ID}' \
    --source-names '${SOURCE_NAMES:-soar_native_v2,dreamer4_hf_expert,hf_robot_bridge_orig_lerobot_dreamer4}' \
    --num-contexts '${NUM_CONTEXTS:-256}' \
    --batch-size '${BATCH_SIZE:-8}' \
    --num-workers '${NUM_WORKERS:-2}' \
    --num-bank '${NUM_BANK:-64}' \
    --k-sweep '${K_SWEEP:-1,4,8,16,32,64}' \
    --eval-chunk '${EVAL_CHUNK:-16}' \
    --seed '${SEED:-20260708}' \
    --inverse-plan-mode '${INVERSE_PLAN_MODE:-candidate}' \
    --score-plan-mode '${SCORE_PLAN_MODE:-plan}' \
    --device '${DEVICE:-cuda}' \
    2>&1 | tee -a '/workspace/sensenova_drone_agent/output/act_by_imagination_${RUN_ID}/logs/payload.log'"

echo "Started ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
