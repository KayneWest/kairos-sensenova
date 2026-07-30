#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-all_data_v1}"
NAME="${NAME:-sda-latent-imagination-planner-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/latent_imagination_planner_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"

mkdir -p "${OUT}"
printf '%s\n' "${NAME}" > "${OUT}/container_name.txt"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus "${GPU_SELECTOR:-device=0}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e RUN_ID="${RUN_ID}" \
  -e OUT="/workspace/sensenova_drone_agent/output/latent_imagination_planner_${RUN_ID}" \
  -e MANIFEST_JSON="${MANIFEST_JSON:-}" \
  -e TOKENIZER_CKPT="${TOKENIZER_CKPT:-}" \
  -e RESUME_CKPT="${RESUME_CKPT:-}" \
  -e TASKS_JSON="${TASKS_JSON:-}" \
  -e SOURCE_NAMES="${SOURCE_NAMES:-}" \
  -e NO_MANIFEST_WEIGHTS="${NO_MANIFEST_WEIGHTS:-0}" \
  -e SEQ_LEN="${SEQ_LEN:-24}" \
  -e CTX_LEN="${CTX_LEN:-8}" \
  -e HORIZON="${HORIZON:-8}" \
  -e IMG_SIZE="${IMG_SIZE:-128}" \
  -e BATCH_SIZE="${BATCH_SIZE:-8}" \
  -e NUM_WORKERS="${NUM_WORKERS:-2}" \
  -e MAX_STEPS="${MAX_STEPS:-500000}" \
  -e EVAL_EVERY="${EVAL_EVERY:-1000}" \
  -e EVAL_BATCHES="${EVAL_BATCHES:-64}" \
  -e SAVE_EVERY="${SAVE_EVERY:-10000}" \
  -e TRACE_EVERY="${TRACE_EVERY:-5000}" \
  -e ACTION_DIM="${ACTION_DIM:-49}" \
  -e RAW_ACTION_DIM="${RAW_ACTION_DIM:-49}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:--1}" \
  -e HIDDEN_DIM="${HIDDEN_DIM:-1024}" \
  -e PLAN_DIM="${PLAN_DIM:-128}" \
  -e NUM_CANDIDATES="${NUM_CANDIDATES:-16}" \
  -e LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
  -e WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}" \
  -e GRAD_CLIP="${GRAD_CLIP:-1.0}" \
  -e FUTURE_LOSS_WEIGHT="${FUTURE_LOSS_WEIGHT:-1.0}" \
  -e REWARD_LOSS_WEIGHT="${REWARD_LOSS_WEIGHT:-0.25}" \
  -e INVERSE_LOSS_WEIGHT="${INVERSE_LOSS_WEIGHT:-0.25}" \
  -e CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-1.0}" \
  -e CONTRAST_MARGIN="${CONTRAST_MARGIN:-0.02}" \
  -e CONTRAST_RELATIVE_MARGIN="${CONTRAST_RELATIVE_MARGIN:-0.0}" \
  -e PLAN_L2_WEIGHT="${PLAN_L2_WEIGHT:-0.0001}" \
  -e EFFECT_LOSS_WEIGHT="${EFFECT_LOSS_WEIGHT:-0.10}" \
  -e PLAN_UNIT_NORM="${PLAN_UNIT_NORM:-0}" \
  -e PLAN_STEP_CONDITIONING="${PLAN_STEP_CONDITIONING:-0}" \
  -e RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.0}" \
  -e RANK_NUM_BANK="${RANK_NUM_BANK:-4}" \
  -e RANK_NUM_MATCHED="${RANK_NUM_MATCHED:-4}" \
  -e RANK_MARGIN="${RANK_MARGIN:-0.05}" \
  -e RANK_MSE_GAP="${RANK_MSE_GAP:-1.1}" \
  -e INVERSE_PLAN_DROPOUT="${INVERSE_PLAN_DROPOUT:-0.0}" \
  -e INVERSE_IMAGINED_WEIGHT="${INVERSE_IMAGINED_WEIGHT:-0.0}" \
  -e INVERSE_CROSS_WEIGHT="${INVERSE_CROSS_WEIGHT:-0.0}" \
  -e GAMMA="${GAMMA:-0.997}" \
  -e SCORE_PLAN_DROPOUT="${SCORE_PLAN_DROPOUT:-0.0}" \
  -e BC_HEAD_WEIGHT="${BC_HEAD_WEIGHT:-0.0}" \
  -e BC_ENCODER_GRAD="${BC_ENCODER_GRAD:-0}" \
  -e HORIZON_CURRICULUM_MAX="${HORIZON_CURRICULUM_MAX:-0}" \
  -e HORIZON_CURRICULUM_WEIGHT="${HORIZON_CURRICULUM_WEIGHT:-0.5}" \
  -e CONTRAST_MODES="${CONTRAST_MODES:-shuffle,zero,time_shift,time_shift2,time_perm,time_reverse}" \
  -e REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-1}" \
  -e NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.05}" \
  -e MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-4}" \
  -e REWARD_FILTER_MODE="${REWARD_FILTER_MODE:-none}" \
  -e REWARD_SIGNAL_THRESHOLD="${REWARD_SIGNAL_THRESHOLD:-0.0}" \
  -e MIN_REWARD_SIGNAL_STEPS="${MIN_REWARD_SIGNAL_STEPS:-1}" \
  -e REQUIRE_VISUAL_DELTA="${REQUIRE_VISUAL_DELTA:-1}" \
  -e VISUAL_DELTA_THRESHOLD="${VISUAL_DELTA_THRESHOLD:-0.005}" \
  -e MIN_VISUAL_DELTA_STEPS="${MIN_VISUAL_DELTA_STEPS:-4}" \
  -e VISUAL_DELTA_STRIDE="${VISUAL_DELTA_STRIDE:-4}" \
  -e DEVICE="${DEVICE:-cuda}" \
  -e SEED="${SEED:-20260607}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/latent_imagination_planner_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
