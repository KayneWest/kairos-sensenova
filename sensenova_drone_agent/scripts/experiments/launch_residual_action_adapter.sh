#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-v1}"
NAME="${NAME:-sda-residual-action-adapter-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/residual_action_adapter_${RUN_ID}}"
IMAGE="${IMAGE:-sensenova_drone_agent-dreamer:local}"

mkdir -p "${OUT}"
printf '%s\n' "${NAME}" > "${OUT}/container_name.txt"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus "${GPU_SELECTOR:-all}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.docker-home \
  -e OUT="/workspace/sensenova_drone_agent/output/residual_action_adapter_${RUN_ID}" \
  -e MANIFEST_JSON="${MANIFEST_JSON:-}" \
  -e TOKENIZER_CKPT="${TOKENIZER_CKPT:-}" \
  -e DYNAMICS_CKPT="${DYNAMICS_CKPT:-}" \
  -e TASKS_JSON="${TASKS_JSON:-}" \
  -e SOURCE_NAMES="${SOURCE_NAMES:-soar_native_v2,hf_robot_droid_lerobot_dreamer4}" \
  -e SEQ_LEN="${SEQ_LEN:-16}" \
  -e BATCH_SIZE="${BATCH_SIZE:-8}" \
  -e TRAIN_STEPS="${TRAIN_STEPS:-12000}" \
  -e EVAL_BATCHES="${EVAL_BATCHES:-256}" \
  -e LR="${LR:-3e-4}" \
  -e HIDDEN="${HIDDEN:-256}" \
  -e RESIDUAL_SCALE="${RESIDUAL_SCALE:-1.0}" \
  -e CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-1.0}" \
  -e CONTRAST_MARGIN="${CONTRAST_MARGIN:-0.02}" \
  -e CONTRAST_MODES="${CONTRAST_MODES:-shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse}" \
  -e CONTRAST_ACTION_NORM_WEIGHT="${CONTRAST_ACTION_NORM_WEIGHT:-0.0}" \
  -e CONTRAST_LATENT_DELTA_WEIGHT="${CONTRAST_LATENT_DELTA_WEIGHT:-0.0}" \
  -e CONTRAST_WEIGHT_CLIP="${CONTRAST_WEIGHT_CLIP:-10.0}" \
  -e SIGNAL_LEVEL="${SIGNAL_LEVEL:-0.1}" \
  -e RANDOM_SIGNAL="${RANDOM_SIGNAL:-0}" \
  -e ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:--1}" \
  -e ACTION_DIM="${ACTION_DIM:-49}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-1}" \
  -e NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.1}" \
  -e MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-12}" \
  -e REQUIRE_VISUAL_DELTA="${REQUIRE_VISUAL_DELTA:-}" \
  -e VISUAL_DELTA_THRESHOLD="${VISUAL_DELTA_THRESHOLD:-0.01}" \
  -e MIN_VISUAL_DELTA_STEPS="${MIN_VISUAL_DELTA_STEPS:-8}" \
  -e VISUAL_DELTA_STRIDE="${VISUAL_DELTA_STRIDE:-4}" \
  -e DEVICE="${DEVICE:-cuda}" \
  -e SEED="${SEED:-53}" \
  -e NUM_WORKERS="${NUM_WORKERS:-2}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/residual_action_adapter_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/train.log"
