#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/mkrzus/kairos-sensenova}"
IMAGE="${IMAGE:-sensenova_drone_agent-pybullet-drones-gpu:local}"
GPU_ID="${GPU_ID:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-sda-dreamer4-hf-longrun-$RUN_ID}"
OUT="${OUT:-$REPO/sensenova_drone_agent/output/dreamer4_hf_long_run_v1}"

case "$OUT" in
  "$REPO"/*) OUT_CONTAINER="/workspace${OUT#"$REPO"}" ;;
  *)
    echo "OUT must be inside REPO so it can be mounted into the container: OUT=$OUT REPO=$REPO" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT/logs"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "Container already exists: $NAME"
  exit 1
fi

echo "Launching $NAME on GPU $GPU_ID"
echo "Output: $OUT"

docker run -d \
  --name "$NAME" \
  --gpus "device=$GPU_ID" \
  --user "$(id -u):$(id -g)" \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e OUT="$OUT_CONTAINER" \
  -e TOKENIZER_STEPS="${TOKENIZER_STEPS:-50000}" \
  -e DYNAMICS_STEPS="${DYNAMICS_STEPS:-100000}" \
  -e TOKENIZER_PATCH="${TOKENIZER_PATCH:-8}" \
  -e TOKENIZER_D_MODEL="${TOKENIZER_D_MODEL:-128}" \
  -e TOKENIZER_DEPTH="${TOKENIZER_DEPTH:-4}" \
  -e TOKENIZER_N_LATENTS="${TOKENIZER_N_LATENTS:-16}" \
  -e TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-2}" \
  -e TOKENIZER_SEQ_LEN="${TOKENIZER_SEQ_LEN:-8}" \
  -e TOKENIZER_GRAD_ACCUM="${TOKENIZER_GRAD_ACCUM:-4}" \
  -e SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}" \
  -e DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}" \
  -e DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-4}" \
  -e DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-3}" \
  -e DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-12}" \
  -e DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-8}" \
  -e ACTION_DIM="${ACTION_DIM:-16}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current}" \
  -e ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-0.25}" \
  -e ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.01}" \
  -e ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}" \
  -e ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-0}" \
  -v "$REPO:/workspace" \
  -w /workspace \
  "$IMAGE" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/dreamer4_hf_long_run_payload.sh

echo "$NAME" > "$OUT/container_name.txt"
echo "Logs:"
echo "  docker logs -f $NAME"
echo "  tail -f $OUT/logs/payload.log"
