#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mkrzus/kairos-sensenova}"
RUN_ID="${RUN_ID:-v1}"
NAME="${NAME:-sda-dreamer4-all-data-${RUN_ID}}"
OUT="${OUT:-${ROOT}/sensenova_drone_agent/output/dreamer4_all_data_native_${RUN_ID}}"
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
  -e RUN_ID="${RUN_ID}" \
  -e OUT="/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_${RUN_ID}" \
  -e NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
  -e TASKS_JSON="${TASKS_JSON:-}" \
  -e MANIFEST_JSON="${MANIFEST_JSON:-}" \
  -e DREAMER_RAW="${DREAMER_RAW:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4}" \
  -e DREAMER_SHARDS="${DREAMER_SHARDS:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full}" \
  -e SOAR_ROOT="${SOAR_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}" \
  -e ROBONET_ROOT="${ROBONET_ROOT:-/workspace/sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64}" \
  -e HF_ACTION_EXPORT_ROOT="${HF_ACTION_EXPORT_ROOT:-/workspace/sensenova_drone_agent/data/robotics/hf_action_exports}" \
  -e HF_ACTION_DATASETS="${HF_ACTION_DATASETS:-droid_lerobot_dreamer4,fractal20220817_data_lerobot_dreamer4,bridge_orig_lerobot_dreamer4}" \
  -e SOURCE_DEFAULT_WEIGHT="${SOURCE_DEFAULT_WEIGHT:-1}" \
  -e SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-}" \
  -e BASE_TOKENIZER_CKPT="${BASE_TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_smoke/tokenizer_ckpts/latest.pt}" \
  -e BASE_DYNAMICS_CKPT="${BASE_DYNAMICS_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_smoke/dynamics_ckpts/latest.pt}" \
  -e SKIP_TOKENIZER="${SKIP_TOKENIZER:-1}" \
  -e TOKENIZER_STEPS="${TOKENIZER_STEPS:-25000}" \
  -e TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-4}" \
  -e TOKENIZER_SEQ_LEN="${TOKENIZER_SEQ_LEN:-8}" \
  -e TOKENIZER_GRAD_ACCUM="${TOKENIZER_GRAD_ACCUM:-4}" \
  -e DYNAMICS_STEPS="${DYNAMICS_STEPS:-150000}" \
  -e DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-4}" \
  -e DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-16}" \
  -e DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-4}" \
  -e DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}" \
  -e DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-4}" \
  -e ACTION_DIM="${ACTION_DIM:-49}" \
  -e ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  -e ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:-0}" \
  -e REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-0}" \
  -e NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.0}" \
  -e MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-1}" \
  -e REWARD_FILTER_MODE="${REWARD_FILTER_MODE:-none}" \
  -e REWARD_SIGNAL_THRESHOLD="${REWARD_SIGNAL_THRESHOLD:-0.0}" \
  -e MIN_REWARD_SIGNAL_STEPS="${MIN_REWARD_SIGNAL_STEPS:-1}" \
  -e REQUIRE_VISUAL_DELTA="${REQUIRE_VISUAL_DELTA:-0}" \
  -e VISUAL_DELTA_THRESHOLD="${VISUAL_DELTA_THRESHOLD:-0.0}" \
  -e MIN_VISUAL_DELTA_STEPS="${MIN_VISUAL_DELTA_STEPS:-1}" \
  -e VISUAL_DELTA_STRIDE="${VISUAL_DELTA_STRIDE:-4}" \
  -e ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-0.5}" \
  -e ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.01}" \
  -e ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}" \
  -e ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-5000}" \
  -e ACTION_CONTRAST_NEGATIVE_MODES="${ACTION_CONTRAST_NEGATIVE_MODES:-shuffle,zero,time_shift}" \
  -e ACTION_CONTRAST_MIN_ACTION_NORM="${ACTION_CONTRAST_MIN_ACTION_NORM:-0.0}" \
  -e ACTION_CONTRAST_TEMPORAL_START="${ACTION_CONTRAST_TEMPORAL_START:-1}" \
  -e ACTION_CONTRAST_ZERO_MASK_MODE="${ACTION_CONTRAST_ZERO_MASK_MODE:-original}" \
  -e ACTION_CONTRAST_ACTION_NORM_WEIGHT="${ACTION_CONTRAST_ACTION_NORM_WEIGHT:-0.0}" \
  -e ACTION_CONTRAST_LATENT_DELTA_WEIGHT="${ACTION_CONTRAST_LATENT_DELTA_WEIGHT:-0.0}" \
  -e ACTION_CONTRAST_WEIGHT_CLIP="${ACTION_CONTRAST_WEIGHT_CLIP:-10.0}" \
  -e CLOSED_LOOP_WEIGHT="${CLOSED_LOOP_WEIGHT:-0.0}" \
  -e CLOSED_LOOP_START="${CLOSED_LOOP_START:-0}" \
  -e CLOSED_LOOP_CTX="${CLOSED_LOOP_CTX:-8}" \
  -e CLOSED_LOOP_HORIZON="${CLOSED_LOOP_HORIZON:-4}" \
  -e CLOSED_LOOP_SIGNAL="${CLOSED_LOOP_SIGNAL:-0.1}" \
  -e CLOSED_LOOP_BACKPROP_HISTORY="${CLOSED_LOOP_BACKPROP_HISTORY:-0}" \
  -e CLOSED_LOOP_CONTRAST_WEIGHT="${CLOSED_LOOP_CONTRAST_WEIGHT:-0.0}" \
  -e CLOSED_LOOP_CONTRAST_MARGIN="${CLOSED_LOOP_CONTRAST_MARGIN:-0.01}" \
  -e CLOSED_LOOP_NEGATIVE_MODES="${CLOSED_LOOP_NEGATIVE_MODES:-shuffle,zero,time_shift}" \
  -e CLOSED_LOOP_MIN_ACTION_NORM="${CLOSED_LOOP_MIN_ACTION_NORM:-0.0}" \
  -e CLOSED_LOOP_ZERO_MASK_MODE="${CLOSED_LOOP_ZERO_MASK_MODE:-original}" \
  -e CLOSED_LOOP_ACTION_NORM_WEIGHT="${CLOSED_LOOP_ACTION_NORM_WEIGHT:-0.0}" \
  -e CLOSED_LOOP_LATENT_DELTA_WEIGHT="${CLOSED_LOOP_LATENT_DELTA_WEIGHT:-0.0}" \
  -e CLOSED_LOOP_WEIGHT_CLIP="${CLOSED_LOOP_WEIGHT_CLIP:-10.0}" \
  -e SELF_FRACTION="${SELF_FRACTION:-0.25}" \
  -e BOOTSTRAP_START="${BOOTSTRAP_START:-5000}" \
  -e DYNAMICS_LR="${DYNAMICS_LR:-5e-5}" \
  -e EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-256}" \
  -e EVAL_CAUSAL_MIN_RATIO="${EVAL_CAUSAL_MIN_RATIO:-1.02}" \
  -e EVAL_ACTION_FRAME_OFFSET="${EVAL_ACTION_FRAME_OFFSET:-${ACTION_FRAME_OFFSET:-0}}" \
  -e EVAL_NEGATIVE_MODES="${EVAL_NEGATIVE_MODES:-shuffle,zero,time_shift}" \
  -e WANDB_MODE="${WANDB_MODE:-offline}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v "${ROOT}:/workspace" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -w /workspace \
  "${IMAGE}" \
  bash /workspace/sensenova_drone_agent/scripts/experiments/all_data_native_dreamer4_payload.sh

echo "Started ${NAME}"
echo "Logs:"
echo "  docker logs -f ${NAME}"
echo "  tail -f ${OUT}/logs/payload.log"
