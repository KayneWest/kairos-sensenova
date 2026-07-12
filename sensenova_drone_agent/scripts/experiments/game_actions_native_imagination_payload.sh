#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-blocks_v1}"
OUT="${OUT:-/workspace/sensenova_drone_agent/output/dreamer4_game_actions_imagination_${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-/workspace/sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
FRAME_DIR="${FRAME_DIR:-${DATA_ROOT}/frames}"
TASKS_JSON="${TASKS_JSON:-${DATA_ROOT}/tasks.json}"

NATIVE_RUN="${NATIVE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_game_actions_native_blocks_v1}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${NATIVE_RUN}/tokenizer_ckpts/latest.pt}"
DYNAMICS_CKPT="${DYNAMICS_CKPT:-${NATIVE_RUN}/dynamics_ckpts/latest.pt}"

SEQ_LEN="${SEQ_LEN:-16}"
CTX_LEN="${CTX_LEN:-8}"
IMAGINATION_HORIZON="${IMAGINATION_HORIZON:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BC_STEPS="${BC_STEPS:-1200}"
IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-400}"
IMAGINATION_MODE="${IMAGINATION_MODE:-train}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
ACTION_DIM="${ACTION_DIM:-61}"
RAW_ACTION_DIM="${RAW_ACTION_DIM:-15}"
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}"
POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}"
ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}"
REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-0}"
NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.0}"
MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-1}"
REWARD_FILTER_MODE="${REWARD_FILTER_MODE:-none}"
REWARD_SIGNAL_THRESHOLD="${REWARD_SIGNAL_THRESHOLD:-0.0}"
MIN_REWARD_SIGNAL_STEPS="${MIN_REWARD_SIGNAL_STEPS:-1}"

LEARNING_RATE="${LEARNING_RATE:-3e-4}"
IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-3e-5}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-per_task}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"
VALUE_CLIP="${VALUE_CLIP:-5.0}"
EVAL_HOLDOUT_FRACTION="${EVAL_HOLDOUT_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-20260518}"
EVAL_SEED="${EVAL_SEED:-20260518}"
SEED="${SEED:-20260518}"

ADVANTAGE_MODE="${ADVANTAGE_MODE:-centered_sign}"
ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-bc_return}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-2.0}"
POLICY_LOSS_MIN_ADVANTAGE_ABS="${POLICY_LOSS_MIN_ADVANTAGE_ABS:-0.0}"
POLICY_LOSS_MAX_PRIOR_MSE="${POLICY_LOSS_MAX_PRIOR_MSE:-0.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-1.0}"
PRIOR_HINGE_WEIGHT="${PRIOR_HINGE_WEIGHT:-25.0}"
PRIOR_HINGE_TARGET="${PRIOR_HINGE_TARGET:-0.008}"
MEAN_PRIOR_WEIGHT="${MEAN_PRIOR_WEIGHT:-10.0}"
MEAN_PRIOR_HINGE_WEIGHT="${MEAN_PRIOR_HINGE_WEIGHT:-100.0}"
MEAN_PRIOR_HINGE_TARGET="${MEAN_PRIOR_HINGE_TARGET:-0.004}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.10}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.0005}"
LOG_STD_INIT="${LOG_STD_INIT:--2.5}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/logs/payload.log") 2>&1

echo "[game-actions-imagination] started $(date -Is)"
echo "[game-actions-imagination] out=${OUT}"
echo "[game-actions-imagination] data_root=${DATA_ROOT}"
echo "[game-actions-imagination] raw=${RAW_DIR}"
echo "[game-actions-imagination] frames=${FRAME_DIR}"
echo "[game-actions-imagination] tasks_json=${TASKS_JSON}"
echo "[game-actions-imagination] tokenizer=${TOKENIZER_CKPT}"
echo "[game-actions-imagination] dynamics=${DYNAMICS_CKPT}"
echo "[game-actions-imagination] bc_steps=${BC_STEPS} imagination_updates=${IMAGINATION_UPDATES} imagination_mode=${IMAGINATION_MODE}"
echo "[game-actions-imagination] action_dim=${ACTION_DIM} action_features=${ACTION_FEATURES}"
echo "[game-actions-imagination] policy_action_source=${POLICY_ACTION_SOURCE} raw_action_dim=${RAW_ACTION_DIM} action_chunk_len=${ACTION_CHUNK_LEN}"
echo "[game-actions-imagination] reward_filter_mode=${REWARD_FILTER_MODE} reward_signal_threshold=${REWARD_SIGNAL_THRESHOLD} min_reward_signal_steps=${MIN_REWARD_SIGNAL_STEPS}"
echo "[game-actions-imagination] torch/cuda:"
python - <<'PY'
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}, indent=2))
PY

for path in "${RAW_DIR}" "${FRAME_DIR}" "${TASKS_JSON}" "${TOKENIZER_CKPT}" "${DYNAMICS_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[game-actions-imagination] missing required path: ${path}" >&2
    exit 1
  fi
done

python /workspace/sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py \
  --data-dir "${RAW_DIR}" \
  --frames-dir "${FRAME_DIR}" \
  --tasks-json "${TASKS_JSON}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --dynamics-ckpt "${DYNAMICS_CKPT}" \
  --out-dir "${OUT}" \
  --seq-len "${SEQ_LEN}" \
  --ctx-len "${CTX_LEN}" \
  --imagination-horizon "${IMAGINATION_HORIZON}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --bc-steps "${BC_STEPS}" \
  --imagination-updates "${IMAGINATION_UPDATES}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-seed "${EVAL_SEED}" \
  --action-dim "${ACTION_DIM}" \
  --raw-action-dim "${RAW_ACTION_DIM}" \
  --action-features "${ACTION_FEATURES}" \
  --policy-action-source "${POLICY_ACTION_SOURCE}" \
  --action-chunk-len "${ACTION_CHUNK_LEN}" \
  --learning-rate "${LEARNING_RATE}" \
  --imagination-learning-rate "${IMAGINATION_LEARNING_RATE}" \
  --target-normalization "${TARGET_NORMALIZATION}" \
  --reward-clip "${REWARD_CLIP}" \
  --value-clip "${VALUE_CLIP}" \
  --eval-holdout-fraction "${EVAL_HOLDOUT_FRACTION}" \
  --split-seed "${SPLIT_SEED}" \
  --imagination-mode "${IMAGINATION_MODE}" \
  --advantage-mode "${ADVANTAGE_MODE}" \
  --advantage-baseline "${ADVANTAGE_BASELINE}" \
  --advantage-clip "${ADVANTAGE_CLIP}" \
  --policy-loss-min-advantage-abs "${POLICY_LOSS_MIN_ADVANTAGE_ABS}" \
  --policy-loss-max-prior-mse "${POLICY_LOSS_MAX_PRIOR_MSE}" \
  --prior-weight "${PRIOR_WEIGHT}" \
  --prior-hinge-weight "${PRIOR_HINGE_WEIGHT}" \
  --prior-hinge-target "${PRIOR_HINGE_TARGET}" \
  --mean-prior-weight "${MEAN_PRIOR_WEIGHT}" \
  --mean-prior-hinge-weight "${MEAN_PRIOR_HINGE_WEIGHT}" \
  --mean-prior-hinge-target "${MEAN_PRIOR_HINGE_TARGET}" \
  --value-loss-weight "${VALUE_LOSS_WEIGHT}" \
  --entropy-weight "${ENTROPY_WEIGHT}" \
  --log-std-init "${LOG_STD_INIT}" \
  $(if [[ "${REQUIRE_NON_NOOP}" == "1" || "${REQUIRE_NON_NOOP}" == "true" ]]; then echo "--require-non-noop"; fi) \
  --no-op-threshold "${NO_OP_THRESHOLD}" \
  --min-non-noop-steps "${MIN_NON_NOOP_STEPS}" \
  --reward-filter-mode "${REWARD_FILTER_MODE}" \
  --reward-signal-threshold "${REWARD_SIGNAL_THRESHOLD}" \
  --min-reward-signal-steps "${MIN_REWARD_SIGNAL_STEPS}" \
  --device cuda \
  --seed "${SEED}"

echo "[game-actions-imagination] finished $(date -Is)"
