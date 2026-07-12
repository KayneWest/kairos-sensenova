#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-blocks_v1}"
OUT="${OUT:-/workspace/sensenova_drone_agent/output/dreamer4_game_actions_native_${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-/workspace/sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
FRAME_DIR="${FRAME_DIR:-${DATA_ROOT}/frames}"
TASKS_JSON="${TASKS_JSON:-${DATA_ROOT}/tasks.json}"

BASE_TOKENIZER_CKPT="${BASE_TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_v1/tokenizer_ckpts/latest.pt}"
SKIP_TOKENIZER="${SKIP_TOKENIZER:-1}"
TOKENIZER_STEPS="${TOKENIZER_STEPS:-5000}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-8}"
TOKENIZER_SEQ_LEN="${TOKENIZER_SEQ_LEN:-8}"
TOKENIZER_GRAD_ACCUM="${TOKENIZER_GRAD_ACCUM:-2}"

DYNAMICS_STEPS="${DYNAMICS_STEPS:-20000}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-8}"
DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-16}"
DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-2}"
DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}"
DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-4}"
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}"
ACTION_DIM="${ACTION_DIM:-61}"
ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-1.0}"
ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.01}"
ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}"
ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-1000}"
SELF_FRACTION="${SELF_FRACTION:-0.25}"
BOOTSTRAP_START="${BOOTSTRAP_START:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/logs/payload.log") 2>&1

echo "[game-actions] started $(date -Is)"
echo "[game-actions] out=${OUT}"
echo "[game-actions] data_root=${DATA_ROOT}"
echo "[game-actions] raw=${RAW_DIR}"
echo "[game-actions] frames=${FRAME_DIR}"
echo "[game-actions] tasks_json=${TASKS_JSON}"
echo "[game-actions] action_dim=${ACTION_DIM} action_features=${ACTION_FEATURES}"
echo "[game-actions] torch/cuda:"
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

for path in "${RAW_DIR}" "${FRAME_DIR}" "${TASKS_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[game-actions] missing required path: ${path}" >&2
    exit 1
  fi
done

TOKENIZER_DIR="${OUT}/tokenizer_ckpts"
TOKENIZER_CKPT="${TOKENIZER_DIR}/latest.pt"
cd /workspace/dreamer4/dreamer4

if [[ "${SKIP_TOKENIZER}" == "1" ]]; then
  if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
    if [[ ! -f "${BASE_TOKENIZER_CKPT}" ]]; then
      echo "[tokenizer] missing BASE_TOKENIZER_CKPT=${BASE_TOKENIZER_CKPT}" >&2
      exit 1
    fi
    mkdir -p "${TOKENIZER_DIR}"
    cp "${BASE_TOKENIZER_CKPT}" "${TOKENIZER_CKPT}"
  fi
  echo "[tokenizer] using ${TOKENIZER_CKPT}"
else
  TOKENIZER_RESUME=()
  if [[ -f "${TOKENIZER_CKPT}" ]]; then
    TOKENIZER_RESUME=(--resume "${TOKENIZER_CKPT}")
  elif [[ -f "${BASE_TOKENIZER_CKPT}" ]]; then
    TOKENIZER_RESUME=(--resume "${BASE_TOKENIZER_CKPT}" --resume_reset_optim)
  fi
  echo "[tokenizer] training/continuing tokenizer for ${TOKENIZER_STEPS} steps"
  torchrun --standalone --nproc_per_node=2 train_tokenizer.py \
    --data_dirs "${FRAME_DIR}" \
    --tasks_from_data \
    --seq_len "${TOKENIZER_SEQ_LEN}" \
    --num_workers 4 \
    --batch_size "${TOKENIZER_BATCH_SIZE}" \
    --H 128 --W 128 --C 3 --patch 8 \
    --d_model 128 \
    --n_heads 4 \
    --depth 4 \
    --n_latents 16 \
    --d_bottleneck 32 \
    --dropout 0.05 \
    --mlp_ratio 4.0 \
    --time_every 1 \
    --mae_p_min 0.0 \
    --mae_p_max 0.9 \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --max_steps "${TOKENIZER_STEPS}" \
    --grad_accum "${TOKENIZER_GRAD_ACCUM}" \
    --lpips_weight 0.0 \
    --log_every 100 \
    --print_every 100 \
    --viz_every 0 \
    --wandb_project dreamer4-tokenizer \
    --wandb_run_name "game-actions-tokenizer-${RUN_ID}" \
    --wandb_mode "${WANDB_MODE}" \
    --ckpt_dir "${TOKENIZER_DIR}" \
    --save_every 2500 \
    "${TOKENIZER_RESUME[@]}"
fi

DYNAMICS_DIR="${OUT}/dynamics_ckpts"
DYNAMICS_CKPT="${DYNAMICS_DIR}/latest.pt"
DYNAMICS_RESUME=()
if [[ -f "${DYNAMICS_CKPT}" ]]; then
  DYNAMICS_RESUME=(--resume "${DYNAMICS_CKPT}")
fi

echo "[dynamics] training game action-conditioned dynamics for ${DYNAMICS_STEPS} steps"
torchrun --standalone --nproc_per_node=2 train_dynamics.py \
  --use_actions \
  --data_dirs "${RAW_DIR}" \
  --frame_dirs "${FRAME_DIR}" \
  --tasks_json "${TASKS_JSON}" \
  --tasks_from_data \
  --seq_len "${DYNAMICS_SEQ_LEN}" \
  --action_dim "${ACTION_DIM}" \
  --action_features "${ACTION_FEATURES}" \
  --num_workers 4 \
  --batch_size "${DYNAMICS_BATCH_SIZE}" \
  --tokenizer_ckpt "${TOKENIZER_CKPT}" \
  --d_model_dyn "${DYNAMICS_D_MODEL}" \
  --dyn_depth "${DYNAMICS_DEPTH}" \
  --n_heads 4 \
  --dropout 0.0 \
  --mlp_ratio 4.0 \
  --time_every 1 \
  --packing_factor 2 \
  --n_register 8 \
  --n_agent 1 \
  --space_mode wm_agent_isolated \
  --k_max 8 \
  --bootstrap_start "${BOOTSTRAP_START}" \
  --self_fraction "${SELF_FRACTION}" \
  --action_frame_offset 0 \
  --action_contrast_weight "${ACTION_CONTRAST_WEIGHT}" \
  --action_contrast_margin "${ACTION_CONTRAST_MARGIN}" \
  --action_contrast_signal "${ACTION_CONTRAST_SIGNAL}" \
  --action_contrast_start "${ACTION_CONTRAST_START}" \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --max_steps "${DYNAMICS_STEPS}" \
  --grad_accum "${DYNAMICS_GRAD_ACCUM}" \
  --grad_clip 1.0 \
  --eval_every 2500 \
  --eval_batch_size 2 \
  --eval_max_items 2 \
  --eval_ctx 8 \
  --eval_horizon 8 \
  --eval_schedule shortcut \
  --eval_d 0.25 \
  --log_every 100 \
  --wandb_project dreamer4-dynamics \
  --wandb_run_name "game-actions-dynamics-${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}" \
  --ckpt_dir "${DYNAMICS_DIR}" \
  --save_every 5000 \
  "${DYNAMICS_RESUME[@]}"

cd /workspace
echo "[eval] evaluating game action grounding"
python /workspace/sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py \
  --data-dir "${RAW_DIR}" \
  --frames-dir "${FRAME_DIR}" \
  --tasks-json "${TASKS_JSON}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --dynamics-ckpt "${DYNAMICS_CKPT}" \
  --out "${OUT}/native_dynamics_eval_h8_game_actions.json" \
  --seq-len 16 \
  --batch-size 4 \
  --max-batches "${EVAL_MAX_BATCHES}" \
  --rollout-horizon 8 \
  --ctx-len 8 \
  --eval-d 0.25 \
  --action-dim "${ACTION_DIM}" \
  --action-features "${ACTION_FEATURES}" \
  --action-frame-offset 0 \
  --device cuda \
  --seed 31

echo "[game-actions] finished $(date -Is)"
