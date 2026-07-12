#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-/workspace/sensenova_drone_agent/output/dreamer4_hf_long_run_v1}"
RAW="${RAW:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4}"
SHARDS="${SHARDS:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full}"
TASKS_JSON="${TASKS_JSON:-/workspace/dreamer4/tasks.json}"

TARGET_SIZE="${TARGET_SIZE:-128}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

TOKENIZER_STEPS="${TOKENIZER_STEPS:-50000}"
DYNAMICS_STEPS="${DYNAMICS_STEPS:-100000}"

TOKENIZER_PATCH="${TOKENIZER_PATCH:-8}"
TOKENIZER_D_MODEL="${TOKENIZER_D_MODEL:-128}"
TOKENIZER_DEPTH="${TOKENIZER_DEPTH:-4}"
TOKENIZER_N_LATENTS="${TOKENIZER_N_LATENTS:-16}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-2}"
TOKENIZER_SEQ_LEN="${TOKENIZER_SEQ_LEN:-8}"
TOKENIZER_GRAD_ACCUM="${TOKENIZER_GRAD_ACCUM:-4}"
SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"

DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}"
DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-4}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-3}"
DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-12}"
DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-8}"
ACTION_DIM="${ACTION_DIM:-16}"
ACTION_FEATURES="${ACTION_FEATURES:-current}"
ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-0.25}"
ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.01}"
ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}"
ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-0}"

mkdir -p "$OUT/logs" "$SHARDS"

exec > >(tee -a "$OUT/logs/payload.log") 2>&1

echo "[dreamer4-hf-long-run] started $(date -Is)"
echo "[dreamer4-hf-long-run] raw=$RAW"
echo "[dreamer4-hf-long-run] shards=$SHARDS"
echo "[dreamer4-hf-long-run] out=$OUT"
echo "[dreamer4-hf-long-run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY

for split in expert mixed-small mixed-large; do
  if [[ ! -d "$RAW/$split" ]]; then
    echo "[preprocess] skipping missing split: $split"
    continue
  fi
  echo "[preprocess] split=$split"
  python /workspace/dreamer4/dreamer4/preprocess_dataset.py \
    --filedir "$RAW/$split" \
    --outdir "$SHARDS/$split" \
    --target-size "$TARGET_SIZE" \
    --shard-size "$SHARD_SIZE" \
    --tasks-from-data \
    2>&1 | tee -a "$OUT/logs/preprocess_${split}.log"
done

TOKENIZER_DIR="$OUT/tokenizer_ckpts"
TOKENIZER_CKPT="$TOKENIZER_DIR/latest.pt"
if [[ -f "$TOKENIZER_CKPT" ]]; then
  echo "[tokenizer] found existing checkpoint, resuming: $TOKENIZER_CKPT"
  TOKENIZER_RESUME=(--resume "$TOKENIZER_CKPT")
else
  TOKENIZER_RESUME=()
fi

if [[ "$SKIP_TOKENIZER" == "1" ]]; then
  if [[ ! -f "$TOKENIZER_CKPT" ]]; then
    echo "[tokenizer] SKIP_TOKENIZER=1 but missing checkpoint: $TOKENIZER_CKPT" >&2
    exit 1
  fi
  echo "[tokenizer] SKIP_TOKENIZER=1; reusing checkpoint: $TOKENIZER_CKPT"
else
  echo "[tokenizer] training for max_steps=$TOKENIZER_STEPS"
  python /workspace/dreamer4/dreamer4/train_tokenizer.py \
    --data_dirs "$SHARDS/expert" "$SHARDS/mixed-small" "$SHARDS/mixed-large" \
    --tasks_from_data \
    --seq_len "$TOKENIZER_SEQ_LEN" \
    --num_workers 4 \
    --batch_size "$TOKENIZER_BATCH_SIZE" \
    --H 128 --W 128 --C 3 --patch "$TOKENIZER_PATCH" \
    --d_model "$TOKENIZER_D_MODEL" \
    --n_heads 4 \
    --depth "$TOKENIZER_DEPTH" \
    --n_latents "$TOKENIZER_N_LATENTS" \
    --d_bottleneck 32 \
    --dropout 0.05 \
    --mlp_ratio 4.0 \
    --time_every 1 \
    --mae_p_min 0.0 \
    --mae_p_max 0.9 \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --max_steps "$TOKENIZER_STEPS" \
    --grad_accum "$TOKENIZER_GRAD_ACCUM" \
    --lpips_weight 0.0 \
    --log_every 100 \
    --print_every 100 \
    --viz_every 0 \
    --wandb_project dreamer4-tokenizer \
    --wandb_run_name dreamer4-hf-long-run-tokenizer \
    --wandb_mode "$WANDB_MODE" \
    --ckpt_dir "$TOKENIZER_DIR" \
    --save_every 5000 \
    "${TOKENIZER_RESUME[@]}" \
    2>&1 | tee -a "$OUT/logs/tokenizer_train.log"
fi

DYNAMICS_DIR="$OUT/dynamics_ckpts"
DYNAMICS_CKPT="$DYNAMICS_DIR/latest.pt"
if [[ -f "$DYNAMICS_CKPT" ]]; then
  echo "[dynamics] found existing checkpoint, resuming: $DYNAMICS_CKPT"
  DYNAMICS_RESUME=(--resume "$DYNAMICS_CKPT")
else
  DYNAMICS_RESUME=()
fi

echo "[dynamics] training for max_steps=$DYNAMICS_STEPS"
python /workspace/dreamer4/dreamer4/train_dynamics.py \
  --use_actions \
  --data_dirs "$RAW/expert" "$RAW/mixed-small" "$RAW/mixed-large" \
  --frame_dirs "$SHARDS/expert" "$SHARDS/mixed-small" "$SHARDS/mixed-large" \
  --tasks_json "$TASKS_JSON" \
  --tasks_from_data \
  --seq_len "$DYNAMICS_SEQ_LEN" \
  --action_dim "$ACTION_DIM" \
  --action_features "$ACTION_FEATURES" \
  --num_workers 4 \
  --batch_size "$DYNAMICS_BATCH_SIZE" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" \
  --d_model_dyn "$DYNAMICS_D_MODEL" \
  --dyn_depth "$DYNAMICS_DEPTH" \
  --n_heads 4 \
  --dropout 0.0 \
  --mlp_ratio 4.0 \
  --time_every 1 \
  --packing_factor 2 \
  --n_register 8 \
  --n_agent 1 \
  --space_mode wm_agent_isolated \
  --k_max 8 \
  --bootstrap_start 5000 \
  --self_fraction 0.25 \
  --action_frame_offset 0 \
  --action_contrast_weight "$ACTION_CONTRAST_WEIGHT" \
  --action_contrast_margin "$ACTION_CONTRAST_MARGIN" \
  --action_contrast_signal "$ACTION_CONTRAST_SIGNAL" \
  --action_contrast_start "$ACTION_CONTRAST_START" \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --max_steps "$DYNAMICS_STEPS" \
  --grad_accum "$DYNAMICS_GRAD_ACCUM" \
  --grad_clip 1.0 \
  --eval_every 1000 \
  --eval_batch_size 2 \
  --eval_max_items 2 \
  --eval_ctx 8 \
  --eval_horizon 8 \
  --eval_schedule shortcut \
  --eval_d 0.25 \
  --log_every 100 \
  --wandb_project dreamer4-dynamics \
  --wandb_run_name dreamer4-hf-long-run-dynamics \
  --wandb_mode "$WANDB_MODE" \
  --ckpt_dir "$DYNAMICS_DIR" \
  --save_every 5000 \
  "${DYNAMICS_RESUME[@]}" \
  2>&1 | tee -a "$OUT/logs/dynamics_train.log"

echo "[eval] running action-grounding eval"
python /workspace/sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py \
  --data-dir "$RAW/expert" \
  --data-dir "$RAW/mixed-small" \
  --data-dir "$RAW/mixed-large" \
  --frames-dir "$SHARDS/expert" \
  --frames-dir "$SHARDS/mixed-small" \
  --frames-dir "$SHARDS/mixed-large" \
  --tasks-json "$TASKS_JSON" \
  --tokenizer-ckpt "$TOKENIZER_CKPT" \
  --dynamics-ckpt "$DYNAMICS_CKPT" \
  --out "$OUT/native_dynamics_eval_h8.json" \
  --seq-len 16 \
  --batch-size 3 \
  --max-batches 128 \
  --rollout-horizon 8 \
  --ctx-len 8 \
  --eval-d 0.25 \
  --action-dim "$ACTION_DIM" \
  --action-features "$ACTION_FEATURES" \
  --action-frame-offset 0 \
  2>&1 | tee -a "$OUT/logs/final_eval.log"

echo "[dreamer4-hf-long-run] finished $(date -Is)"
