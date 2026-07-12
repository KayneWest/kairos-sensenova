#!/usr/bin/env bash
set -euo pipefail

cd /workspace

RUN_NAME="${RUN_NAME:-dreamer4_soar_native_v2_action_contrast}"
DATA_OUT="${DATA_OUT:-/workspace/sensenova_drone_agent/data/robotics/soar/${RUN_NAME}}"
RUN_OUT="${RUN_OUT:-/workspace/sensenova_drone_agent/output/${RUN_NAME}}"
SOAR_ZIP="${SOAR_ZIP:-/workspace/sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip}"
V1_OUT="${V1_OUT:-/workspace/sensenova_drone_agent/output/dreamer4_soar_native_v1}"

MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-1024}"
TARGET_TASK_COUNT="${TARGET_TASK_COUNT:-64}"
MAX_STEPS_PER_TRAJ="${MAX_STEPS_PER_TRAJ:-128}"
FRAME_STRIDE="${FRAME_STRIDE:-2}"

TOKENIZER_MAX_STEPS="${TOKENIZER_MAX_STEPS:-7000}"
DYNAMICS_MAX_STEPS="${DYNAMICS_MAX_STEPS:-18000}"
DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-12}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-2}"
DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-2}"
ACTION_DIM="${ACTION_DIM:-16}"
ACTION_FEATURES="${ACTION_FEATURES:-current}"

mkdir -p "${RUN_OUT}"
exec > >(tee -a "${RUN_OUT}/native_run.log") 2>&1

echo "[${RUN_NAME}] start $(date -Iseconds)"
echo "[${RUN_NAME}] data_out=${DATA_OUT}"
echo "[${RUN_NAME}] run_out=${RUN_OUT}"

echo "[${RUN_NAME}] exporting SOAR Dreamer4 dataset"
python sensenova_drone_agent/scripts/export_soar_dreamer4_dataset.py \
  --zip "${SOAR_ZIP}" \
  --out "${DATA_OUT}" \
  --max-trajectories "${MAX_TRAJECTORIES}" \
  --target-task-count "${TARGET_TASK_COUNT}" \
  --min-trajectories-per-task 4 \
  --max-trajectories-per-task 16 \
  --require-both-outcomes-per-task \
  --max-steps-per-trajectory "${MAX_STEPS_PER_TRAJ}" \
  --frame-stride "${FRAME_STRIDE}" \
  --frame-size 128 \
  --shard-size 2048 \
  --selection-mode task_balanced \
  --action-aggregation sum \
  --reward-mode trajectory_success \
  --task-name-mode language \
  --seed 17

echo "[${RUN_NAME}] continuing tokenizer"
cd /workspace/dreamer4/dreamer4
python train_tokenizer.py \
  --data_dirs "${DATA_OUT}/frames" \
  --tasks_from_data \
  --seq_len 8 \
  --num_workers 2 \
  --batch_size 2 \
  --H 128 \
  --W 128 \
  --C 3 \
  --patch 8 \
  --d_model 128 \
  --depth 3 \
  --n_heads 4 \
  --n_latents 16 \
  --d_bottleneck 32 \
  --dropout 0.0 \
  --mae_p_min 0.0 \
  --mae_p_max 0.9 \
  --lpips_weight 0.0 \
  --lr 7e-5 \
  --weight_decay 1e-2 \
  --max_steps "${TOKENIZER_MAX_STEPS}" \
  --grad_accum 2 \
  --log_every 100 \
  --print_every 100 \
  --viz_every 0 \
  --save_every 500 \
  --wandb_mode disabled \
  --ckpt_dir "${RUN_OUT}/tokenizer_ckpts" \
  --resume "${V1_OUT}/tokenizer_ckpts/latest.pt" \
  --resume_reset_optim \
  --seed 17

echo "[${RUN_NAME}] continuing action-conditioned dynamics with contrast"
python train_dynamics.py \
  --use_actions \
  --data_dirs "${DATA_OUT}/raw" \
  --frame_dirs "${DATA_OUT}/frames" \
  --tasks_json "${DATA_OUT}/tasks.json" \
  --tasks_from_data \
  --seq_len "${DYNAMICS_SEQ_LEN}" \
  --action_dim "${ACTION_DIM}" \
  --action_features "${ACTION_FEATURES}" \
  --num_workers 2 \
  --batch_size "${DYNAMICS_BATCH_SIZE}" \
  --tokenizer_ckpt "${RUN_OUT}/tokenizer_ckpts/latest.pt" \
  --d_model_dyn 128 \
  --dyn_depth 3 \
  --n_heads 4 \
  --dropout 0.0 \
  --packing_factor 2 \
  --n_register 4 \
  --n_agent 1 \
  --k_max 8 \
  --bootstrap_start 8000 \
  --self_fraction 0.0 \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --max_steps "${DYNAMICS_MAX_STEPS}" \
  --grad_accum "${DYNAMICS_GRAD_ACCUM}" \
  --grad_clip 1.0 \
  --eval_every 0 \
  --log_every 100 \
  --save_every 500 \
  --action_frame_offset 0 \
  --action_contrast_weight 0.25 \
  --action_contrast_margin 0.01 \
  --action_contrast_signal 0.1 \
  --action_contrast_start 5000 \
  --wandb_mode disabled \
  --ckpt_dir "${RUN_OUT}/dynamics_ckpts" \
  --resume "${V1_OUT}/dynamics_ckpts/latest.pt" \
  --resume_reset_optim \
  --seed 17

echo "[${RUN_NAME}] evaluating action alignment offsets"
cd /workspace
for offset in -2 -1 0 1 2; do
  python sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py \
    --data-dir "${DATA_OUT}/raw" \
    --frames-dir "${DATA_OUT}/frames" \
    --tasks-json "${DATA_OUT}/tasks.json" \
    --tokenizer-ckpt "${RUN_OUT}/tokenizer_ckpts/latest.pt" \
    --dynamics-ckpt "${RUN_OUT}/dynamics_ckpts/latest.pt" \
    --out "${RUN_OUT}/native_dynamics_eval_h8_offset_${offset}.json" \
    --seq-len 16 \
    --batch-size 2 \
    --max-batches 64 \
    --rollout-horizon 8 \
    --ctx-len 8 \
    --eval-d 0.25 \
    --action-dim "${ACTION_DIM}" \
    --action-features "${ACTION_FEATURES}" \
    --action-frame-offset "${offset}" \
    --device cuda \
    --seed 17
done

echo "[${RUN_NAME}] complete $(date -Iseconds)"
