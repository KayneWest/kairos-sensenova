#!/usr/bin/env bash
set -euo pipefail

cd /workspace

RUN_NAME="${RUN_NAME:-dreamer4_soar_action_dynamics_continuation_v1}"
DATA_ROOT="${DATA_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}"
SOURCE_RUN="${SOURCE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast}"
RUN_OUT="${RUN_OUT:-/workspace/sensenova_drone_agent/output/${RUN_NAME}}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
FRAMES_DIR="${FRAMES_DIR:-${DATA_ROOT}/frames}"
TASKS_JSON="${TASKS_JSON:-${DATA_ROOT}/tasks.json}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${SOURCE_RUN}/tokenizer_ckpts/latest.pt}"
DYNAMICS_RESUME="${DYNAMICS_RESUME:-${SOURCE_RUN}/dynamics_ckpts/latest.pt}"
DYNAMICS_DIR="${DYNAMICS_DIR:-${RUN_OUT}/dynamics_ckpts}"
SELECTION_DIR="${SELECTION_DIR:-${RUN_OUT}/dynamics_selection}"

DYNAMICS_MAX_STEPS="${DYNAMICS_MAX_STEPS:-26000}"
DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-24}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-1}"
DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-4}"
DYNAMICS_LR="${DYNAMICS_LR:-2e-5}"
DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}"
DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-3}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

ACTION_DIM="${ACTION_DIM:-64}"
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}"
ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:-0}"
RESUME_ALLOW_MISMATCH="${RESUME_ALLOW_MISMATCH:-1}"

REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-1}"
NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.01}"
MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-2}"
REQUIRE_VISUAL_DELTA="${REQUIRE_VISUAL_DELTA:-1}"
VISUAL_DELTA_THRESHOLD="${VISUAL_DELTA_THRESHOLD:-0.002}"
MIN_VISUAL_DELTA_STEPS="${MIN_VISUAL_DELTA_STEPS:-2}"
VISUAL_DELTA_STRIDE="${VISUAL_DELTA_STRIDE:-4}"

ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-0.75}"
ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.02}"
ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}"
ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-0}"
ACTION_CONTRAST_NEGATIVE_MODES="${ACTION_CONTRAST_NEGATIVE_MODES:-shuffle,zero,time_shift,time_shift2,far_shuffle,effect_far_shuffle}"
ACTION_CONTRAST_MIN_ACTION_NORM="${ACTION_CONTRAST_MIN_ACTION_NORM:-0.01}"
ACTION_CONTRAST_TEMPORAL_START="${ACTION_CONTRAST_TEMPORAL_START:-1}"
ACTION_CONTRAST_ZERO_MASK_MODE="${ACTION_CONTRAST_ZERO_MASK_MODE:-original}"
ACTION_CONTRAST_ACTION_NORM_WEIGHT="${ACTION_CONTRAST_ACTION_NORM_WEIGHT:-0.5}"
ACTION_CONTRAST_LATENT_DELTA_WEIGHT="${ACTION_CONTRAST_LATENT_DELTA_WEIGHT:-0.5}"
ACTION_CONTRAST_WEIGHT_CLIP="${ACTION_CONTRAST_WEIGHT_CLIP:-10.0}"

CLOSED_LOOP_WEIGHT="${CLOSED_LOOP_WEIGHT:-0.25}"
CLOSED_LOOP_START="${CLOSED_LOOP_START:-0}"
CLOSED_LOOP_CTX="${CLOSED_LOOP_CTX:-8}"
CLOSED_LOOP_HORIZON="${CLOSED_LOOP_HORIZON:-8}"
CLOSED_LOOP_SIGNAL="${CLOSED_LOOP_SIGNAL:-0.1}"
CLOSED_LOOP_CONTRAST_WEIGHT="${CLOSED_LOOP_CONTRAST_WEIGHT:-1.0}"
CLOSED_LOOP_CONTRAST_MARGIN="${CLOSED_LOOP_CONTRAST_MARGIN:-0.02}"
CLOSED_LOOP_NEGATIVE_MODES="${CLOSED_LOOP_NEGATIVE_MODES:-shuffle,zero,time_shift,time_shift2,far_shuffle,effect_far_shuffle}"
CLOSED_LOOP_MIN_ACTION_NORM="${CLOSED_LOOP_MIN_ACTION_NORM:-0.01}"
CLOSED_LOOP_ZERO_MASK_MODE="${CLOSED_LOOP_ZERO_MASK_MODE:-original}"
CLOSED_LOOP_ACTION_NORM_WEIGHT="${CLOSED_LOOP_ACTION_NORM_WEIGHT:-0.5}"
CLOSED_LOOP_LATENT_DELTA_WEIGHT="${CLOSED_LOOP_LATENT_DELTA_WEIGHT:-0.5}"
CLOSED_LOOP_WEIGHT_CLIP="${CLOSED_LOOP_WEIGHT_CLIP:-10.0}"

SELECT_HORIZONS="${SELECT_HORIZONS:-8,16}"
SELECT_MAX_BATCHES="${SELECT_MAX_BATCHES:-256}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-2}"
SELECT_MAX_CHECKPOINTS="${SELECT_MAX_CHECKPOINTS:-4}"
SELECT_NEGATIVE_MODES="${SELECT_NEGATIVE_MODES:-shuffle,zero,time_shift,far_shuffle}"
SELECT_CAUSAL_MIN_RATIO="${SELECT_CAUSAL_MIN_RATIO:-1.02}"

mkdir -p "${RUN_OUT}/logs" "${DYNAMICS_DIR}" "${SELECTION_DIR}"
exec > >(tee -a "${RUN_OUT}/logs/payload.log") 2>&1

echo "[${RUN_NAME}] start $(date -Iseconds)"
echo "[${RUN_NAME}] data_root=${DATA_ROOT}"
echo "[${RUN_NAME}] source_run=${SOURCE_RUN}"
echo "[${RUN_NAME}] run_out=${RUN_OUT}"
echo "[${RUN_NAME}] tokenizer_ckpt=${TOKENIZER_CKPT}"
echo "[${RUN_NAME}] dynamics_resume=${DYNAMICS_RESUME}"
echo "[${RUN_NAME}] action_dim=${ACTION_DIM} action_features=${ACTION_FEATURES} action_frame_offset=${ACTION_FRAME_OFFSET}"
echo "[${RUN_NAME}] dynamics_max_steps=${DYNAMICS_MAX_STEPS} seq_len=${DYNAMICS_SEQ_LEN} batch=${DYNAMICS_BATCH_SIZE} grad_accum=${DYNAMICS_GRAD_ACCUM} lr=${DYNAMICS_LR}"
echo "[${RUN_NAME}] filters non_noop=${REQUIRE_NON_NOOP} visual_delta=${REQUIRE_VISUAL_DELTA}"
echo "[${RUN_NAME}] action_contrast modes=${ACTION_CONTRAST_NEGATIVE_MODES} weight=${ACTION_CONTRAST_WEIGHT} margin=${ACTION_CONTRAST_MARGIN}"
echo "[${RUN_NAME}] closed_loop modes=${CLOSED_LOOP_NEGATIVE_MODES} weight=${CLOSED_LOOP_WEIGHT} contrast_weight=${CLOSED_LOOP_CONTRAST_WEIGHT}"
echo "[${RUN_NAME}] selector horizons=${SELECT_HORIZONS} max_batches=${SELECT_MAX_BATCHES} modes=${SELECT_NEGATIVE_MODES}"

for required in "${RAW_DIR}" "${FRAMES_DIR}" "${TASKS_JSON}" "${TOKENIZER_CKPT}" "${DYNAMICS_RESUME}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[${RUN_NAME}] missing required path: ${required}" >&2
    exit 1
  fi
done

cd /workspace/dreamer4/dreamer4
RESUME_ARGS=(--resume "${DYNAMICS_RESUME}" --resume_reset_optim)
if [[ "${RESUME_ALLOW_MISMATCH}" == "1" ]]; then
  RESUME_ARGS+=(--resume_allow_mismatch)
fi

python train_dynamics.py \
  --use_actions \
  --data_dirs "${RAW_DIR}" \
  --frame_dirs "${FRAMES_DIR}" \
  --tasks_json "${TASKS_JSON}" \
  --tasks_from_data \
  --seq_len "${DYNAMICS_SEQ_LEN}" \
  --action_dim "${ACTION_DIM}" \
  --action_features "${ACTION_FEATURES}" \
  --num_workers 2 \
  --batch_size "${DYNAMICS_BATCH_SIZE}" \
  --tokenizer_ckpt "${TOKENIZER_CKPT}" \
  --d_model_dyn "${DYNAMICS_D_MODEL}" \
  --dyn_depth "${DYNAMICS_DEPTH}" \
  --n_heads 4 \
  --dropout 0.0 \
  --mlp_ratio 4.0 \
  --time_every 1 \
  --packing_factor 2 \
  --n_register 4 \
  --n_agent 1 \
  --space_mode wm_agent_isolated \
  --k_max 8 \
  --bootstrap_start 5000 \
  --self_fraction 0.0 \
  --action_frame_offset "${ACTION_FRAME_OFFSET}" \
  $([[ "${REQUIRE_NON_NOOP}" == "1" ]] && printf '%s' "--require_non_noop") \
  --no_op_threshold "${NO_OP_THRESHOLD}" \
  --min_non_noop_steps "${MIN_NON_NOOP_STEPS}" \
  $([[ "${REQUIRE_VISUAL_DELTA}" == "1" ]] && printf '%s' "--require_visual_delta") \
  --visual_delta_threshold "${VISUAL_DELTA_THRESHOLD}" \
  --min_visual_delta_steps "${MIN_VISUAL_DELTA_STEPS}" \
  --visual_delta_stride "${VISUAL_DELTA_STRIDE}" \
  --action_contrast_weight "${ACTION_CONTRAST_WEIGHT}" \
  --action_contrast_margin "${ACTION_CONTRAST_MARGIN}" \
  --action_contrast_signal "${ACTION_CONTRAST_SIGNAL}" \
  --action_contrast_start "${ACTION_CONTRAST_START}" \
  --action_contrast_negative_modes "${ACTION_CONTRAST_NEGATIVE_MODES}" \
  --action_contrast_min_action_norm "${ACTION_CONTRAST_MIN_ACTION_NORM}" \
  --action_contrast_temporal_start "${ACTION_CONTRAST_TEMPORAL_START}" \
  --action_contrast_zero_mask_mode "${ACTION_CONTRAST_ZERO_MASK_MODE}" \
  --action_contrast_action_norm_weight "${ACTION_CONTRAST_ACTION_NORM_WEIGHT}" \
  --action_contrast_latent_delta_weight "${ACTION_CONTRAST_LATENT_DELTA_WEIGHT}" \
  --action_contrast_weight_clip "${ACTION_CONTRAST_WEIGHT_CLIP}" \
  --closed_loop_weight "${CLOSED_LOOP_WEIGHT}" \
  --closed_loop_start "${CLOSED_LOOP_START}" \
  --closed_loop_ctx "${CLOSED_LOOP_CTX}" \
  --closed_loop_horizon "${CLOSED_LOOP_HORIZON}" \
  --closed_loop_signal "${CLOSED_LOOP_SIGNAL}" \
  --closed_loop_contrast_weight "${CLOSED_LOOP_CONTRAST_WEIGHT}" \
  --closed_loop_contrast_margin "${CLOSED_LOOP_CONTRAST_MARGIN}" \
  --closed_loop_negative_modes "${CLOSED_LOOP_NEGATIVE_MODES}" \
  --closed_loop_min_action_norm "${CLOSED_LOOP_MIN_ACTION_NORM}" \
  --closed_loop_zero_mask_mode "${CLOSED_LOOP_ZERO_MASK_MODE}" \
  --closed_loop_action_norm_weight "${CLOSED_LOOP_ACTION_NORM_WEIGHT}" \
  --closed_loop_latent_delta_weight "${CLOSED_LOOP_LATENT_DELTA_WEIGHT}" \
  --closed_loop_weight_clip "${CLOSED_LOOP_WEIGHT_CLIP}" \
  --lr "${DYNAMICS_LR}" \
  --weight_decay 1e-2 \
  --max_steps "${DYNAMICS_MAX_STEPS}" \
  --grad_accum "${DYNAMICS_GRAD_ACCUM}" \
  --grad_clip 1.0 \
  --eval_every 0 \
  --log_every 100 \
  --save_every "${SAVE_EVERY}" \
  --wandb_mode disabled \
  --ckpt_dir "${DYNAMICS_DIR}" \
  "${RESUME_ARGS[@]}"

cd /workspace
python /workspace/sensenova_drone_agent/scripts/select_dreamer4_soar_dynamics_checkpoint.py \
  --data-dir "${RAW_DIR}" \
  --frames-dir "${FRAMES_DIR}" \
  --tasks-json "${TASKS_JSON}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --ckpt-dir "${DYNAMICS_DIR}" \
  --out-dir "${SELECTION_DIR}" \
  --max-checkpoints "${SELECT_MAX_CHECKPOINTS}" \
  --horizons "${SELECT_HORIZONS}" \
  --ctx-len 8 \
  --batch-size "${SELECT_BATCH_SIZE}" \
  --max-batches "${SELECT_MAX_BATCHES}" \
  --eval-d 0.25 \
  --action-dim "${ACTION_DIM}" \
  --action-features "${ACTION_FEATURES}" \
  --action-frame-offset "${ACTION_FRAME_OFFSET}" \
  $([[ "${REQUIRE_NON_NOOP}" == "1" ]] && printf '%s' "--require-non-noop") \
  --no-op-threshold "${NO_OP_THRESHOLD}" \
  --min-non-noop-steps "${MIN_NON_NOOP_STEPS}" \
  $([[ "${REQUIRE_VISUAL_DELTA}" == "1" ]] && printf '%s' "--require-visual-delta") \
  --visual-delta-threshold "${VISUAL_DELTA_THRESHOLD}" \
  --min-visual-delta-steps "${MIN_VISUAL_DELTA_STEPS}" \
  --visual-delta-stride "${VISUAL_DELTA_STRIDE}" \
  --negative-modes "${SELECT_NEGATIVE_MODES}" \
  --causal-min-ratio "${SELECT_CAUSAL_MIN_RATIO}" \
  --device cuda \
  --seed 20260602 \
  --force

echo "[${RUN_NAME}] complete $(date -Iseconds)"
