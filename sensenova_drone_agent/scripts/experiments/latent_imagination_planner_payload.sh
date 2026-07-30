#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-all_data_v1}"
OUT="${OUT:-/workspace/sensenova_drone_agent/output/latent_imagination_planner_${RUN_ID}}"
MANIFEST_JSON="${MANIFEST_JSON:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/all_data_manifest.json}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/tokenizer_ckpts/latest.pt}"
TASKS_JSON="${TASKS_JSON:-}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/logs/payload.log") 2>&1

echo "[latent-imagination-planner] started $(date -Is)"
echo "[latent-imagination-planner] out=${OUT}"
echo "[latent-imagination-planner] manifest=${MANIFEST_JSON}"
echo "[latent-imagination-planner] tokenizer=${TOKENIZER_CKPT}"
echo "[latent-imagination-planner] resume=${RESUME_CKPT:-}"

for path in "${MANIFEST_JSON}" "${TOKENIZER_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[latent-imagination-planner] missing required path: ${path}" >&2
    exit 1
  fi
done

filter_args=()
if [[ "${REQUIRE_NON_NOOP:-1}" == "1" || "${REQUIRE_NON_NOOP:-1}" == "true" ]]; then
  filter_args+=(--require-non-noop)
fi
if [[ "${REQUIRE_VISUAL_DELTA:-1}" == "1" || "${REQUIRE_VISUAL_DELTA:-1}" == "true" ]]; then
  filter_args+=(--require-visual-delta)
fi
if [[ "${NO_MANIFEST_WEIGHTS:-0}" == "1" || "${NO_MANIFEST_WEIGHTS:-0}" == "true" ]]; then
  filter_args+=(--no-manifest-weights)
fi
if [[ "${PLAN_UNIT_NORM:-0}" == "1" || "${PLAN_UNIT_NORM:-0}" == "true" ]]; then
  filter_args+=(--plan-unit-norm)
fi
if [[ "${PLAN_STEP_CONDITIONING:-0}" == "1" || "${PLAN_STEP_CONDITIONING:-0}" == "true" ]]; then
  filter_args+=(--plan-step-conditioning)
fi
if [[ "${BC_ENCODER_GRAD:-0}" == "1" || "${BC_ENCODER_GRAD:-0}" == "true" ]]; then
  filter_args+=(--bc-encoder-grad)
fi

tasks_args=()
if [[ -n "${TASKS_JSON}" ]]; then
  tasks_args+=(--tasks-json "${TASKS_JSON}")
fi

resume_args=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  if [[ ! -e "${RESUME_CKPT}" ]]; then
    echo "[latent-imagination-planner] missing resume checkpoint: ${RESUME_CKPT}" >&2
    exit 1
  fi
  resume_args+=(--resume-ckpt "${RESUME_CKPT}")
fi

python /workspace/sensenova_drone_agent/scripts/train_latent_imagination_planner.py \
  --manifest-json "${MANIFEST_JSON}" \
  --source-names "${SOURCE_NAMES:-}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --out-dir "${OUT}" \
  "${resume_args[@]}" \
  --seq-len "${SEQ_LEN:-24}" \
  --ctx-len "${CTX_LEN:-8}" \
  --horizon "${HORIZON:-8}" \
  --img-size "${IMG_SIZE:-128}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --max-steps "${MAX_STEPS:-500000}" \
  --eval-every "${EVAL_EVERY:-1000}" \
  --eval-batches "${EVAL_BATCHES:-64}" \
  --save-every "${SAVE_EVERY:-10000}" \
  --trace-every "${TRACE_EVERY:-5000}" \
  --action-dim "${ACTION_DIM:-49}" \
  --raw-action-dim "${RAW_ACTION_DIM:-49}" \
  --action-features "${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  --action-frame-offset "${ACTION_FRAME_OFFSET:--1}" \
  --hidden-dim "${HIDDEN_DIM:-1024}" \
  --plan-dim "${PLAN_DIM:-128}" \
  --num-candidates "${NUM_CANDIDATES:-16}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --future-loss-weight "${FUTURE_LOSS_WEIGHT:-1.0}" \
  --reward-loss-weight "${REWARD_LOSS_WEIGHT:-0.25}" \
  --inverse-loss-weight "${INVERSE_LOSS_WEIGHT:-0.25}" \
  --contrast-weight "${CONTRAST_WEIGHT:-1.0}" \
  --contrast-margin "${CONTRAST_MARGIN:-0.02}" \
  --contrast-relative-margin "${CONTRAST_RELATIVE_MARGIN:-0.0}" \
  --plan-l2-weight "${PLAN_L2_WEIGHT:-0.0001}" \
  --effect-loss-weight "${EFFECT_LOSS_WEIGHT:-0.10}" \
  --rank-loss-weight "${RANK_LOSS_WEIGHT:-0.0}" \
  --rank-num-bank "${RANK_NUM_BANK:-4}" \
  --rank-num-matched "${RANK_NUM_MATCHED:-4}" \
  --rank-margin "${RANK_MARGIN:-0.05}" \
  --rank-mse-gap "${RANK_MSE_GAP:-1.1}" \
  --inverse-plan-dropout "${INVERSE_PLAN_DROPOUT:-0.0}" \
  --inverse-imagined-weight "${INVERSE_IMAGINED_WEIGHT:-0.0}" \
  --inverse-cross-weight "${INVERSE_CROSS_WEIGHT:-0.0}" \
  --gamma "${GAMMA:-0.997}" \
  --score-plan-dropout "${SCORE_PLAN_DROPOUT:-0.0}" \
  --bc-head-weight "${BC_HEAD_WEIGHT:-0.0}" \
  --horizon-curriculum-max "${HORIZON_CURRICULUM_MAX:-0}" \
  --horizon-curriculum-weight "${HORIZON_CURRICULUM_WEIGHT:-0.5}" \
  --contrast-modes "${CONTRAST_MODES:-shuffle,zero,time_shift,time_shift2,time_perm,time_reverse}" \
  --no-op-threshold "${NO_OP_THRESHOLD:-0.05}" \
  --min-non-noop-steps "${MIN_NON_NOOP_STEPS:-4}" \
  --reward-filter-mode "${REWARD_FILTER_MODE:-none}" \
  --reward-signal-threshold "${REWARD_SIGNAL_THRESHOLD:-0.0}" \
  --min-reward-signal-steps "${MIN_REWARD_SIGNAL_STEPS:-1}" \
  --visual-delta-threshold "${VISUAL_DELTA_THRESHOLD:-0.005}" \
  --min-visual-delta-steps "${MIN_VISUAL_DELTA_STEPS:-4}" \
  --visual-delta-stride "${VISUAL_DELTA_STRIDE:-4}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-20260607}" \
  "${tasks_args[@]}" \
  "${filter_args[@]}"

echo "[latent-imagination-planner] finished $(date -Is)"
