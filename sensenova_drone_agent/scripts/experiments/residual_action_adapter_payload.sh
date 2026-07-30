#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${OUT:-/workspace/sensenova_drone_agent/output/residual_action_adapter_v1}"
mkdir -p "${OUT}"

extra_args=()
if [[ "${REQUIRE_NON_NOOP:-1}" == "1" || "${REQUIRE_NON_NOOP:-1}" == "true" ]]; then
  extra_args+=(--require-non-noop)
fi
if [[ "${REQUIRE_VISUAL_DELTA:-0}" == "1" || "${REQUIRE_VISUAL_DELTA:-0}" == "true" ]]; then
  extra_args+=(--require-visual-delta)
fi
if [[ "${RANDOM_SIGNAL:-0}" == "1" || "${RANDOM_SIGNAL:-0}" == "true" ]]; then
  extra_args+=(--random-signal)
fi

python3 /workspace/sensenova_drone_agent/scripts/train_residual_action_adapter.py \
  --manifest-json "${MANIFEST_JSON:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/all_data_manifest.json}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/tokenizer_ckpts/latest.pt}" \
  --dynamics-ckpt "${DYNAMICS_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/dynamics_ckpts/final_step_0275000.pt}" \
  --tasks-json "${TASKS_JSON:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/tasks_all_data.json}" \
  --source-names "${SOURCE_NAMES:-soar_native_v2,hf_robot_droid_lerobot_dreamer4}" \
  --out-dir "${OUT}" \
  --seq-len "${SEQ_LEN:-16}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --train-steps "${TRAIN_STEPS:-12000}" \
  --eval-batches "${EVAL_BATCHES:-256}" \
  --lr "${LR:-3e-4}" \
  --hidden "${HIDDEN:-256}" \
  --residual-scale "${RESIDUAL_SCALE:-1.0}" \
  --contrast-weight "${CONTRAST_WEIGHT:-1.0}" \
  --contrast-margin "${CONTRAST_MARGIN:-0.02}" \
  --contrast-modes "${CONTRAST_MODES:-shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse}" \
  --contrast-action-norm-weight "${CONTRAST_ACTION_NORM_WEIGHT:-0.0}" \
  --contrast-latent-delta-weight "${CONTRAST_LATENT_DELTA_WEIGHT:-0.0}" \
  --contrast-weight-clip "${CONTRAST_WEIGHT_CLIP:-10.0}" \
  --signal-level "${SIGNAL_LEVEL:-0.1}" \
  --action-frame-offset "${ACTION_FRAME_OFFSET:--1}" \
  --action-dim "${ACTION_DIM:-49}" \
  --action-features "${ACTION_FEATURES:-current,prev,delta,mean4,norm}" \
  --no-op-threshold "${NO_OP_THRESHOLD:-0.1}" \
  --min-non-noop-steps "${MIN_NON_NOOP_STEPS:-12}" \
  --visual-delta-threshold "${VISUAL_DELTA_THRESHOLD:-0.01}" \
  --min-visual-delta-steps "${MIN_VISUAL_DELTA_STEPS:-8}" \
  --visual-delta-stride "${VISUAL_DELTA_STRIDE:-4}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-53}" \
  --num-workers "${NUM_WORKERS:-2}" \
  "${extra_args[@]}" \
  2>&1 | tee "${OUT}/train.log"
