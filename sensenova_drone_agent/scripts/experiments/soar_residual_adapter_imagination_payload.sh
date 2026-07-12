#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-v1}"
OUT="${OUT:-/workspace/sensenova_drone_agent/output/soar_residual_adapter_imagination_${RUN_ID}}"
SOURCE_RUN="${SOURCE_RUN:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1}"
SOAR_ROOT="${SOAR_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}"
DROID_ROOT="${DROID_ROOT:-/workspace/sensenova_drone_agent/data/robotics/hf_action_exports/droid_lerobot_dreamer4}"
DATA_SOURCES="${DATA_SOURCES:-soar,droid}"

TASKS_JSON="${TASKS_JSON:-${SOURCE_RUN}/tasks_all_data.json}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${SOURCE_RUN}/tokenizer_ckpts/latest.pt}"
DYNAMICS_CKPT="${DYNAMICS_CKPT:-${SOURCE_RUN}/dynamics_ckpts/final_step_0275000.pt}"
RESIDUAL_ADAPTER_CKPT="${RESIDUAL_ADAPTER_CKPT:-/workspace/sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt}"
if [[ "${RESIDUAL_ADAPTER_CKPT}" == "none" || "${RESIDUAL_ADAPTER_CKPT}" == "NONE" || "${RESIDUAL_ADAPTER_CKPT}" == "off" || "${RESIDUAL_ADAPTER_CKPT}" == "OFF" ]]; then
  RESIDUAL_ADAPTER_CKPT=""
fi

SEQ_LEN="${SEQ_LEN:-16}"
CTX_LEN="${CTX_LEN:-8}"
IMAGINATION_HORIZON="${IMAGINATION_HORIZON:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BC_STEPS="${BC_STEPS:-1200}"
IMAGINATION_UPDATES="${IMAGINATION_UPDATES:-400}"
IMAGINATION_MODE="${IMAGINATION_MODE:-train}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:--1}"

ACTION_DIM="${ACTION_DIM:-49}"
RAW_ACTION_DIM="${RAW_ACTION_DIM:-12}"
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}"
POLICY_ACTION_SOURCE="${POLICY_ACTION_SOURCE:-raw}"
ACTION_CHUNK_LEN="${ACTION_CHUNK_LEN:-4}"

TRAIN_BALANCE_SPEC="${TRAIN_BALANCE_SPEC:-soar_game_positive=0.35,soar_game_active=0.25,hf_robot_active=0.40}"
TRAIN_SAMPLING_MODE="${TRAIN_SAMPLING_MODE:-dreamer4_reward_mixture}"
TRAIN_BALANCED_SAMPLES="${TRAIN_BALANCED_SAMPLES:-0}"
TRAIN_BALANCE_RETURN_THRESHOLD="${TRAIN_BALANCE_RETURN_THRESHOLD:-0.0}"
TRAIN_BALANCE_SEED="${TRAIN_BALANCE_SEED:-0}"
TRAIN_ACTION_ACTIVE_THRESHOLD="${TRAIN_ACTION_ACTIVE_THRESHOLD:-0.0}"
TRAIN_MIN_ACTION_ACTIVE_STEPS="${TRAIN_MIN_ACTION_ACTIVE_STEPS:-1}"

LEARNING_RATE="${LEARNING_RATE:-3e-4}"
IMAGINATION_LEARNING_RATE="${IMAGINATION_LEARNING_RATE:-2e-5}"
TARGET_NORMALIZATION="${TARGET_NORMALIZATION:-per_task}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"
VALUE_CLIP="${VALUE_CLIP:-5.0}"
EVAL_HOLDOUT_FRACTION="${EVAL_HOLDOUT_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-20260527}"
EVAL_SEED="${EVAL_SEED:-20260527}"
SEED="${SEED:-20260527}"

SELECT_BEST_IMAGINATION="${SELECT_BEST_IMAGINATION:-1}"
IMAGINATION_EVAL_EVERY="${IMAGINATION_EVAL_EVERY:-50}"
MIN_IMAGINATION_SELECTION_UPDATE="${MIN_IMAGINATION_SELECTION_UPDATE:-0}"
BEST_IMAGINATION_METRIC="${BEST_IMAGINATION_METRIC:-policy_minus_bc}"
EVAL_CAUSAL_DYNAMICS="${EVAL_CAUSAL_DYNAMICS:-1}"
DETACH_POLICY_LOG_PROB="${DETACH_POLICY_LOG_PROB:-1}"
IMAGINATION_DYNAMICS_ACTION_MODE="${IMAGINATION_DYNAMICS_ACTION_MODE:-policy}"
IMAGINATION_AGENT_ACTION_CONTEXT_MODE="${IMAGINATION_AGENT_ACTION_CONTEXT_MODE:-policy}"
REWARD_VALUE_ACTION_CONTEXT_MODE="${REWARD_VALUE_ACTION_CONTEXT_MODE:-policy}"
REWARD_CONTRAST_WEIGHT="${REWARD_CONTRAST_WEIGHT:-0.5}"
REWARD_CONTRAST_MARGIN="${REWARD_CONTRAST_MARGIN:-0.05}"
REWARD_CONTRAST_START="${REWARD_CONTRAST_START:-0}"
REWARD_CONTRAST_EVERY="${REWARD_CONTRAST_EVERY:-1}"
REWARD_CONTRAST_NEGATIVE_MODES="${REWARD_CONTRAST_NEGATIVE_MODES:-zero,shuffle}"
REWARD_CONTRAST_HORIZON="${REWARD_CONTRAST_HORIZON:-1}"
REWARD_CONTRAST_POSITIVE_THRESHOLD="${REWARD_CONTRAST_POSITIVE_THRESHOLD:-0.0}"
REWARD_CONTRAST_MIN_ACTION_NORM="${REWARD_CONTRAST_MIN_ACTION_NORM:-0.0}"
CAUSAL_POLICY_MODE="${CAUSAL_POLICY_MODE:-advantage_gate}"
CAUSAL_POLICY_NEGATIVE_MODES="${CAUSAL_POLICY_NEGATIVE_MODES:-zero,shuffle}"
CAUSAL_POLICY_MIN_MARGIN="${CAUSAL_POLICY_MIN_MARGIN:-0.0}"
CAUSAL_SHORTFALL_POLICY_WEIGHT="${CAUSAL_SHORTFALL_POLICY_WEIGHT:-0.0}"
CAUSAL_SHORTFALL_MARGIN="${CAUSAL_SHORTFALL_MARGIN:--1.0}"
SOURCE_EVAL_SOURCES="${SOURCE_EVAL_SOURCES:-}"
SOURCE_EVAL_BATCHES="${SOURCE_EVAL_BATCHES:-0}"
SOURCE_GATE_HARD_SOURCES="${SOURCE_GATE_HARD_SOURCES:-all,soar}"
SOURCE_GATE_SOFT_SOURCES="${SOURCE_GATE_SOFT_SOURCES:-droid}"
SOURCE_GATE_SOFT_MIN_MARGIN="${SOURCE_GATE_SOFT_MIN_MARGIN:--0.005}"
AUX_INVERSE_WEIGHT="${AUX_INVERSE_WEIGHT:-0.1}"
AUX_EFFECT_WEIGHT="${AUX_EFFECT_WEIGHT:-0.1}"
AUX_ACTION_EFFECT_MIN_NORM="${AUX_ACTION_EFFECT_MIN_NORM:-0.0}"
ADVANTAGE_MODE="${ADVANTAGE_MODE:-centered_sign}"
ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-bc_return}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-2.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-1.0}"
PRIOR_HINGE_WEIGHT="${PRIOR_HINGE_WEIGHT:-25.0}"
PRIOR_HINGE_TARGET="${PRIOR_HINGE_TARGET:-0.008}"
MEAN_PRIOR_WEIGHT="${MEAN_PRIOR_WEIGHT:-10.0}"
MEAN_PRIOR_HINGE_WEIGHT="${MEAN_PRIOR_HINGE_WEIGHT:-100.0}"
MEAN_PRIOR_HINGE_TARGET="${MEAN_PRIOR_HINGE_TARGET:-0.004}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.10}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.0005}"
LOG_STD_INIT="${LOG_STD_INIT:--2.5}"
TRAIN_VALUE_DURING_IMAGINATION="${TRAIN_VALUE_DURING_IMAGINATION:-0}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/logs/payload.log") 2>&1

echo "[soar-residual-adapter-imagination] started $(date -Is)"
echo "[soar-residual-adapter-imagination] out=${OUT}"
echo "[soar-residual-adapter-imagination] source_run=${SOURCE_RUN}"
echo "[soar-residual-adapter-imagination] tokenizer=${TOKENIZER_CKPT}"
echo "[soar-residual-adapter-imagination] dynamics=${DYNAMICS_CKPT}"
echo "[soar-residual-adapter-imagination] residual_adapter=${RESIDUAL_ADAPTER_CKPT}"
echo "[soar-residual-adapter-imagination] train_sampling_mode=${TRAIN_SAMPLING_MODE}"
echo "[soar-residual-adapter-imagination] train_balance_spec=${TRAIN_BALANCE_SPEC}"
echo "[soar-residual-adapter-imagination] data_sources=${DATA_SOURCES}"
echo "[soar-residual-adapter-imagination] bc_steps=${BC_STEPS} imagination_updates=${IMAGINATION_UPDATES}"
echo "[soar-residual-adapter-imagination] causal_policy_mode=${CAUSAL_POLICY_MODE} eval_causal=${EVAL_CAUSAL_DYNAMICS}"

RAW_DIRS=()
FRAME_DIRS=()
IFS=',' read -r -a DATA_SOURCE_ITEMS <<< "${DATA_SOURCES}"
for item in "${DATA_SOURCE_ITEMS[@]}"; do
  source_name="$(echo "${item}" | xargs)"
  case "${source_name}" in
    soar)
      RAW_DIRS+=("${SOAR_ROOT}/raw")
      FRAME_DIRS+=("${SOAR_ROOT}/frames")
      ;;
    droid)
      RAW_DIRS+=("${DROID_ROOT}/raw")
      FRAME_DIRS+=("${DROID_ROOT}/frames")
      ;;
    "")
      ;;
    *)
      echo "[soar-residual-adapter-imagination] unknown DATA_SOURCES item: ${source_name}" >&2
      exit 2
      ;;
  esac
done
if [[ "${#RAW_DIRS[@]}" -eq 0 ]]; then
  echo "[soar-residual-adapter-imagination] DATA_SOURCES selected no datasets" >&2
  exit 2
fi

REQUIRED_PATHS=("${TASKS_JSON}" "${TOKENIZER_CKPT}" "${DYNAMICS_CKPT}" "${RAW_DIRS[@]}" "${FRAME_DIRS[@]}")
if [[ -n "${RESIDUAL_ADAPTER_CKPT}" ]]; then
  REQUIRED_PATHS+=("${RESIDUAL_ADAPTER_CKPT}")
fi

for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[soar-residual-adapter-imagination] missing required path: ${path}" >&2
    exit 1
  fi
done

ARGS=()
for raw_dir in "${RAW_DIRS[@]}"; do
  ARGS+=(--data-dir "${raw_dir}")
done
for frame_dir in "${FRAME_DIRS[@]}"; do
  ARGS+=(--frames-dir "${frame_dir}")
done

SELECT_BEST_ARGS=()
if [[ "${SELECT_BEST_IMAGINATION}" == "1" || "${SELECT_BEST_IMAGINATION}" == "true" || "${SELECT_BEST_IMAGINATION}" == "TRUE" ]]; then
  SELECT_BEST_ARGS+=(--select-best-imagination)
fi

LOG_PROB_ARGS=()
if [[ "${DETACH_POLICY_LOG_PROB}" == "0" || "${DETACH_POLICY_LOG_PROB}" == "false" || "${DETACH_POLICY_LOG_PROB}" == "FALSE" ]]; then
  LOG_PROB_ARGS+=(--no-detach-policy-log-prob)
fi

CAUSAL_EVAL_ARGS=()
if [[ "${EVAL_CAUSAL_DYNAMICS}" == "1" || "${EVAL_CAUSAL_DYNAMICS}" == "true" || "${EVAL_CAUSAL_DYNAMICS}" == "TRUE" ]]; then
  CAUSAL_EVAL_ARGS+=(--eval-causal-dynamics)
fi

TRAIN_VALUE_ARGS=()
if [[ "${TRAIN_VALUE_DURING_IMAGINATION}" == "1" || "${TRAIN_VALUE_DURING_IMAGINATION}" == "true" || "${TRAIN_VALUE_DURING_IMAGINATION}" == "TRUE" ]]; then
  TRAIN_VALUE_ARGS+=(--train-value-during-imagination)
fi

RESIDUAL_ADAPTER_ARGS=()
if [[ -n "${RESIDUAL_ADAPTER_CKPT}" ]]; then
  RESIDUAL_ADAPTER_ARGS+=(--residual-adapter-ckpt "${RESIDUAL_ADAPTER_CKPT}")
fi

python /workspace/sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py \
  "${ARGS[@]}" \
  "${SELECT_BEST_ARGS[@]}" \
  "${LOG_PROB_ARGS[@]}" \
  "${CAUSAL_EVAL_ARGS[@]}" \
  "${TRAIN_VALUE_ARGS[@]}" \
  --tasks-json "${TASKS_JSON}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --dynamics-ckpt "${DYNAMICS_CKPT}" \
  "${RESIDUAL_ADAPTER_ARGS[@]}" \
  --out-dir "${OUT}" \
  --seq-len "${SEQ_LEN}" \
  --ctx-len "${CTX_LEN}" \
  --imagination-horizon "${IMAGINATION_HORIZON}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --bc-steps "${BC_STEPS}" \
  --imagination-updates "${IMAGINATION_UPDATES}" \
  --eval-batches "${EVAL_BATCHES}" \
  --imagination-eval-every "${IMAGINATION_EVAL_EVERY}" \
  --min-imagination-selection-update "${MIN_IMAGINATION_SELECTION_UPDATE}" \
  --best-imagination-metric "${BEST_IMAGINATION_METRIC}" \
  --action-dim "${ACTION_DIM}" \
  --raw-action-dim "${RAW_ACTION_DIM}" \
  --action-features "${ACTION_FEATURES}" \
  --policy-action-source "${POLICY_ACTION_SOURCE}" \
  --action-chunk-len "${ACTION_CHUNK_LEN}" \
  --action-frame-offset "${ACTION_FRAME_OFFSET}" \
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
  --imagination-dynamics-action-mode "${IMAGINATION_DYNAMICS_ACTION_MODE}" \
  --imagination-agent-action-context-mode "${IMAGINATION_AGENT_ACTION_CONTEXT_MODE}" \
  --reward-value-action-context-mode "${REWARD_VALUE_ACTION_CONTEXT_MODE}" \
  --reward-contrast-weight "${REWARD_CONTRAST_WEIGHT}" \
  --reward-contrast-margin "${REWARD_CONTRAST_MARGIN}" \
  --reward-contrast-start "${REWARD_CONTRAST_START}" \
  --reward-contrast-every "${REWARD_CONTRAST_EVERY}" \
  --reward-contrast-negative-modes "${REWARD_CONTRAST_NEGATIVE_MODES}" \
  --reward-contrast-horizon "${REWARD_CONTRAST_HORIZON}" \
  --reward-contrast-positive-threshold "${REWARD_CONTRAST_POSITIVE_THRESHOLD}" \
  --reward-contrast-min-action-norm "${REWARD_CONTRAST_MIN_ACTION_NORM}" \
  --causal-policy-mode "${CAUSAL_POLICY_MODE}" \
  --causal-policy-negative-modes "${CAUSAL_POLICY_NEGATIVE_MODES}" \
  --causal-policy-min-margin "${CAUSAL_POLICY_MIN_MARGIN}" \
  --causal-shortfall-policy-weight "${CAUSAL_SHORTFALL_POLICY_WEIGHT}" \
  --causal-shortfall-margin "${CAUSAL_SHORTFALL_MARGIN}" \
  --source-eval-sources "${SOURCE_EVAL_SOURCES}" \
  --source-eval-batches "${SOURCE_EVAL_BATCHES}" \
  --source-gate-hard-sources "${SOURCE_GATE_HARD_SOURCES}" \
  --source-gate-soft-sources "${SOURCE_GATE_SOFT_SOURCES}" \
  --source-gate-soft-min-margin "${SOURCE_GATE_SOFT_MIN_MARGIN}" \
  --aux-inverse-weight "${AUX_INVERSE_WEIGHT}" \
  --aux-effect-weight "${AUX_EFFECT_WEIGHT}" \
  --aux-action-effect-min-norm "${AUX_ACTION_EFFECT_MIN_NORM}" \
  --prior-weight "${PRIOR_WEIGHT}" \
  --prior-hinge-weight "${PRIOR_HINGE_WEIGHT}" \
  --prior-hinge-target "${PRIOR_HINGE_TARGET}" \
  --mean-prior-weight "${MEAN_PRIOR_WEIGHT}" \
  --mean-prior-hinge-weight "${MEAN_PRIOR_HINGE_WEIGHT}" \
  --mean-prior-hinge-target "${MEAN_PRIOR_HINGE_TARGET}" \
  --value-loss-weight "${VALUE_LOSS_WEIGHT}" \
  --entropy-weight "${ENTROPY_WEIGHT}" \
  --log-std-init "${LOG_STD_INIT}" \
  --train-sampling-mode "${TRAIN_SAMPLING_MODE}" \
  --train-balance-spec "${TRAIN_BALANCE_SPEC}" \
  --train-balance-return-threshold "${TRAIN_BALANCE_RETURN_THRESHOLD}" \
  --train-balanced-samples "${TRAIN_BALANCED_SAMPLES}" \
  --train-balance-seed "${TRAIN_BALANCE_SEED}" \
  --train-action-active-threshold "${TRAIN_ACTION_ACTIVE_THRESHOLD}" \
  --train-min-action-active-steps "${TRAIN_MIN_ACTION_ACTIVE_STEPS}" \
  --device cuda \
  --eval-seed "${EVAL_SEED}" \
  --seed "${SEED}"

echo "[soar-residual-adapter-imagination] finished $(date -Is)"
