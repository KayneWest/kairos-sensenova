#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="/workspace/.pydeps:/workspace/dreamer4/dreamer4:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-v1}"
OUT="${OUT:-/workspace/sensenova_drone_agent/output/dreamer4_all_data_native_${RUN_ID}}"
TASKS_JSON="${TASKS_JSON:-${OUT}/tasks_all_data.json}"
MANIFEST_JSON="${MANIFEST_JSON:-${OUT}/all_data_manifest.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

DREAMER_RAW="${DREAMER_RAW:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4}"
DREAMER_SHARDS="${DREAMER_SHARDS:-/workspace/sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full}"
SOAR_ROOT="${SOAR_ROOT:-/workspace/sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast}"
ROBONET_ROOT="${ROBONET_ROOT:-/workspace/sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64}"
HF_ACTION_EXPORT_ROOT="${HF_ACTION_EXPORT_ROOT:-/workspace/sensenova_drone_agent/data/robotics/hf_action_exports}"
HF_ACTION_DATASETS="${HF_ACTION_DATASETS:-droid_lerobot_dreamer4,fractal20220817_data_lerobot_dreamer4,bridge_orig_lerobot_dreamer4}"
SOURCE_DEFAULT_WEIGHT="${SOURCE_DEFAULT_WEIGHT:-1}"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-}"

BASE_TOKENIZER_CKPT="${BASE_TOKENIZER_CKPT:-/workspace/sensenova_drone_agent/output/dreamer4_hf_long_run_v1/tokenizer_ckpts/latest.pt}"
BASE_DYNAMICS_CKPT="${BASE_DYNAMICS_CKPT:-}"

TOKENIZER_STEPS="${TOKENIZER_STEPS:-25000}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-4}"
TOKENIZER_SEQ_LEN="${TOKENIZER_SEQ_LEN:-8}"
TOKENIZER_GRAD_ACCUM="${TOKENIZER_GRAD_ACCUM:-4}"
SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"

DYNAMICS_STEPS="${DYNAMICS_STEPS:-150000}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-4}"
DYNAMICS_SEQ_LEN="${DYNAMICS_SEQ_LEN:-16}"
DYNAMICS_GRAD_ACCUM="${DYNAMICS_GRAD_ACCUM:-4}"
DYNAMICS_D_MODEL="${DYNAMICS_D_MODEL:-128}"
DYNAMICS_DEPTH="${DYNAMICS_DEPTH:-4}"
ACTION_DIM="${ACTION_DIM:-49}"
ACTION_FEATURES="${ACTION_FEATURES:-current,prev,delta,mean4,norm}"
ACTION_FRAME_OFFSET="${ACTION_FRAME_OFFSET:-0}"
REQUIRE_NON_NOOP="${REQUIRE_NON_NOOP:-0}"
NO_OP_THRESHOLD="${NO_OP_THRESHOLD:-0.0}"
MIN_NON_NOOP_STEPS="${MIN_NON_NOOP_STEPS:-1}"
REWARD_FILTER_MODE="${REWARD_FILTER_MODE:-none}"
REWARD_SIGNAL_THRESHOLD="${REWARD_SIGNAL_THRESHOLD:-0.0}"
MIN_REWARD_SIGNAL_STEPS="${MIN_REWARD_SIGNAL_STEPS:-1}"
REQUIRE_VISUAL_DELTA="${REQUIRE_VISUAL_DELTA:-0}"
VISUAL_DELTA_THRESHOLD="${VISUAL_DELTA_THRESHOLD:-0.0}"
MIN_VISUAL_DELTA_STEPS="${MIN_VISUAL_DELTA_STEPS:-1}"
VISUAL_DELTA_STRIDE="${VISUAL_DELTA_STRIDE:-4}"
ACTION_CONTRAST_WEIGHT="${ACTION_CONTRAST_WEIGHT:-0.5}"
ACTION_CONTRAST_MARGIN="${ACTION_CONTRAST_MARGIN:-0.01}"
ACTION_CONTRAST_SIGNAL="${ACTION_CONTRAST_SIGNAL:-0.1}"
ACTION_CONTRAST_START="${ACTION_CONTRAST_START:-5000}"
ACTION_CONTRAST_NEGATIVE_MODES="${ACTION_CONTRAST_NEGATIVE_MODES:-shuffle,zero,time_shift}"
ACTION_CONTRAST_MIN_ACTION_NORM="${ACTION_CONTRAST_MIN_ACTION_NORM:-0.0}"
ACTION_CONTRAST_TEMPORAL_START="${ACTION_CONTRAST_TEMPORAL_START:-1}"
ACTION_CONTRAST_ZERO_MASK_MODE="${ACTION_CONTRAST_ZERO_MASK_MODE:-original}"
ACTION_CONTRAST_ACTION_NORM_WEIGHT="${ACTION_CONTRAST_ACTION_NORM_WEIGHT:-0.0}"
ACTION_CONTRAST_LATENT_DELTA_WEIGHT="${ACTION_CONTRAST_LATENT_DELTA_WEIGHT:-0.0}"
ACTION_CONTRAST_WEIGHT_CLIP="${ACTION_CONTRAST_WEIGHT_CLIP:-10.0}"
CLOSED_LOOP_WEIGHT="${CLOSED_LOOP_WEIGHT:-0.0}"
CLOSED_LOOP_START="${CLOSED_LOOP_START:-0}"
CLOSED_LOOP_CTX="${CLOSED_LOOP_CTX:-8}"
CLOSED_LOOP_HORIZON="${CLOSED_LOOP_HORIZON:-4}"
CLOSED_LOOP_SIGNAL="${CLOSED_LOOP_SIGNAL:-0.1}"
CLOSED_LOOP_BACKPROP_HISTORY="${CLOSED_LOOP_BACKPROP_HISTORY:-0}"
CLOSED_LOOP_CONTRAST_WEIGHT="${CLOSED_LOOP_CONTRAST_WEIGHT:-0.0}"
CLOSED_LOOP_CONTRAST_MARGIN="${CLOSED_LOOP_CONTRAST_MARGIN:-0.01}"
CLOSED_LOOP_NEGATIVE_MODES="${CLOSED_LOOP_NEGATIVE_MODES:-shuffle,zero,time_shift}"
CLOSED_LOOP_MIN_ACTION_NORM="${CLOSED_LOOP_MIN_ACTION_NORM:-0.0}"
CLOSED_LOOP_ZERO_MASK_MODE="${CLOSED_LOOP_ZERO_MASK_MODE:-original}"
CLOSED_LOOP_ACTION_NORM_WEIGHT="${CLOSED_LOOP_ACTION_NORM_WEIGHT:-0.0}"
CLOSED_LOOP_LATENT_DELTA_WEIGHT="${CLOSED_LOOP_LATENT_DELTA_WEIGHT:-0.0}"
CLOSED_LOOP_WEIGHT_CLIP="${CLOSED_LOOP_WEIGHT_CLIP:-10.0}"
SELF_FRACTION="${SELF_FRACTION:-0.25}"
BOOTSTRAP_START="${BOOTSTRAP_START:-5000}"
DYNAMICS_LR="${DYNAMICS_LR:-5e-5}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-256}"
EVAL_CAUSAL_MIN_RATIO="${EVAL_CAUSAL_MIN_RATIO:-1.02}"
EVAL_ACTION_FRAME_OFFSET="${EVAL_ACTION_FRAME_OFFSET:-${ACTION_FRAME_OFFSET}}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/logs/payload.log") 2>&1

echo "[all-data] started $(date -Is)"
echo "[all-data] out=${OUT}"
echo "[all-data] dreamer_raw=${DREAMER_RAW}"
echo "[all-data] dreamer_shards=${DREAMER_SHARDS}"
echo "[all-data] soar_root=${SOAR_ROOT}"
echo "[all-data] robonet_root=${ROBONET_ROOT}"
echo "[all-data] hf_action_export_root=${HF_ACTION_EXPORT_ROOT}"
echo "[all-data] hf_action_datasets=${HF_ACTION_DATASETS}"
echo "[all-data] source_default_weight=${SOURCE_DEFAULT_WEIGHT} source_weights=${SOURCE_WEIGHTS}"
echo "[all-data] nproc_per_node=${NPROC_PER_NODE}"
echo "[all-data] action_dim=${ACTION_DIM} action_features=${ACTION_FEATURES} action_frame_offset=${ACTION_FRAME_OFFSET}"
echo "[all-data] require_non_noop=${REQUIRE_NON_NOOP} no_op_threshold=${NO_OP_THRESHOLD} min_non_noop_steps=${MIN_NON_NOOP_STEPS}"
echo "[all-data] reward_filter_mode=${REWARD_FILTER_MODE} reward_signal_threshold=${REWARD_SIGNAL_THRESHOLD} min_reward_signal_steps=${MIN_REWARD_SIGNAL_STEPS}"
echo "[all-data] require_visual_delta=${REQUIRE_VISUAL_DELTA} visual_delta_threshold=${VISUAL_DELTA_THRESHOLD} min_visual_delta_steps=${MIN_VISUAL_DELTA_STEPS} visual_delta_stride=${VISUAL_DELTA_STRIDE}"
echo "[all-data] action_contrast_weight=${ACTION_CONTRAST_WEIGHT} margin=${ACTION_CONTRAST_MARGIN} signal=${ACTION_CONTRAST_SIGNAL} start=${ACTION_CONTRAST_START}"
echo "[all-data] action_contrast_negative_modes=${ACTION_CONTRAST_NEGATIVE_MODES} min_action_norm=${ACTION_CONTRAST_MIN_ACTION_NORM} temporal_start=${ACTION_CONTRAST_TEMPORAL_START} zero_mask_mode=${ACTION_CONTRAST_ZERO_MASK_MODE}"
echo "[all-data] action_contrast_action_norm_weight=${ACTION_CONTRAST_ACTION_NORM_WEIGHT} latent_delta_weight=${ACTION_CONTRAST_LATENT_DELTA_WEIGHT} weight_clip=${ACTION_CONTRAST_WEIGHT_CLIP}"
echo "[all-data] closed_loop_weight=${CLOSED_LOOP_WEIGHT} start=${CLOSED_LOOP_START} ctx=${CLOSED_LOOP_CTX} horizon=${CLOSED_LOOP_HORIZON} signal=${CLOSED_LOOP_SIGNAL} backprop_history=${CLOSED_LOOP_BACKPROP_HISTORY}"
echo "[all-data] closed_loop_contrast_weight=${CLOSED_LOOP_CONTRAST_WEIGHT} margin=${CLOSED_LOOP_CONTRAST_MARGIN} negative_modes=${CLOSED_LOOP_NEGATIVE_MODES} min_action_norm=${CLOSED_LOOP_MIN_ACTION_NORM} zero_mask_mode=${CLOSED_LOOP_ZERO_MASK_MODE}"
echo "[all-data] closed_loop_action_norm_weight=${CLOSED_LOOP_ACTION_NORM_WEIGHT} latent_delta_weight=${CLOSED_LOOP_LATENT_DELTA_WEIGHT} weight_clip=${CLOSED_LOOP_WEIGHT_CLIP}"

export OUT TASKS_JSON MANIFEST_JSON DREAMER_RAW DREAMER_SHARDS SOAR_ROOT ROBONET_ROOT HF_ACTION_EXPORT_ROOT HF_ACTION_DATASETS SOURCE_DEFAULT_WEIGHT SOURCE_WEIGHTS ACTION_DIM ACTION_FEATURES ACTION_FRAME_OFFSET
python - <<'PY'
import json
import os
from pathlib import Path
import torch

out = Path(os.environ["OUT"])
tasks_json = Path(os.environ["TASKS_JSON"])
manifest_json = Path(os.environ["MANIFEST_JSON"])
dreamer_raw = Path(os.environ["DREAMER_RAW"])
dreamer_shards = Path(os.environ["DREAMER_SHARDS"])
soar_root = Path(os.environ["SOAR_ROOT"])
robonet_root = Path(os.environ["ROBONET_ROOT"])
hf_action_root = Path(os.environ["HF_ACTION_EXPORT_ROOT"])
hf_action_datasets = [item.strip() for item in os.environ.get("HF_ACTION_DATASETS", "").split(",") if item.strip()]
action_dim = int(os.environ["ACTION_DIM"])
action_features = os.environ["ACTION_FEATURES"]
source_default_weight = int(os.environ.get("SOURCE_DEFAULT_WEIGHT", "1"))
source_weight_map = {}
for item in os.environ.get("SOURCE_WEIGHTS", "").split(","):
    item = item.strip()
    if not item:
        continue
    if "=" not in item:
        raise SystemExit(f"invalid SOURCE_WEIGHTS item '{item}', expected name=integer_weight")
    key, value = item.split("=", 1)
    source_weight_map[key.strip()] = int(value.strip())


def source_weight(name: str) -> int:
    return max(0, int(source_weight_map.get(name, source_default_weight)))


sources = [
    {
        "name": "dreamer4_hf_expert",
        "raw": dreamer_raw / "expert",
        "frames": dreamer_shards / "expert",
        "tasks_json": Path("/workspace/dreamer4/tasks.json"),
    },
    {
        "name": "dreamer4_hf_mixed_small",
        "raw": dreamer_raw / "mixed-small",
        "frames": dreamer_shards / "mixed-small",
        "tasks_json": Path("/workspace/dreamer4/tasks.json"),
    },
    {
        "name": "dreamer4_hf_mixed_large",
        "raw": dreamer_raw / "mixed-large",
        "frames": dreamer_shards / "mixed-large",
        "tasks_json": Path("/workspace/dreamer4/tasks.json"),
    },
    {
        "name": "soar_native_v2",
        "raw": soar_root / "raw",
        "frames": soar_root / "frames",
        "tasks_json": soar_root / "tasks.json",
    },
    {
        "name": "robonet_sample_64",
        "raw": robonet_root / "raw",
        "frames": robonet_root / "frames",
        "tasks_json": robonet_root / "tasks.json",
    },
]
for dataset in hf_action_datasets:
    root = hf_action_root / dataset
    sources.append(
        {
            "name": f"hf_robot_{dataset}",
            "raw": root / "raw",
            "frames": root / "frames",
            "tasks_json": root / "tasks.json",
        }
    )

merged = {}
source_rows = []
for source in sources:
    weight = source_weight(source["name"])
    raw = source["raw"]
    frames = source["frames"]
    if not raw.exists():
        raise SystemExit(f"missing raw path for {source['name']}: {raw}")
    if not frames.exists():
        raise SystemExit(f"missing frames path for {source['name']}: {frames}")
    meta = {}
    if source["tasks_json"].exists():
        meta = json.loads(source["tasks_json"].read_text())
    pt_tasks = sorted(path.stem for path in raw.glob("*.pt"))
    frame_tasks = sorted(path.name for path in frames.iterdir() if path.is_dir())
    usable_tasks = sorted(set(pt_tasks) & set(frame_tasks))
    for task in usable_tasks:
        row = dict(meta.get(task, {}))
        row.setdefault("action_dim", None)
        if row["action_dim"] is None:
            try:
                td = torch.load(raw / f"{task}.pt", map_location="cpu", weights_only=False)
                action = td.get("action")
                if action is not None:
                    row["action_dim"] = int(action.shape[-1]) if action.ndim > 1 else 1
            except Exception:
                row["action_dim"] = action_dim
        row.setdefault("text", f"{source['name']}: {task}")
        if task in merged and merged[task] != row:
            # Same control task can appear in several Dreamer4 splits. Keep the first
            # equivalent task name but preserve the widest action dim.
            prev = dict(merged[task])
            try:
                prev["action_dim"] = max(int(prev.get("action_dim", 0)), int(row.get("action_dim", 0)))
            except Exception:
                pass
            merged[task] = prev
        else:
            merged[task] = row
    source_rows.append(
        {
            "name": source["name"],
            "raw": str(raw),
            "frames": str(frames),
            "tasks_json": str(source["tasks_json"]),
            "weight": weight,
            "raw_task_count": len(pt_tasks),
            "frame_task_count": len(frame_tasks),
            "usable_task_count": len(usable_tasks),
        }
    )

tasks_json.write_text(json.dumps(dict(sorted(merged.items())), indent=2), encoding="utf-8")
manifest = {
    "phase": "all_data_native_dreamer4",
    "sources": source_rows,
    "tasks_json": str(tasks_json),
    "merged_task_count": len(merged),
    "action_dim": action_dim,
    "action_features": action_features,
    "action_frame_offset": int(os.environ.get("ACTION_FRAME_OFFSET", "0")),
    "require_non_noop": bool(int(os.environ.get("REQUIRE_NON_NOOP", "0"))),
    "no_op_threshold": float(os.environ.get("NO_OP_THRESHOLD", "0.0")),
    "min_non_noop_steps": int(os.environ.get("MIN_NON_NOOP_STEPS", "1")),
    "reward_filter_mode": os.environ.get("REWARD_FILTER_MODE", "none"),
    "reward_signal_threshold": float(os.environ.get("REWARD_SIGNAL_THRESHOLD", "0.0")),
    "min_reward_signal_steps": int(os.environ.get("MIN_REWARD_SIGNAL_STEPS", "1")),
    "require_visual_delta": bool(int(os.environ.get("REQUIRE_VISUAL_DELTA", "0"))),
    "visual_delta_threshold": float(os.environ.get("VISUAL_DELTA_THRESHOLD", "0.0")),
    "min_visual_delta_steps": int(os.environ.get("MIN_VISUAL_DELTA_STEPS", "1")),
    "visual_delta_stride": int(os.environ.get("VISUAL_DELTA_STRIDE", "4")),
    "closed_loop_weight": float(os.environ.get("CLOSED_LOOP_WEIGHT", "0.0")),
    "closed_loop_ctx": int(os.environ.get("CLOSED_LOOP_CTX", "8")),
    "closed_loop_horizon": int(os.environ.get("CLOSED_LOOP_HORIZON", "4")),
    "closed_loop_signal": float(os.environ.get("CLOSED_LOOP_SIGNAL", "0.1")),
    "closed_loop_contrast_weight": float(os.environ.get("CLOSED_LOOP_CONTRAST_WEIGHT", "0.0")),
    "closed_loop_negative_modes": os.environ.get("CLOSED_LOOP_NEGATIVE_MODES", "shuffle,zero,time_shift"),
    "source_default_weight": source_default_weight,
    "source_weights": source_weight_map,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}
manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

declare -A SOURCE_WEIGHT_MAP=()
IFS=',' read -r -a SOURCE_WEIGHT_ITEMS <<< "${SOURCE_WEIGHTS}"
for item in "${SOURCE_WEIGHT_ITEMS[@]}"; do
  item="${item//[[:space:]]/}"
  [[ -n "${item}" ]] || continue
  if [[ "${item}" != *=* ]]; then
    echo "[all-data] invalid SOURCE_WEIGHTS item '${item}', expected name=integer_weight" >&2
    exit 1
  fi
  SOURCE_WEIGHT_MAP["${item%%=*}"]="${item#*=}"
done

source_weight() {
  local name="$1"
  local weight="${SOURCE_WEIGHT_MAP[$name]:-${SOURCE_DEFAULT_WEIGHT}}"
  if ! [[ "${weight}" =~ ^[0-9]+$ ]]; then
    echo "[all-data] invalid source weight for ${name}: ${weight}" >&2
    exit 1
  fi
  printf '%s' "${weight}"
}

RAW_DIRS=()
FRAME_DIRS=()
WEIGHTED_SOURCE_NAMES=()
add_weighted_source() {
  local name="$1"
  local raw="$2"
  local frames="$3"
  local weight
  weight="$(source_weight "${name}")"
  echo "[all-data] source ${name} weight=${weight} raw=${raw} frames=${frames}"
  for ((i=0; i<weight; i++)); do
    RAW_DIRS+=("${raw}")
    FRAME_DIRS+=("${frames}")
    WEIGHTED_SOURCE_NAMES+=("${name}")
  done
}

add_weighted_source "dreamer4_hf_expert" "${DREAMER_RAW}/expert" "${DREAMER_SHARDS}/expert"
add_weighted_source "dreamer4_hf_mixed_small" "${DREAMER_RAW}/mixed-small" "${DREAMER_SHARDS}/mixed-small"
add_weighted_source "dreamer4_hf_mixed_large" "${DREAMER_RAW}/mixed-large" "${DREAMER_SHARDS}/mixed-large"
add_weighted_source "soar_native_v2" "${SOAR_ROOT}/raw" "${SOAR_ROOT}/frames"
add_weighted_source "robonet_sample_64" "${ROBONET_ROOT}/raw" "${ROBONET_ROOT}/frames"
IFS=',' read -r -a HF_DATASET_ARRAY <<< "${HF_ACTION_DATASETS}"
for dataset in "${HF_DATASET_ARRAY[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  [[ -n "${dataset}" ]] || continue
  add_weighted_source "hf_robot_${dataset}" "${HF_ACTION_EXPORT_ROOT}/${dataset}/raw" "${HF_ACTION_EXPORT_ROOT}/${dataset}/frames"
done

if [[ "${#RAW_DIRS[@]}" -eq 0 ]]; then
  echo "[all-data] no weighted sources selected; check SOURCE_DEFAULT_WEIGHT/SOURCE_WEIGHTS" >&2
  exit 1
fi
printf '[all-data] effective_weighted_sources=%s\n' "${WEIGHTED_SOURCE_NAMES[*]}"

TOKENIZER_DIR="${OUT}/tokenizer_ckpts"
TOKENIZER_CKPT="${TOKENIZER_DIR}/latest.pt"
if [[ -f "${TOKENIZER_CKPT}" ]]; then
  TOKENIZER_RESUME=(--resume "${TOKENIZER_CKPT}")
elif [[ -f "${BASE_TOKENIZER_CKPT}" ]]; then
  TOKENIZER_RESUME=(--resume "${BASE_TOKENIZER_CKPT}" --resume_reset_optim)
else
  TOKENIZER_RESUME=()
fi

cd /workspace/dreamer4/dreamer4
if [[ "${SKIP_TOKENIZER}" == "1" ]]; then
  if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
    if [[ -f "${BASE_TOKENIZER_CKPT}" ]]; then
      mkdir -p "${TOKENIZER_DIR}"
      cp "${BASE_TOKENIZER_CKPT}" "${TOKENIZER_CKPT}"
    else
      echo "[tokenizer] SKIP_TOKENIZER=1 but no tokenizer checkpoint exists." >&2
      exit 1
    fi
  fi
  echo "[tokenizer] skipping; using ${TOKENIZER_CKPT}"
else
  echo "[tokenizer] training all-data tokenizer continuation for ${TOKENIZER_STEPS} steps"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_tokenizer.py \
    --data_dirs "${FRAME_DIRS[@]}" \
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
    --wandb_run_name "all-data-tokenizer-${RUN_ID}" \
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
elif [[ -n "${BASE_DYNAMICS_CKPT}" && -f "${BASE_DYNAMICS_CKPT}" ]]; then
  DYNAMICS_RESUME=(--resume "${BASE_DYNAMICS_CKPT}" --resume_reset_optim)
fi

echo "[dynamics] training all-data action-token dynamics for ${DYNAMICS_STEPS} steps"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_dynamics.py \
  --use_actions \
  --data_dirs "${RAW_DIRS[@]}" \
  --frame_dirs "${FRAME_DIRS[@]}" \
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
  --action_frame_offset "${ACTION_FRAME_OFFSET}" \
  $([[ "${REQUIRE_NON_NOOP}" == "1" ]] && printf '%s' "--require_non_noop") \
  --no_op_threshold "${NO_OP_THRESHOLD}" \
  --min_non_noop_steps "${MIN_NON_NOOP_STEPS}" \
  --reward_filter_mode "${REWARD_FILTER_MODE}" \
  --reward_signal_threshold "${REWARD_SIGNAL_THRESHOLD}" \
  --min_reward_signal_steps "${MIN_REWARD_SIGNAL_STEPS}" \
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
  $([[ "${CLOSED_LOOP_BACKPROP_HISTORY}" == "1" ]] && printf '%s' "--closed_loop_backprop_history") \
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
  --max_steps "${DYNAMICS_STEPS}" \
  --grad_accum "${DYNAMICS_GRAD_ACCUM}" \
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
  --wandb_run_name "all-data-dynamics-${RUN_ID}" \
  --wandb_mode "${WANDB_MODE}" \
  --ckpt_dir "${DYNAMICS_DIR}" \
  --save_every 5000 \
  "${DYNAMICS_RESUME[@]}"

cd /workspace
echo "[eval] evaluating all-data action grounding"
EVAL_SOURCE_ARGS=()
for raw_dir in "${RAW_DIRS[@]}"; do
  EVAL_SOURCE_ARGS+=(--data-dir "${raw_dir}")
done
for frame_dir in "${FRAME_DIRS[@]}"; do
  EVAL_SOURCE_ARGS+=(--frames-dir "${frame_dir}")
done
python /workspace/sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py \
  "${EVAL_SOURCE_ARGS[@]}" \
  --tasks-json "${TASKS_JSON}" \
  --tokenizer-ckpt "${TOKENIZER_CKPT}" \
  --dynamics-ckpt "${DYNAMICS_CKPT}" \
  --out "${OUT}/native_dynamics_eval_h8_all_data.json" \
  --seq-len 16 \
  --batch-size 4 \
  --max-batches "${EVAL_MAX_BATCHES}" \
  --rollout-horizon 8 \
  --ctx-len 8 \
  --eval-d 0.25 \
  --action-dim "${ACTION_DIM}" \
  --action-features "${ACTION_FEATURES}" \
  --negative-modes "${EVAL_NEGATIVE_MODES:-shuffle,zero,time_shift}" \
  --action-frame-offset "${EVAL_ACTION_FRAME_OFFSET}" \
  $([[ "${REQUIRE_NON_NOOP}" == "1" ]] && printf '%s' "--require-non-noop") \
  --no-op-threshold "${NO_OP_THRESHOLD}" \
  --min-non-noop-steps "${MIN_NON_NOOP_STEPS}" \
  --reward-filter-mode "${REWARD_FILTER_MODE}" \
  --reward-signal-threshold "${REWARD_SIGNAL_THRESHOLD}" \
  --min-reward-signal-steps "${MIN_REWARD_SIGNAL_STEPS}" \
  $([[ "${REQUIRE_VISUAL_DELTA}" == "1" ]] && printf '%s' "--require-visual-delta") \
  --visual-delta-threshold "${VISUAL_DELTA_THRESHOLD}" \
  --min-visual-delta-steps "${MIN_VISUAL_DELTA_STEPS}" \
  --visual-delta-stride "${VISUAL_DELTA_STRIDE}" \
  --causal-min-ratio "${EVAL_CAUSAL_MIN_RATIO}" \
  --device cuda \
  --seed 29

echo "[all-data] finished $(date -Is)"
