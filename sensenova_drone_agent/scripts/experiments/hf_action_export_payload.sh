#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

FRAME_SIZE="${FRAME_SIZE:-128}"
FRAME_STRIDE="${FRAME_STRIDE:-2}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
REWARD_MODE="${REWARD_MODE:-zero}"
TASK_MODE="${TASK_MODE:-fixed}"

run_export() {
  local src="$1"
  local out="$2"
  local name="$3"
  echo "[export] start ${name} $(date -Is)"
  python sensenova_drone_agent/scripts/export_lerobot_hf_dreamer4_dataset.py \
    --input "${src}" \
    --out "${out}" \
    --dataset-name "${name}" \
    --max-trajectories "${MAX_TRAJECTORIES}" \
    --paired-video-parquets-only \
    --frame-stride "${FRAME_STRIDE}" \
    --frame-size "${FRAME_SIZE}" \
    --shard-size "${SHARD_SIZE}" \
    --task-mode "${TASK_MODE}" \
    --reward-mode "${REWARD_MODE}" \
    --force
  echo "[export] done ${name} $(date -Is)"
}

run_export \
  sensenova_drone_agent/data/robotics/hf_action_sources/IPEC_COMMUNITY_droid_lerobot \
  sensenova_drone_agent/data/robotics/hf_action_exports/droid_lerobot_dreamer4 \
  droid_lerobot

run_export \
  sensenova_drone_agent/data/robotics/hf_action_sources/IPEC_COMMUNITY_fractal20220817_data_lerobot \
  sensenova_drone_agent/data/robotics/hf_action_exports/fractal20220817_data_lerobot_dreamer4 \
  fractal20220817_data_lerobot

run_export \
  sensenova_drone_agent/data/robotics/hf_action_sources/IPEC_COMMUNITY_bridge_orig_lerobot \
  sensenova_drone_agent/data/robotics/hf_action_exports/bridge_orig_lerobot_dreamer4 \
  bridge_orig_lerobot

python sensenova_drone_agent/scripts/collect_action_world_model_data.py \
  --sources dreamer4-hf,soar,robonet,hf-robot

echo "[export] all done $(date -Is)"
