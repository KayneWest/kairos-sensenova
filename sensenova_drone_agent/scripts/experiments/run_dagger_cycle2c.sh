#!/usr/bin/env bash
# DAgger cycle 2b: the dilution-hypothesis test. Cycle 2 (drone_v5 manifest,
# DAgger fraction 1/2, ~94% collisions) regressed CI-clean vs cycle 1's
# two-seed win. This retrains on the SAME data with the DAgger fraction
# restored to cycle-1's 1/3: base w4 + dagger_c1_rtg w1 + dagger_c2_rtg w1.
# No new collection. If dilution caused the regression, think returns to
# ~6%; if not, the cycle-1 win is fragile to retraining.
# Both seeds chained, GPU 1. Between-stage thermal gates (optimal-z
# pattern) let the box settle before each heavy stage.
#
# Usage: setsid nohup bash run_dagger_cycle2c.sh > output/dagger_c2c_chain.log 2>&1 &
set -Eeuo pipefail
ROOT=/home/mkrzus/kairos-sensenova
SDA="${ROOT}/sensenova_drone_agent"
W=/workspace/sensenova_drone_agent
IMAGE=sensenova_drone_agent-dreamer:local
GPU=device=1
STATUS="${SDA}/output/dagger_c2c_chain_status.log"

st() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

# Wait (up to 15 min) for the box to settle before starting a heavy stage:
# max temp <= 75C and total power <= 600W. The systemd guard still covers
# everything continuously; this just gives natural cooldown breaks.
thermal_gate() {
  local i
  for i in $(seq 1 60); do
    read -r t w < <(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits \
      | awk -F', *' 'BEGIN{t=0;s=0} {if($1>t)t=$1; s+=$2} END{print int(t), int(s)}')
    [ "${t:-99}" -le 75 ] && [ "${w:-9999}" -le 600 ] && return 0
    [ "$i" = 1 ] && st "thermal_gate: waiting (temp=${t}C sum=${w}W)"
    sleep 15
  done
  st "thermal_gate: proceeding after 15min wait (temp=${t}C sum=${w}W)"
}

DG() {
  local name="sda-dagger-c2c-$1"; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run --rm --name "$name" --gpus "$GPU" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/workspace/.docker-home \
    -e PYTHONUNBUFFERED=1 \
    -v "${ROOT}:/workspace" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -w /workspace "$IMAGE" "$@"
}

train_planner() {  # $1=RUN_ID  $2=SEED
  if [ -f "${SDA}/output/latent_imagination_planner_$1/planner_ckpts/final.pt" ]; then
    st "SKIP planner $1 - final.pt exists"; return 0
  fi
  thermal_gate
  RUN_ID="$1" \
  MANIFEST_JSON="${W}/data/gym_drone_game_dreamer4/drone_v7_dagger2_only_manifest.json" \
  TOKENIZER_CKPT="${W}/output/drone_game_tokenizer_v1/latest.pt" \
  ACTION_DIM=9 RAW_ACTION_DIM=9 ACTION_FEATURES=current \
  MAX_STEPS=60000 SAVE_EVERY=2000 EVAL_EVERY=2000 TRACE_EVERY=0 \
  GAMMA=0.0 REWARD_LOSS_WEIGHT=0.5 SCORE_PLAN_DROPOUT=0.5 \
  RANK_LOSS_WEIGHT=1.0 CONTRAST_RELATIVE_MARGIN=0.0 PLAN_UNIT_NORM=1 PLAN_STEP_CONDITIONING=1 \
  INVERSE_PLAN_DROPOUT=0.5 INVERSE_IMAGINED_WEIGHT=0.25 INVERSE_CROSS_WEIGHT=0.5 \
  SEED="$2" REQUIRE_NON_NOOP=0 REQUIRE_VISUAL_DELTA=0 GPU_SELECTOR="$GPU" \
  bash "${SDA}/scripts/experiments/launch_latent_imagination_planner.sh"
  local rc
  rc=$(docker wait "sda-latent-imagination-planner-$1")
  [ "$rc" = "0" ] || { st "ABORT planner $1 exited rc=$rc"; exit 1; }
  [ -f "${SDA}/output/latent_imagination_planner_$1/planner_ckpts/final.pt" ] \
    || { st "ABORT planner $1 final.pt missing"; exit 1; }
}

bc_head() {  # $1=planner RUN_ID  $2=out .pt  $3=extra args...
  local run_id="$1" out="$2"; shift 2
  if [ -f "${SDA}/output/${out}" ]; then st "SKIP bc head ${out} - exists"; return 0; fi
  thermal_gate
  DG "bc-${out##*head_}" python3 "${W}/scripts/train_drone_bc_chunk_head.py" \
    --planner-ckpt "${W}/output/latent_imagination_planner_${run_id}/planner_ckpts/final.pt" \
    --data-dir "${W}/data/gym_drone_game_dreamer4/drone_bc_v2/raw" \
    --frames-dir "${W}/data/gym_drone_game_dreamer4/drone_bc_v2/frames" \
    --tasks-json "${W}/data/gym_drone_game_dreamer4/drone_bc_v2/tasks.json" \
    --out "${W}/output/${out}" --steps 15000 --hidden-dim 1024 "$@"
}

closed_loop() {  # $1=planner RUN_ID  $2=bc head .pt  $3=out dir  $4=extra args...
  local run_id="$1" head="$2" out="$3"; shift 3
  if [ -f "${SDA}/output/${out}/summary.json" ]; then
    st "SKIP eval ${out} - summary.json exists"
  else
    thermal_gate
    DG "eval-${out##*game_}" python3 "${W}/scripts/eval_gym_drone_game_act_by_imagination.py" \
      --ckpt "${W}/output/latent_imagination_planner_${run_id}/planner_ckpts/final.pt" \
      --out-dir "${W}/output/${out}" \
      --episodes 1000 --num-candidates 32 --replan-every 4 --score-plan-mode zero \
      --bc-head "${W}/output/${head}" --bc-temperature 0.8 \
      --policies act_bc,act_bc_think,act_bc_random "$@"
  fi
  python3 -c "import json; d=json.load(open('${SDA}/output/${out}/summary.json')); print(json.dumps(d['gates']))" \
    | tee -a "$STATUS"
}

st "=== DAGGER CYCLE 2b (rebalanced mix, dagger fraction 1/3) START (GPU 1) ==="

st "[1/5] write drone_v7_dagger2_only_manifest.json (base w2 + c2 w1, no c1)"
cat > "${SDA}/data/gym_drone_game_dreamer4/drone_v7_dagger2_only_manifest.json" <<EOF
{
  "name": "gym_drone_game_dreamer4_v7_dagger2_only",
  "action_dim": 9,
  "action_features": "current",
  "action_frame_offset": -1,
  "tasks_json": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/tasks.json",
  "sources": [
    {"name": "gym_drone_game", "raw": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/raw", "frames": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/frames", "weight": 2},
    {"name": "gym_drone_game_dagger_c2", "raw": "${W}/data/gym_drone_game_dreamer4/dagger_c2_rtg/raw", "frames": "${W}/data/gym_drone_game_dreamer4/dagger_c2_rtg/frames", "weight": 1}
  ]
}
EOF

st "[2/5] planner seed1 (drone_game_v11_dagger_c2_only, SEED=20260710)"
train_planner drone_game_v11_dagger_c2_only 20260710

st "[3/5] BC head seed1 -> drone_bc_chunk_head_dagger_c2_only.pt"
bc_head drone_game_v11_dagger_c2_only drone_bc_chunk_head_dagger_c2_only.pt \
  > "${SDA}/output/drone_bc_chunk_head_dagger_c2_only_log.txt" 2>&1

st "[4/5] closed-loop eval seed1 n=1000 -> closed_loop_drone_game_v15_dagger_c2_only"
closed_loop drone_game_v11_dagger_c2_only drone_bc_chunk_head_dagger_c2_only.pt \
  closed_loop_drone_game_v15_dagger_c2_only

st "[5/5] seed2: planner + BC + eval -> closed_loop_drone_game_v15_dagger_c2_only_seed2"
train_planner drone_game_v11_dagger_c2_only_seed2 20260711
bc_head drone_game_v11_dagger_c2_only_seed2 drone_bc_chunk_head_dagger_c2_only_seed2.pt --seed 20260712 \
  > "${SDA}/output/drone_bc_chunk_head_dagger_c2_only_seed2_log.txt" 2>&1
closed_loop drone_game_v11_dagger_c2_only_seed2 drone_bc_chunk_head_dagger_c2_only_seed2.pt \
  closed_loop_drone_game_v15_dagger_c2_only_seed2 --seed 20270300

st "=== DAGGER CYCLE 2b CHAIN COMPLETE ==="
