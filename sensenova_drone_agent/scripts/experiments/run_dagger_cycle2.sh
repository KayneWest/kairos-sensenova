#!/usr/bin/env bash
# DAgger cycle 2, both seeds, chained on one GPU. Mirrors the cycle-1 chain
# (docs/WORKLOG.md "TRACK 2: DAGGER CYCLE 1") with c1 -> c2 bumped:
#   collect 400 episodes with the cycle-1 winner agent -> RTG relabel ->
#   manifest base(w2)+dagger_c1(w1)+dagger_c2(w1) -> planner 60k -> BC head
#   15k -> n=1000 closed-loop eval -> repeat with fresh training seeds.
# Eval seeds match cycle 1 (seed1 default 20260710, seed2 20270300) so
# cycle-over-cycle comparisons are on matched env instances.
#
# Usage: setsid nohup bash run_dagger_cycle2.sh > output/dagger_c2_chain.log 2>&1 &
set -Eeuo pipefail
ROOT=/home/mkrzus/kairos-sensenova
SDA="${ROOT}/sensenova_drone_agent"
W=/workspace/sensenova_drone_agent
IMAGE=sensenova_drone_agent-dreamer:local
GPU=device=1
STATUS="${SDA}/output/dagger_c2_chain_status.log"

st() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

# Foreground docker run; container names start with sda- so the thermal
# guard can pause them. First arg = short step name, rest = command.
DG() {
  local name="sda-dagger-c2-$1"; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run --rm --name "$name" --gpus "$GPU" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/workspace/.docker-home \
    -e PYTHONUNBUFFERED=1 \
    -v "${ROOT}:/workspace" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -w /workspace "$IMAGE" "$@"
}
DC() {  # CPU-only variant
  local name="sda-dagger-c2-$1"; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run --rm --name "$name" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/workspace/.docker-home \
    -e PYTHONUNBUFFERED=1 \
    -v "${ROOT}:/workspace" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -w /workspace "$IMAGE" "$@"
}

train_planner() {  # $1=RUN_ID  $2=SEED
  if [ -f "${SDA}/output/latent_imagination_planner_$1/planner_ckpts/final.pt" ]; then
    st "SKIP planner $1 - final.pt exists"; return 0
  fi
  RUN_ID="$1" \
  MANIFEST_JSON="${W}/data/gym_drone_game_dreamer4/drone_v5_dagger2_manifest.json" \
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
    python3 -c "import json; d=json.load(open('${SDA}/output/${out}/summary.json')); print(json.dumps(d['gates']))" | tee -a "$STATUS"
    return 0
  fi
  DG "eval-${out##*game_}" python3 "${W}/scripts/eval_gym_drone_game_act_by_imagination.py" \
    --ckpt "${W}/output/latent_imagination_planner_${run_id}/planner_ckpts/final.pt" \
    --out-dir "${W}/output/${out}" \
    --episodes 1000 --num-candidates 32 --replan-every 4 --score-plan-mode zero \
    --bc-head "${W}/output/${head}" --bc-temperature 0.8 \
    --policies act_bc,act_bc_think,act_bc_random "$@"
  python3 -c "import json; d=json.load(open('${SDA}/output/${out}/summary.json')); print(json.dumps(d['gates']))" \
    | tee -a "$STATUS"
}

st "=== DAGGER CYCLE 2 CHAIN START (GPU 1) ==="

# Steps 1-3 are skipped when their artifacts already exist, so the chain can
# be relaunched after a crash without redoing (and changing) the collection.
if [ -f "${SDA}/data/gym_drone_game_dreamer4/dagger_c2/summary.json" ]; then
  st "[1/9] SKIP collect - dagger_c2/summary.json exists"
else
  st "[1/9] collect 400 episodes with cycle-1 agent -> dagger_c2"
  DG collect python3 "${W}/scripts/collect_dagger_episodes.py" \
    --ckpt "${W}/output/latent_imagination_planner_drone_game_v8_dagger_c1/planner_ckpts/final.pt" \
    --bc-head "${W}/output/drone_bc_chunk_head_dagger_c1.pt" \
    --out "${W}/data/gym_drone_game_dreamer4/dagger_c2" \
    --episodes 400 --seed 20270200 --overwrite \
    > "${SDA}/output/dagger_c2_collect_log.txt" 2>&1
  tail -1 "${SDA}/output/dagger_c2_collect_log.txt" | tee -a "$STATUS"
fi

if [ -f "${SDA}/data/gym_drone_game_dreamer4/dagger_c2_rtg/rtg_summary.json" ]; then
  st "[2/9] SKIP relabel - dagger_c2_rtg exists"
else
  st "[2/9] RTG relabel -> dagger_c2_rtg"
  DC relabel python3 "${W}/scripts/relabel_rewards_return_to_go.py" \
    --src "${W}/data/gym_drone_game_dreamer4/dagger_c2" \
    --out "${W}/data/gym_drone_game_dreamer4/dagger_c2_rtg" --overwrite
fi

st "[3/9] write drone_v5_dagger2_manifest.json (base w2 + c1 w1 + c2 w1)"
cat > "${SDA}/data/gym_drone_game_dreamer4/drone_v5_dagger2_manifest.json" <<EOF
{
  "name": "gym_drone_game_dreamer4_v5_dagger2",
  "action_dim": 9,
  "action_features": "current",
  "action_frame_offset": -1,
  "tasks_json": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/tasks.json",
  "sources": [
    {"name": "gym_drone_game", "raw": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/raw", "frames": "${W}/data/gym_drone_game_dreamer4/drone_v2_pad_rtg/frames", "weight": 2},
    {"name": "gym_drone_game_dagger", "raw": "${W}/data/gym_drone_game_dreamer4/dagger_c1_rtg/raw", "frames": "${W}/data/gym_drone_game_dreamer4/dagger_c1_rtg/frames", "weight": 1},
    {"name": "gym_drone_game_dagger_c2", "raw": "${W}/data/gym_drone_game_dreamer4/dagger_c2_rtg/raw", "frames": "${W}/data/gym_drone_game_dreamer4/dagger_c2_rtg/frames", "weight": 1}
  ]
}
EOF

st "[4/9] planner seed1 (drone_game_v9_dagger_c2, SEED=20260710, 60k steps)"
train_planner drone_game_v9_dagger_c2 20260710

st "[5/9] BC head seed1 -> drone_bc_chunk_head_dagger_c2.pt"
bc_head drone_game_v9_dagger_c2 drone_bc_chunk_head_dagger_c2.pt \
  > "${SDA}/output/drone_bc_chunk_head_dagger_c2_log.txt" 2>&1

st "[6/9] closed-loop eval seed1 n=1000 -> closed_loop_drone_game_v13_dagger_c2"
closed_loop drone_game_v9_dagger_c2 drone_bc_chunk_head_dagger_c2.pt \
  closed_loop_drone_game_v13_dagger_c2

st "[7/9] planner seed2 (drone_game_v9_dagger_c2_seed2, SEED=20260711)"
train_planner drone_game_v9_dagger_c2_seed2 20260711

st "[8/9] BC head seed2 -> drone_bc_chunk_head_dagger_c2_seed2.pt"
bc_head drone_game_v9_dagger_c2_seed2 drone_bc_chunk_head_dagger_c2_seed2.pt --seed 20260712 \
  > "${SDA}/output/drone_bc_chunk_head_dagger_c2_seed2_log.txt" 2>&1

st "[9/9] closed-loop eval seed2 -> closed_loop_drone_game_v13_dagger_c2_seed2"
closed_loop drone_game_v9_dagger_c2_seed2 drone_bc_chunk_head_dagger_c2_seed2.pt \
  closed_loop_drone_game_v13_dagger_c2_seed2 --seed 20270300

st "=== DAGGER CYCLE 2 CHAIN COMPLETE ==="
