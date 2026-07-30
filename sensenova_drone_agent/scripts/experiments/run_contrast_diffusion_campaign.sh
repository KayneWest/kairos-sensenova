#!/usr/bin/env bash
# Contrast-trained diffusion proposer campaign (paper §6.2 follow-up):
# does adding the GRU recipe's contrast losses to the diffusion objective
# make the generative prior action-causal — and does value guidance then
# have something to steer? plan_grad (soft optimization on the plan sphere
# through the GRU proposer) rides along as the search-layer comparison.
#
# Usage: setsid nohup bash run_contrast_diffusion_campaign.sh > output/contrastdiff_chain.log 2>&1 &
set -Eeuo pipefail
ROOT=/home/mkrzus/kairos-sensenova
SDA="${ROOT}/sensenova_drone_agent"
W=/workspace/sensenova_drone_agent
IMAGE=sensenova_drone_agent-dreamer:local
GPU=device=1
STATUS="${SDA}/output/contrastdiff_chain_status.log"
POLICIES="bc,bc_random,plan_grad,diff_prior,diff_argmax,diff_guided"

st() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

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
  local name="sda-cdiff-$1"; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run --rm --name "$name" --gpus "$GPU" --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/workspace/.docker-home \
    -e PYTHONUNBUFFERED=1 \
    -v "${ROOT}:/workspace" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -w /workspace "$IMAGE" "$@"
}

train_proposer() {  # $1=out name  $2=planner run id  $3=seed
  if [ -f "${SDA}/output/$1/final.pt" ]; then st "SKIP proposer $1 - exists"; return 0; fi
  thermal_gate
  DG "train-$1" python3 "${W}/scripts/train_latent_diffusion_proposer.py" \
    --planner-ckpt "${W}/output/latent_imagination_planner_$2/planner_ckpts/final.pt" \
    --manifest-json "${W}/data/gym_drone_game_dreamer4/drone_v4_dagger_manifest.json" \
    --out-dir "${W}/output/$1" --steps 30000 --seed "$3" \
    --contrast-weight 1.0 --contrast-margin 0.05 \
    > "${SDA}/output/$1_train_log.txt" 2>&1
  grep '"eval"' "${SDA}/output/$1_train_log.txt" | tail -1 | tee -a "$STATUS"
}

run_eval() {  # $1=out dir  $2=proposer planner  $3=judge planner  $4=bc head  $5=diffusion  $6=lambda  $7=episodes  $8=seed  $9=policies
  if [ -f "${SDA}/output/$1/summary.json" ]; then st "SKIP eval $1 - exists"; return 0; fi
  thermal_gate
  DG "eval-$1" python3 "${W}/scripts/eval_gym_drone_game_diffusion_think.py" \
    --planner-ckpt "${W}/output/latent_imagination_planner_$2/planner_ckpts/final.pt" \
    --judge-ckpt "${W}/output/latent_imagination_planner_$3/planner_ckpts/final.pt" \
    --bc-head "${W}/output/$4" \
    --diffusion-ckpt "${W}/output/$5/final.pt" \
    --out-dir "${W}/output/$1" \
    --episodes "$7" --guidance-scale "$6" --seed "$8" --policies "$9"
  python3 -c "import json; d=json.load(open('${SDA}/output/$1/summary.json')); print(json.dumps({k:v for k,v in d['gates'].items() if k.endswith('_wins')}))" | tee -a "$STATUS"
}

pick_lambda() {
  python3 - <<'EOF'
import json
best = None
for lam, tag in [(0.25, "l025"), (1.0, "l1"), (4.0, "l4")]:
    try:
        d = json.load(open(f"/home/mkrzus/kairos-sensenova/sensenova_drone_agent/output/diffusion_lambda_sweep_v3_{tag}/summary.json"))
        pp = d["per_policy"]["diff_guided"]
        key = (pp["success_rate"], pp["mean_return"])
        if best is None or key > best[0]:
            best = (key, lam)
    except FileNotFoundError:
        pass
print(best[1])
EOF
}

st "=== CONTRAST-DIFFUSION CAMPAIGN START (GPU 1) ==="

st "[1/6] train contrast proposer seed1"
train_proposer latent_diffusion_proposer_v3_contrast drone_game_v8_dagger_c1 20260710

st "[2/6] lambda sweep (n=200, good judge)"
for pair in "0.25 l025" "1.0 l1" "4.0 l4"; do
  set -- $pair
  run_eval "diffusion_lambda_sweep_v3_$2" drone_game_v8_dagger_c1 drone_game_v8_dagger_c1 \
    drone_bc_chunk_head_dagger_c1.pt latent_diffusion_proposer_v3_contrast "$1" 200 20260710 "diff_guided"
done
LAM=$(pick_lambda)
st "lambda picked: ${LAM}"

st "[3/6] full evals seed1 (good + inverted judge, n=1000, lambda=${LAM})"
run_eval closed_loop_drone_game_v18_contrastdiff_goodjudge drone_game_v8_dagger_c1 drone_game_v8_dagger_c1 \
  drone_bc_chunk_head_dagger_c1.pt latent_diffusion_proposer_v3_contrast "$LAM" 1000 20260710 "$POLICIES"
run_eval closed_loop_drone_game_v18_contrastdiff_badjudge drone_game_v8_dagger_c1 drone_game_v11_dagger_c2_only \
  drone_bc_chunk_head_dagger_c1.pt latent_diffusion_proposer_v3_contrast "$LAM" 1000 20260710 "$POLICIES"

st "[4/6] train contrast proposer seed2"
train_proposer latent_diffusion_proposer_v3_contrast_seed2 drone_game_v8_dagger_c1_seed2 20260711

st "[5/6] full evals seed2 (eval seed 20270300, lambda=${LAM})"
run_eval closed_loop_drone_game_v18_contrastdiff_goodjudge_seed2 drone_game_v8_dagger_c1_seed2 drone_game_v8_dagger_c1_seed2 \
  drone_bc_chunk_head_dagger_c1_seed2.pt latent_diffusion_proposer_v3_contrast_seed2 "$LAM" 1000 20270300 "$POLICIES"
run_eval closed_loop_drone_game_v18_contrastdiff_badjudge_seed2 drone_game_v8_dagger_c1_seed2 drone_game_v11_dagger_c2_only_seed2 \
  drone_bc_chunk_head_dagger_c1_seed2.pt latent_diffusion_proposer_v3_contrast_seed2 "$LAM" 1000 20270300 "$POLICIES"

st "[6/6] === CONTRAST-DIFFUSION CAMPAIGN COMPLETE ==="
