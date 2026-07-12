# Dreamer4-Lite Drone Game

## Goal

Implement the practical subset of the Dreamer 4 recipe that fits this repo:

```text
frozen Sensenova world-model encoder/dynamics
  -> task-conditioned BC policy prior
  -> learned reward/value/risk heads
  -> short imagined rollouts
  -> KL-constrained policy/value update
```

This is not a reproduction of Dreamer 4. It is a small offline-imagination scaffold
for testing whether our world-model features can support policy improvement.

## Implementation

```text
sensenova_drone_agent/scripts/train_gym_drone_game_dreamer4_lite.py
```

The script uses:

```text
world model:
  sensenova_drone_agent/output/gym_drone_game_world_model_v1/best.pt

BC actions:
  sensenova_drone_agent/data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl

branch reward/risk labels:
  sensenova_drone_agent/data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl
```

## Run

Use Docker; host Python is not expected to have PyTorch installed.

```bash
cd /home/mkrzus/kairos-sensenova

docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  sensenova_drone_agent-dreamer:local \
  python sensenova_drone_agent/scripts/train_gym_drone_game_dreamer4_lite.py \
    --out-dir sensenova_drone_agent/output/dreamer4_lite_v2_conservative \
    --supervised-epochs 3 \
    --imagination-updates 80 \
    --imagination-learning-rate 5e-5 \
    --kl-to-prior-weight 2.0 \
    --entropy-weight 0.02 \
    --batch-size 128 \
    --eval-episodes 128 \
    --device cpu
```

Open:

```text
sensenova_drone_agent/output/dreamer4_lite_v2_conservative/index.html
```

## Results

Smoke run:

```text
output:
  sensenova_drone_agent/output/dreamer4_lite_smoke/

result:
  pipeline works end-to-end
```

First unconstrained run:

```text
output:
  sensenova_drone_agent/output/dreamer4_lite_v1/

supervised:
  success_rate: 0.359375
  collision_rate: 0.4375
  mean_return: 9.6375

after imagination:
  success_rate: 0.0
  collision_rate: 0.0
  mean_return: -11.4329

conclusion:
  failed; imagination exploited the learned reward model and collapsed to strafing.
```

Conservative KL-gated run:

```text
output:
  sensenova_drone_agent/output/dreamer4_lite_v2_conservative/

supervised:
  success_rate: 0.359375
  collision_rate: 0.4375
  mean_return: 9.6375
  deployment_score: 22.5281

after imagination:
  success_rate: 0.3828125
  collision_rate: 0.421875
  mean_return: 10.3060
  deployment_score: 26.3997

recommended checkpoint:
  selected_checkpoint.pt
```

## Interpretation

The useful result is narrow:

```text
- The Dreamer4-lite scaffold works.
- Reward-model exploitation is real if KL is too weak.
- Strong KL to the BC prior makes imagination updates safe enough to improve slightly.
- The result still does not beat the best world-model DQN controller.
```

Best known DQN baseline remains materially stronger:

```text
world_model_dqn_v4_shield_in_loop_10:
  success_rate: about 0.67
  collision_rate: about 0.23
```

## Next Fixes

```text
1. Train reward/risk heads on more diverse branch data.
2. Add a hard action-diversity or no-collapse penalty during imagination.
3. Use held-out imagined-policy evaluation as a gate, not imagined reward alone.
4. Add a shield-in-loop evaluation mode for Dreamer4-lite.
5. Compare selected_checkpoint.pt against DQN/BC in the existing dashboard.
```
