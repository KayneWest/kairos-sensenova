# Gym Drone Game

This adds a fast Gymnasium-style drone navigation game for decision-rich training.

The environment is not PX4/Gazebo. It is a lightweight first-person simulator with:

- RGB observation rendered from a drone-forward camera.
- Compact state features for fast RL.
- Circular tree obstacles.
- Waypoint goal.
- Collision, success, timeout, and oscillation signals.
- The same discrete action vocabulary used by the SITL agent.

The purpose is to generate and test navigation policies quickly before evaluating them in Gazebo.

## Environment

Python API:

```python
from sensenova_drone.gym_drone_game import DroneMazeEnv

env = DroneMazeEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(5)
```

Observation:

```text
obs["image"]  uint8 RGB, H x W x 3
obs["state"]  float32 vector with goal, clearance, step, and action features
```

Actions use `sensenova_drone.bc_data.ACTION_VOCAB`:

```text
0 hover
1 yaw_left
2 yaw_right
3 ascend
4 descend
5 forward
6 backward
7 strafe_left
8 strafe_right
```

Reward terms:

```text
+ progress toward waypoint
+ front-clearance improvement
+ success
- time penalty
- near-obstacle penalty
- oscillation penalty
- collision
- out of bounds
```

## Train

Short smoke run:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
PYTHONPATH=src python3 scripts/train_gym_drone_game.py \
  --total-steps 500 \
  --eval-every 250 \
  --eval-episodes 4 \
  --out-dir output/gym_drone_game_dqn_smoke
```

The trainer keeps the full repo action vocabulary but masks the 2D game to navigation actions by default:

```text
hover,yaw_left,yaw_right,forward,strafe_left,strafe_right
```

Override with:

```bash
--enabled-actions hover,yaw_left,yaw_right,forward,strafe_left,strafe_right,backward
```

Overnight-style run:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
PYTHONPATH=src python3 scripts/train_gym_drone_game.py \
  --total-steps 100000 \
  --eval-every 2500 \
  --eval-episodes 32 \
  --out-dir output/gym_drone_game_dqn_overnight
```

## Direct Visual RL

The state-DQN trainer above learns from compact privileged state. To test visual grounding directly, use the image-DQN trainer:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
PYTHONPATH=src python3 scripts/train_gym_drone_game_image_dqn.py \
  --total-steps 50000 \
  --eval-every 2500 \
  --eval-episodes 24 \
  --out-dir output/gym_drone_game_image_dqn_v1
```

This policy learns from stacked RGB frames:

```text
RGB frame stack -> CNN Q-network -> drone action
```

It does not receive the compact `obs["state"]` vector.

## Risk-Aware Visual Policy

The risk-aware policy trains from RGB frames, action labels, and privileged simulator labels for safety/progress:

```text
RGB frame -> action
RGB frame -> collision risk
RGB frame -> stall risk
RGB frame -> front clearance
RGB frame -> progress
```

Train from an exported Gym manifest:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_risk_policy.py \
  --manifest data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl \
  --out-dir output/gym_drone_game_risk_policy_v1 \
  --epochs 6 \
  --batch-size 64 \
  --image-size 96 \
  --no-class-weights
```

Evaluate with the learned safety shield:

```bash
PYTHONPATH=src:. python3 scripts/eval_gym_drone_game_risk_policy.py \
  --checkpoint output/gym_drone_game_risk_policy_v1/best.pt \
  --out-dir output/gym_drone_game_risk_policy_eval_v1_128_fixed_shield_35 \
  --episodes 128 \
  --seed 900000 \
  --shield-front-clearance-m 3.5 \
  --shield-collision-threshold 0.75
```

The shield is intentionally explicit:

```text
if predicted forward action is risky:
    block forward
    choose the best safe alternative from strafe/yaw/hover
```

This lets us measure safety/progress tradeoffs instead of only measuring action imitation accuracy.

## Matched-Seed Benchmark

Compare all current model types on identical seeds:

```bash
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --out-dir output/gym_drone_game_model_benchmark_v2_risk \
  --episodes 128 \
  --seed 900000 \
  --device cpu \
  --image-width 64 \
  --image-height 48 \
  --models random,heuristic,state_dqn,image_bc,world_model_policy,risk_visual_policy \
  --risk-shield-front-clearance-m 3.5 \
  --risk-shield-collision-threshold 0.75
```

## Action-Conditioned Risk Scorer

The action-conditioned scorer creates labels for every candidate action from the same camera frame:

```text
I_t + candidate action -> predicted collision/progress/clearance/utility
```

Export branch-labeled data:

```bash
PYTHONPATH=src:. python3 scripts/export_gym_drone_game_action_risk_dataset.py \
  --policy mixed \
  --checkpoint output/gym_drone_game_dqn_overnight_20260509T032655Z/best.pt \
  --episodes 192 \
  --seed 620000 \
  --out-root data/gym_drone_game_action_risk_v2/frames \
  --manifest data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl \
  --summary-json data/gym_drone_game_action_risk_v2/summary.json \
  --device cpu
```

Train:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_action_risk_scorer.py \
  --manifest data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl \
  --out-dir output/gym_drone_game_action_risk_scorer_v2 \
  --epochs 8 \
  --batch-size 128 \
  --image-size 96 \
  --device cpu
```

Evaluate:

```bash
PYTHONPATH=src:. python3 scripts/eval_gym_drone_game_action_risk_planner.py \
  --checkpoint output/gym_drone_game_action_risk_scorer_v2/best.pt \
  --out-dir output/gym_drone_game_action_risk_planner_eval_v2_128 \
  --episodes 128 \
  --seed 900000 \
  --device cpu
```

Current behavior:

```text
The action-risk planner is collision-averse but stalls with repeated yaw/avoidance.
This shows one-step risk labels are not enough; the next version needs multi-step recovery labels.
```

## World-Model Decision Heads

This is the mid-training path that uses the world model as a pretrained visual backbone:

```text
RGB frame
  -> frozen action-conditioned world-model encoder
  -> policy head trained from DQN decisions
  -> risk/utility heads trained from branch labels
  -> candidate-action planner
```

Train:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_world_model_decision_heads.py \
  --world-model-checkpoint output/gym_drone_game_world_model_v1/best.pt \
  --bc-manifest data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl \
  --risk-manifest data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl \
  --out-dir output/gym_drone_game_world_model_decision_heads_v2_weighted \
  --epochs 6 \
  --batch-size 128 \
  --device cpu
```

Evaluate:

```bash
PYTHONPATH=src:. python3 scripts/eval_gym_drone_game_world_model_decision_heads.py \
  --checkpoint output/gym_drone_game_world_model_decision_heads_v2_weighted/best.pt \
  --out-dir output/gym_drone_game_world_model_decision_heads_eval_v2_weighted_128 \
  --episodes 128 \
  --seed 900000 \
  --device cpu
```

Current behavior:

```text
The heads learn offline policy/risk signals, but the closed-loop planner remains too cautious.
The result supports using world-model features, but not yet one-step risk planning as the final controller.
```

## World-Model Feature DQN

This is the first true RL path using the pretrained world model as representation:

```text
RGB frame
  -> frozen action-conditioned world-model encoder
  -> z_t + goal features
  -> DQN Q-values
  -> action
  -> environment reward
```

Train a short run in the Docker tools environment:

```bash
cd /home/mkrzus/kairos-sensenova
docker run --rm -v "$PWD":/workspace -w /workspace/sensenova_drone_agent \
  sensenova_drone_agent-tools:local \
  bash -lc 'PYTHONPATH=src:. /opt/drone-sim-venv/bin/python3 \
    scripts/train_gym_drone_game_world_model_dqn.py \
    --world-model-checkpoint output/gym_drone_game_world_model_v1/best.pt \
    --total-steps 12000 \
    --eval-every 2000 \
    --eval-episodes 16 \
    --warmup-steps 500 \
    --batch-size 128 \
    --out-dir output/gym_drone_game_world_model_dqn_v1 \
    --seed 23 \
    --expert-mix-start 0.10 \
    --expert-mix-end 0.0 \
    --epsilon-decay-steps 10000'
```

Benchmark it on matched seeds:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace/sensenova_drone_agent \
  sensenova_drone_agent-tools:local \
  bash -lc 'PYTHONPATH=src:. /opt/drone-sim-venv/bin/python3 \
    scripts/benchmark_gym_drone_game_models.py \
    --models random,heuristic,state_dqn,image_bc,world_model_dqn \
    --world-model-dqn-checkpoint output/gym_drone_game_world_model_dqn_v1/best.pt \
    --out-dir output/gym_drone_game_model_benchmark_v5_world_model_dqn \
    --episodes 64 \
    --seed 930000'
```

Current behavior:

```text
The world-model feature DQN learns from rewards and beats random/heuristic on success.
It slightly beats image BC on success in the current 64-seed benchmark.
It still collides too often, so the next improvement is constrained RL or risk-aware action filtering.
```

Safety-shaped training is available through reward-shaping flags:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_world_model_dqn.py \
  --world-model-checkpoint output/gym_drone_game_world_model_v1/best.pt \
  --out-dir output/gym_drone_game_world_model_dqn_v2_safety_shaped \
  --total-steps 12000 \
  --extra-collision-penalty 8.0 \
  --extra-out-of-bounds-penalty 5.0 \
  --near-obstacle-threshold-m 1.4 \
  --near-obstacle-penalty 1.5 \
  --forward-low-clearance-threshold-m 2.2 \
  --forward-low-clearance-penalty 1.0 \
  --clearance-recovery-bonus 0.2
```

The currently better safety path is a runtime clearance shield on top of the unshaped DQN:

```bash
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --models heuristic,image_bc,world_model_dqn \
  --world-model-dqn-checkpoint output/gym_drone_game_world_model_dqn_v1/best.pt \
  --world-model-dqn-shield-front-clearance-m 1.0 \
  --out-dir output/gym_drone_game_model_benchmark_v10_world_model_dqn_runtime_shield_10 \
  --episodes 64 \
  --seed 930000
```

Current recommendation:

```text
Use shield threshold 1.0m when optimizing success with moderate collision reduction.
Use shield threshold 1.2m when accepting slightly lower success for lower collision.
Avoid 1.5m for now; it is too conservative and causes timeouts.
```

The next version trains with that same shield inside the RL loop. In this mode, exploration,
greedy action selection, evaluation, and DQN target bootstrapping all mask `FORWARD` when
front clearance is below the threshold:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_world_model_dqn.py \
  --world-model-checkpoint output/gym_drone_game_world_model_v1/best.pt \
  --out-dir output/gym_drone_game_world_model_dqn_v4_shield_in_loop_10 \
  --total-steps 12000 \
  --shield-front-clearance-m 1.0 \
  --expert-mix-start 0.10 \
  --expert-mix-end 0.0 \
  --epsilon-decay-steps 10000
```

Benchmark:

```bash
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --models heuristic,image_bc,world_model_dqn \
  --world-model-dqn-checkpoint output/gym_drone_game_world_model_dqn_v4_shield_in_loop_10/best.pt \
  --world-model-dqn-shield-front-clearance-m 1.0 \
  --out-dir output/gym_drone_game_model_benchmark_v11_world_model_dqn_shield_in_loop_10 \
  --episodes 64 \
  --seed 930000
```

Current best practical controller:

```text
world_model_dqn_v4_shield_in_loop_10 + runtime shield 1.0m
```

To test whether the pretrained encoder matters, run the same shield-in-loop trainer
with a frozen random encoder:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_world_model_dqn.py \
  --encoder-source random \
  --world-model-checkpoint output/gym_drone_game_world_model_v1/best.pt \
  --out-dir output/gym_drone_game_random_encoder_dqn_v1_shield_in_loop_10 \
  --total-steps 12000 \
  --shield-front-clearance-m 1.0 \
  --expert-mix-start 0.10 \
  --expert-mix-end 0.0 \
  --epsilon-decay-steps 10000
```

Benchmark the random-encoder ablation:

```bash
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --models heuristic,image_bc,world_model_dqn \
  --world-model-dqn-checkpoint output/gym_drone_game_random_encoder_dqn_v1_shield_in_loop_10/best.pt \
  --world-model-dqn-shield-front-clearance-m 1.0 \
  --out-dir output/gym_drone_game_model_benchmark_v13_random_encoder_dqn_shield_in_loop_10 \
  --episodes 64 \
  --seed 930000
```

## CNN DQN Baseline

For a stronger generic visual RL comparison, train a CNN DQN from pixels and goal
features with the same shield-in-loop setup:

```bash
PYTHONPATH=src:. python3 scripts/train_gym_drone_game_cnn_dqn.py \
  --out-dir output/gym_drone_game_cnn_dqn_v1_shield_in_loop_10 \
  --total-steps 12000 \
  --shield-front-clearance-m 1.0 \
  --expert-mix-start 0.10 \
  --expert-mix-end 0.0 \
  --epsilon-decay-steps 10000 \
  --random-shift-pixels 4
```

The first comparison run was stopped at 6000 steps because end-to-end CNN training
is much slower than the frozen world-model encoder path. It still produced a useful
early-training baseline:

```bash
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --models heuristic,image_bc,cnn_dqn,world_model_dqn \
  --cnn-dqn-checkpoint output/gym_drone_game_cnn_dqn_v1_shield_in_loop_10/best.pt \
  --cnn-dqn-shield-front-clearance-m 1.0 \
  --world-model-dqn-checkpoint output/gym_drone_game_world_model_dqn_v4_shield_in_loop_10/best.pt \
  --world-model-dqn-shield-front-clearance-m 1.0 \
  --out-dir output/gym_drone_game_model_benchmark_v14_cnn_vs_world_model_dqn \
  --episodes 64 \
  --seed 930000
```

This baseline is useful for sample-efficiency claims:

```text
The generic CNN can learn from reward, but it needs more training to catch up.
The pretrained world-model encoder gives better early closed-loop performance.
```

Watch:

```text
output/gym_drone_game_dqn_overnight/index.html
output/gym_drone_game_dqn_overnight/metrics.jsonl
output/gym_drone_game_dqn_overnight/latest_eval_trace.png
output/gym_drone_game_world_model_dqn_v1/index.html
output/gym_drone_game_world_model_dqn_v1/metrics.jsonl
output/gym_drone_game_model_benchmark_v5_world_model_dqn/index.html
```

## Interpretation

This is a bridge environment.

It helps answer whether a policy can learn:

```text
real observation -> action -> new real observation -> next action
```

before paying the cost of PX4/Gazebo collection. A good policy here is not proof of Gazebo success, but a bad policy here tells us the reward/action/data setup is still wrong.
