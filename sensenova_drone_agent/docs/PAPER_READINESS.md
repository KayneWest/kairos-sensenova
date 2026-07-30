# Paper Readiness

## Working Title

Grounding World-Model Representations for Closed-Loop Visual Drone Control

## One-Sentence Claim

In a lightweight first-person drone navigation game, a learned action-conditioned visual world-model encoder provides a useful control representation: with the same shielded DQN setup, it outperforms frozen random features and a small-budget CNN trained from scratch, while KL-constrained imagination updates remain promising but not yet competitive.

## Claim Boundary

Current strongest evidence is from the repo's learned drone-game world model:

```text
RGB frame + action -> predicted next RGB frame
frozen world-model encoder + goal features -> shielded DQN policy
```

Do not claim yet that the full Kairos/Sensenova foundation video model directly learns robust control. The Kairos action-conditioning experiment is currently best treated as a negative or motivating result: prompt-only and public camera-control rollouts were not yet strong enough to serve as an MPC teacher.

## What We Can Say Today

- The closed-loop scaffold exists: observation, action, environment response, next observation.
- The lightweight drone game is learnable from state with DQN, so the task itself is not broken.
- Pixel policies can learn partial behavior, but are unstable without better representation and safety.
- The action-conditioned world model learned a measurable transition signal rather than identical next-frame predictions for all actions.
- Frozen world-model features plus shielded DQN are the strongest current visual controller in the toy game.
- Frozen pretrained world-model features beat a frozen random encoder under the same DQN/shield setup.
- Small-budget CNN DQN from scratch underperforms the frozen world-model encoder DQN.
- Dreamer4-lite style imagination is wired and can improve slightly under strong KL, but unconstrained imagination exploits the learned reward model.

## What We Cannot Say Yet

- We cannot claim state-of-the-art control.
- We cannot claim robust PX4/Gazebo autonomy from camera frames.
- We cannot claim Kairos/Sensenova action-conditioned rollouts are currently usable as MPC teacher labels.
- We cannot claim generic CNN baselines cannot catch up with longer training.
- We cannot claim statistical significance from a single training seed per trainable method.

## Current Evidence

Generated table:

```bash
python3 sensenova_drone_agent/scripts/generate_paper_results.py \
  --out-dir sensenova_drone_agent/output/paper_results_v1
```

Open:

```text
sensenova_drone_agent/output/paper_results_v1/paper_results.md
```

Best current controller:

```text
output/gym_drone_game_world_model_dqn_v4_shield_in_loop_10
```

Matched benchmark:

```text
output/gym_drone_game_model_benchmark_v14_cnn_vs_world_model_dqn
```

Key result:

```text
Frozen world-model encoder DQN:
  success_rate: 0.671875
  collision_rate: 0.234375
  timeout_rate: 0.109375
  mean_return: 21.01755456711468

Frozen random encoder DQN:
  success_rate: 0.515625
  collision_rate: 0.375
  timeout_rate: 0.109375
  mean_return: 15.751881019792176

CNN DQN from scratch, 6000 steps:
  success_rate: 0.390625
  collision_rate: 0.328125
  timeout_rate: 0.28125
  mean_return: 15.965488824124074
```

## Paper-Ready Bar

`PAPER_READY=false`

Minimum bar before submission:

- Re-run all main methods on one matched evaluation suite with at least 256 episodes.
- Run at least 3 training seeds for trainable policies.
- Add a fairer CNN baseline with longer training budget.
- Add a pretrained generic visual encoder baseline, such as ImageNet ResNet, CLIP, or DINO-style frozen features.
- Include the random-encoder ablation.
- Include Dreamer4-lite as a limitation/negative result unless it improves materially.
- Include clear failure cases: collision-heavy policies, stall-heavy policies, and reward-model exploitation.
- Keep PX4/Gazebo results as system motivation unless a transfer evaluation succeeds.

## Submission Shape

Best initial venue target is a workshop or systems/benchmark paper, not a top-conference main-track claim.

Recommended framing:

```text
We study how to turn video-prediction world-model features into closed-loop visual control.
We build a reproducible drone-game benchmark and show that predictive visual features improve
small-budget policy learning, while pure prompt-conditioned video rollouts and weakly constrained
imagination expose practical failure modes.
```

## Next Decision

The next paper-critical experiment is not more architecture work. It is a matched, repeated benchmark:

```text
world-model encoder DQN
random encoder DQN
CNN DQN, longer run
pretrained generic encoder DQN
image BC
heuristic
Dreamer4-lite supervised
Dreamer4-lite KL imagination
```

Run each with repeated training seeds where applicable, then evaluate all checkpoints on the same held-out seeds.
