# Paper Experiment Plan

## Goal

Produce a defensible paper table for the claim:

```text
Predictive world-model visual representations improve small-budget closed-loop control
in a first-person drone navigation game, compared with random features and CNN-from-scratch baselines.
```

## Main Research Questions

1. Does world-model pretraining improve visual control sample efficiency?
2. Does the benefit remain under matched safety shielding?
3. Does imagination training improve over supervised policy heads, or does it exploit imperfect learned rewards?
4. Which failures are representation failures, reward failures, or safety failures?

## Methods To Compare

Required:

- `heuristic`: hand-coded geometric baseline.
- `image_bc`: RGB behavior cloning from DQN teacher episodes.
- `cnn_dqn`: RGB CNN DQN trained from scratch.
- `random_encoder_dqn`: frozen random world-model encoder plus DQN.
- `world_model_encoder_dqn`: frozen pretrained action-conditioned world-model encoder plus DQN.
- `dreamer4_lite_supervised`: frozen world-model encoder/dynamics with BC/reward/value heads.
- `dreamer4_lite_kl_imagination`: conservative KL-constrained imagination update.

Strongly recommended:

- `resnet_dqn`: frozen ImageNet ResNet features plus DQN.
- `clip_dqn` or `dinov2_dqn`: frozen generic visual foundation features plus DQN.
- `cnn_dqn_long`: CNN DQN from scratch with a budget comparable in wall-clock or environment steps to the world-model path.

## Controlled Evaluation

Use one held-out seed suite for the headline table:

```text
episodes: 256 minimum, 1000 preferred
seed_start: fixed and documented
image_size: fixed
enabled_actions: hover,yaw_left,yaw_right,forward,strafe_left
shield_threshold: 1.0m for shielded methods
```

For trainable methods:

```text
training_seeds: at least 3
report: mean +/- standard error across training seeds
checkpoint selection: validation/eval protocol fixed before test seeds
```

## Metrics

Primary:

- Success rate.
- Collision rate.
- Timeout rate.
- Mean return.

Secondary:

- Mean episode length.
- Minimum front clearance.
- Action distribution.
- Repeated yaw/hover loop rate.
- Deployment score:

```text
100*success_rate + mean_return - 50*collision_rate - 10*timeout_rate
```

The deployment score is only for compact sorting; the paper should discuss the raw metrics.

## Ablations

Representation ablations:

- Frozen pretrained world-model encoder.
- Same architecture, frozen random encoder.
- Trainable CNN from scratch.
- Generic pretrained image encoder.

Safety ablations:

- No shield.
- Runtime shield only.
- Shield in training loop.
- Reward-shaped safety.

Imagination ablations:

- Supervised heads only.
- Weak KL imagination.
- Strong KL imagination.
- Held-out real-environment selection gate.

Dataset ablations:

- Small BC dataset.
- Full DQN dataset.
- Hard-negative branch dataset.
- Multi-step recovery dataset.

## Figures

Figure 1: System diagram.

```text
real observation -> encoder/world model -> policy/value/reward heads -> safety shield -> action -> environment
```

Figure 2: Headline benchmark table/bar chart.

Figure 3: Sample-efficiency curve.

```text
x-axis: environment steps or training examples
y-axis: success and collision
```

Figure 4: Failure taxonomy.

```text
collision-heavy
stall/timeout-heavy
reward-model exploitation
action-conditioning failure
```

Figure 5: Qualitative traces/contact sheets.

## Paper Outline

1. Introduction: closed-loop use of world models should replace generated next state with real observations.
2. Related work: world models, Dreamer-style agents, visual RL, safety-constrained control.
3. Benchmark: first-person drone game, observations, actions, rewards, safety events.
4. Methods: world-model pretraining, frozen-feature DQN, BC, CNN baseline, Dreamer4-lite imagination.
5. Results: matched benchmark and ablations.
6. Failure modes: Kairos action conditioning, reward exploitation, safety/progress tradeoff.
7. Limitations: toy game, no robust PX4 transfer yet, small model scale, limited training seeds.
8. Conclusion: predictive representations help, but action grounding and safety-constrained learning are the bottleneck.

## Immediate Work Queue

1. Generate current paper table:

```bash
python3 sensenova_drone_agent/scripts/generate_paper_results.py \
  --out-dir sensenova_drone_agent/output/paper_results_v1
```

2. Add a longer CNN DQN run.

3. Add one frozen generic pretrained encoder baseline.

4. Build a unified benchmark runner that evaluates all saved checkpoints on the same 256 or 1000 seeds.

5. Repeat the strongest trainable methods across 3 seeds.

6. Update `sensenova_drone_agent/output/paper_results_v1/paper_results.md` from the final summaries.

7. Add an external drone benchmark sanity suite:

```text
gym-pybullet-drones HoverAviary
```

Track it in:

```text
sensenova_drone_agent/docs/PYBULLET_DRONES_BENCHMARK.md
```

Feature-policy result tracking:

```text
sensenova_drone_agent/docs/PYBULLET_DRONES_FEATURE_POLICY.md
```

Current external-benchmark claim boundary:

```text
Kairos/Wan VAE features are usable by a learned action head. Flattened VAE
latents improve over pooled VAE channel stats and beat cnn_pixels/random_projection
by final distance in the longer two-seed suite, but they are not yet superior to
ResNet18 or simple downsampled RGB. Do not claim Kairos/Sensenova controls drones
better than baselines until stronger feature extraction and repeated-seed results
exist.
```

8. Add an external visual RL generalization smoke benchmark:

```text
Procgen CoinRun
```

Track it in:

```text
sensenova_drone_agent/docs/PROCGEN_BENCHMARK.md
```

## Stop Conditions

Do not write the paper around a claim until these are true:

- The world-model encoder beats random encoder on repeated seeds.
- The world-model encoder beats CNN-from-scratch under a fair small-budget comparison.
- The generic pretrained visual encoder baseline does not erase the result, or the claim is revised.
- The limitation that Kairos rollouts are not yet reliable MPC teachers is explicit.
