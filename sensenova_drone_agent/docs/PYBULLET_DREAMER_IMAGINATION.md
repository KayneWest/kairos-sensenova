# PyBullet Drone Imagination Training

## Goal

Test a small Dreamer-style control loop in a drone simulator:

```text
real PyBullet transition data
  -> learned latent dynamics model
  -> imagined rollouts through frozen dynamics
  -> policy update
  -> real PyBullet evaluation
```

This is intentionally lightweight. It is not Dreamer 4, and it does not yet train inside Kairos video rollouts. It proves the local training scaffold for model-based policy improvement.

## Implemented

Trainer:

```text
sensenova_drone_agent/scripts/train_pybullet_drones_imagination_policy.py
```

Docker wrapper:

```text
sensenova_drone_agent/scripts/run_pybullet_drones_imagination_policy.sh
```

GPU Docker image:

```text
sensenova_drone_agent/docker/Dockerfile.pybullet_drones_gpu
sensenova_drone_agent/scripts/build_pybullet_drones_gpu_image.sh
```

Fixed-seed visual/Kairos suite runner:

```text
sensenova_drone_agent/scripts/run_pybullet_drones_visual_imagination_suite.py
```

The trainer does:

```text
1. Collect PyBullet drone transitions.
2. Encode observations into a feature vector z_t.
3. Train LatentDynamics(z_t, action_t) -> z_{t+1}, reward_t, done_t.
4. Train a behavior-cloned actor from the collected actions.
5. Freeze the learned dynamics.
6. Update the actor inside imagined rollouts.
7. Evaluate BC prior and imagined actor back in real PyBullet.
8. Save selected_actor_state, promoting the imagined actor only if real eval improves.
```

## Proper Imagination Objective

The trainer now supports two imagination objectives:

```text
pmpo      Dreamer-style stochastic imagined rollout update
backprop  Older direct reward-gradient path through the learned dynamics
```

The default is now `pmpo`. It follows the paper more closely:

```text
1. Start imagined rollouts from real dataset states.
2. Sample actions from the current stochastic policy.
3. Step the frozen learned dynamics model.
4. Predict rewards and values along the imagined trajectory.
5. Compute lambda returns.
6. Split sampled actions into positive- and negative-advantage groups.
7. Increase likelihood of positive-advantage actions.
8. Decrease likelihood of negative-advantage actions.
9. Penalize drift from the frozen BC prior.
10. Evaluate the resulting policy back in real PyBullet.
```

This avoids the earlier failure mode where the actor directly exploited differentiable reward gradients through an imperfect learned simulator.

## Important Safety Fixes

The first imagination runs exposed expected model-exploitation failure modes:

```text
- The actor could emit direction vectors outside the normalized action manifold.
- The value bootstrap could explode and make the actor optimize critic artifacts.
```

Fixes now in place:

```text
- Actor direction output is normalized to match the collected PyBullet action space.
- Default actor imagination objective uses PMPO-style advantage signs.
- Critic is trained on lambda returns from imagined trajectories.
- Checkpoint selection refuses to promote an imagined actor if real PyBullet eval regresses.
```

## Deep PMPO Run

Command:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_imagination_policy.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_imagination_pmpo_deep_kinematic_v1 \
  --feature kinematic \
  --collect-episodes 64 \
  --eval-episodes 10 \
  --max-steps 192 \
  --world-epochs 300 \
  --bc-epochs 200 \
  --imagination-updates 1000 \
  --imagination-horizon 24 \
  --batch-size 256 \
  --hidden-dim 384 \
  --initial-xy-range 0.8 \
  --initial-z-min 0.1 \
  --initial-z-max 0.9 \
  --behavior random_mix \
  --random-action-prob 0.35 \
  --behavior-noise 0.25 \
  --imagination-objective pmpo \
  --prior-weight 1.0 \
  --policy-lr 0.00005 \
  --critic-lr 0.0005 \
  --policy-std 0.08 \
  --lambda-return 0.95 \
  --max-grad-norm 5.0
```

Result:

```text
transitions: 12288
feature: kinematic
world z_mse: 0.0418

BC prior:
  success_rate: 1.0
  mean_final_distance_m: 0.0840
  mean_min_distance_m: 0.0584

After imagination:
  success_rate: 1.0
  mean_final_distance_m: 0.0674
  mean_min_distance_m: 0.0476

Selected actor:
  after_imagination
```

Output:

```text
sensenova_drone_agent/output/pybullet_drones_imagination_pmpo_deep_kinematic_v1
```

## Current Conclusion

The model-based RL scaffold works technically:

```text
- A learned latent simulator trains from collected drone transitions.
- A policy can be updated through imagined rollouts.
- The updated policy is evaluated in the real PyBullet simulator.
- The system detects and rejects harmful imagined-policy updates.
- With enough kinematic transition data and a conservative PMPO-style objective, imagination improved final distance while preserving 100% success on the held-out evaluation starts.
```

The current positive result is still on privileged kinematic state features. The next question is whether the same improvement survives with visual features, especially Kairos/Sensenova-derived features.

## Pixel/Kairos Transition

The trainer now supports agent-visible temporal visual state:

```text
real PyBullet RGB frame history
  -> visual feature encoder
  -> stacked features + optional deltas + previous action
  -> learned dynamics and policy heads
```

Implemented feature modes include:

```text
rgb_downsample    downsampled RGB pixels
kairos_vae_flat   Kairos/Wan VAE latent flattened from rendered RGB
```

The important control change is that `z_t` no longer has to be privileged simulator state. It can now be built from the rendered camera observation and previous action:

```text
z_t = encoder(frame_t, frame_history, previous_action)
```

The imagined rollout loop remains the same:

```text
z_t, a_t -> learned dynamics -> z_{t+1}, reward_t, done_t
policy(z_t) -> action distribution
PMPO updates policy/value inside imagined z-space
real PyBullet evaluation overwrites imagined state
```

## Pixel Smoke Results

RGB temporal stack run:

```text
output: sensenova_drone_agent/output/pybullet_drones_imagination_rgb_stack_small_v2_stable
feature: rgb_downsample
feature_dim: 1348
transitions: 3994
feature_stack: 4
feature_stack_deltas: true
include_prev_action_in_feature: true

BC prior:
  success_rate: 0.0
  mean_final_distance_m: 0.3547

After imagination:
  success_rate: 0.1667
  mean_final_distance_m: 0.3076

Selected actor:
  after_imagination
```

This is a weak but real pixel result: the policy improved from 0/6 to 1/6 successes and reduced final distance using only stacked RGB-derived features plus previous action.

Kairos VAE-flat smoke run:

```text
output: sensenova_drone_agent/output/pybullet_drones_imagination_kairos_flat_small_v1
feature: kairos_vae_flat
feature_dim: 3076
transitions: 512
feature_stack: 2
feature_stack_deltas: true
include_prev_action_in_feature: true

BC prior:
  success_rate: 0.0
  mean_final_distance_m: 0.3849

After imagination:
  success_rate: 0.0
  mean_final_distance_m: 0.3757

Selected actor:
  after_imagination
```

This proves the Kairos/Wan VAE latent path works end to end, but it is not yet strong evidence that Kairos features solve the control task. It slightly improved final distance but did not reach a success in the small evaluation.

## Cached Fixed-Seed Suite

The trainer now supports reusable transition caches:

```text
--dataset-cache <path>
--reuse-dataset-cache
```

The cache stores precomputed:

```text
z_t, action_t, reward_t, z_{t+1}, done_t
```

This matters for Kairos because VAE encoding is the expensive part. Once a cache exists, repeated policy/world-model training can reuse the same visual/Kairos latent dataset without re-rendering PyBullet frames or recomputing Kairos features.

Kairos VAE encoding now runs on GPU when available:

```text
image: sensenova_drone_agent-pybullet-drones-gpu:local
base: mkrzus/director-mode:5090-runtime
torch: 2.8.0+cu128
GPU: RTX 5090
wrapper behavior: auto-selects GPU image when present and NVIDIA Docker is available
override GPU use: SENSENOVA_DOCKER_GPUS=none
override image: SENSENOVA_PYBULLET_IMAGE=<image>
```

The trainer also supports fixed evaluation seeds:

```text
--eval-seeds 171000,171001,171002,171003,171004,171005
```

This makes feature comparisons stricter: RGB, Kairos, and kinematic baselines can be evaluated on the same held-out starts.

Suite runner smoke checks:

```text
RGB suite:
  output: sensenova_drone_agent/output/visual_suite_rgb_smoke_v1
  cache reuse verified: true
  cached transitions: 32
  selected actor: bc_prior

Kairos VAE-flat suite:
  output: sensenova_drone_agent/output/visual_suite_kairos_smoke_v1
  feature: kairos_vae_flat
  selected actor: after_imagination
  BC final distance: 0.8433
  after-imagination final distance: 0.8428
```

The suite smoke results are not performance evidence; they verify reproducible experiment plumbing.

Kairos CUDA smoke:

```text
output: sensenova_drone_agent/output/visual_suite_kairos_cuda_smoke_v1
feature: kairos_vae_flat
kairos_device: cuda
cached transitions: 32
cache reuse verified: true
```

Kairos CUDA medium:

```text
output: sensenova_drone_agent/output/visual_suite_kairos_cuda_medium_v1
feature: kairos_vae_flat
kairos_device: cuda
elapsed_s: 20.5
cached transitions: 512
eval seeds: 171050, 171051, 171052

BC prior:
  success_rate: 0.0
  mean_final_distance_m: 0.4670

After imagination:
  success_rate: 0.0
  mean_final_distance_m: 0.4389

Selected actor:
  after_imagination
```

Kairos CUDA medium v2:

```text
output: sensenova_drone_agent/output/visual_suite_kairos_cuda_medium_v2
feature: kairos_vae_flat
kairos_device: cuda
elapsed_s: 21.7
cached transitions: 512
eval seeds: 171060, 171061, 171062

BC prior:
  success_rate: 0.0
  mean_final_distance_m: 0.5838

After imagination:
  success_rate: 0.0
  mean_final_distance_m: 0.6447

Selected actor:
  bc_prior
```

This is a useful negative case. The imagined update regressed in real PyBullet evaluation, and the checkpoint gate correctly rejected it.

Fixed-seed RGB small suite:

```text
output: sensenova_drone_agent/output/visual_suite_rgb_small_fixed_v1
feature: rgb_downsample
train seed: 170030
eval seeds: 171030, 171031, 171032
cached transitions: 4096

BC prior:
  success_rate: 0.0
  mean_final_distance_m: 0.4409

After imagination:
  success_rate: 0.3333
  mean_final_distance_m: 0.3232

Selected actor:
  after_imagination
```

This is the strongest pixel-only result so far because it uses a reusable cache and fixed evaluation starts.

## Dreamer3/Dreamer4 Reference Usage

Local reference repos:

```text
dreamerv3/
dreamer4/
```

How they help this project:

```text
Dreamer3:
  - stable separation of world-model learning, imagination, actor/value learning, and real-environment evaluation
  - lambda-return and replay-context discipline

Dreamer4:
  - action-conditioned dynamics interface
  - isolated agent-token idea: policy/reward/value can read world state, but world dynamics should not be contaminated by task/policy tokens
  - action-shuffle/action-sensitivity metrics as future checks for whether the model actually uses actions
```

## Ad Hoc Action Tokens

The learned dynamics now supports two conditioning modes:

```text
concat        original baseline: [z_t, action_t] -> MLP
action_token  Dreamer4-inspired baseline: z_t token + action_t token -> small transformer -> heads
```

The learned dynamics now supports two training modes:

```text
one_step   original baseline: independent transition loss
sequence   contiguous trajectory-window loss over cached episode/step metadata
```

The visual suite defaults to:

```text
--dynamics-action-conditioning action_token
--world-training-mode sequence
```

The trainer now reports action-sensitivity probes:

```text
action_shuffle_loss_ratio
action_shuffle_z_mse_ratio
action_effect_rms
sequence_action_shuffle_loss_ratio
sequence_action_effect_rms
```

Interpretation:

```text
ratio near 1.0   dynamics can predict about as well with shuffled actions, so actions are mostly ignored
ratio above 1.0  dynamics predictions degrade when actions are shuffled, so actions matter
```

Kairos medium comparison on the same cached transition set:

```text
concat dynamics:
  output: sensenova_drone_agent/output/visual_suite_kairos_concat_probe_medium_v1
  action_shuffle_loss_ratio: 1.002
  action_effect_rms: 0.0178
  selected_actor: bc_prior

action-token dynamics:
  output: sensenova_drone_agent/output/visual_suite_kairos_action_token_medium_v1
  action_shuffle_loss_ratio: 1.146
  action_effect_rms: 0.1888
  selected_actor: bc_prior
```

This supports the hypothesis from Dreamer4: explicit action tokens make the learned dynamics more action-sensitive. It does not yet make the imagined policy reliably better. The next issue is safe policy optimization inside imperfect action-conditioned latents.

Dreamer4-style sequence correction:

```text
output: sensenova_drone_agent/output/visual_suite_kairos_sequence_medium_v1
feature: kairos_vae_flat
world_training_mode: sequence
dynamics_action_conditioning: action_token
cached transitions: 482
sequence_windows: 426
sequence_length: 8
cache has episode/step metadata: true
z_mse: 0.0948
sequence_action_shuffle_loss_ratio: 1.004
sequence_action_effect_rms: 0.0212
selected_actor: bc_prior
```

This corrects the training shape mismatch but exposes the next issue: even with sequence training, the Kairos latent dynamics is still only weakly action-sensitive on the current dataset. We need more action-diverse trajectories and/or action-weighted dynamics losses before trusting imagination updates.

Dreamer4-style imagination rollout correction:

```text
output: sensenova_drone_agent/output/visual_suite_kairos_sequence_context_medium_v1
feature: kairos_vae_flat
world_training_mode: sequence
dynamics_action_conditioning: action_token
cached transitions: 482
sequence_windows: 426
sequence_length: 8
imagination_context_length: 8
z_mse: 0.0898
sequence_action_shuffle_loss_ratio: 1.0027
selected_actor: bc_prior
bc_prior_mean_final_distance_m: 0.6387
after_imagination_mean_final_distance_m: 0.6678
```

This fixes the remaining sequence mismatch in policy optimization: imagined PMPO rollouts now call the same `forward_sequence()` path used by the world-model training loop. The result is still not a successful Kairos-latent controller because the learned latent simulator remains weakly action-grounded.

Current implementation status:

```text
We are not yet running official Dreamer3 or Dreamer4 training.
We are using their design pattern to keep our small PyBullet/Kairos scaffold honest.
The next deeper port would replace this small action-token transformer with a larger Dreamer4-style dynamics model and action-weighted training mix.
```

## Claim Boundary

We can claim:

```text
We implemented and verified a Dreamer-style drone imagination-training scaffold where a learned latent simulator can improve a kinematic drone policy under real-simulator validation.
We extended the scaffold from privileged kinematic state to pixel-derived and Kairos/Wan VAE-derived features.
The first RGB pixel run shows small real-simulator improvement after imagination training.
The first Kairos latent run executes end to end and slightly improves final distance.
```

We cannot yet claim:

```text
Kairos/Sensenova itself is the learned simulator for policy improvement.
The learned simulator is accurate enough for safe policy optimization.
Kairos/Sensenova latents currently solve the drone-control task.
The visual policy is robust across seeds, worlds, or harder obstacle layouts.
```

## Next Work

Needed for this to become paper-grade:

```text
1. Add ensemble dynamics or uncertainty penalties to reduce model exploitation.
2. Run the fixed-seed visual/Kairos suite at `small` and `overnight` presets.
3. Add stronger visual encoders: CNN, ResNet, DINO/CLIP, and larger Kairos latent variants.
4. Add Dreamer4-style action-sensitivity checks by comparing normal-action and shuffled-action dynamics losses.
5. Promote imagined updates only with held-out real-sim validation gates.
6. Later replace the small MLP dynamics with native Kairos/Sensenova action-conditioned dynamics.
```
