# Behavior Cloning Midtraining

## Purpose

This is phase 2 in the Dreamer-style stack:

```text
1. World-model pretraining       already done by Kairos/Sensenova
2. Behavior-cloning midtraining  implemented here
3. Imagination RL                next phase, not this stage
```

The phase-2 job is to train agent heads on frozen world-model states:

```text
h_t -> action[t:t+L]
h_t -> reward[t:t+L]
h_t -> value[t]
```

The world model is not updated here.

## Cache Schema

The trainer consumes a generic `.npz` sequence cache:

```text
z        required, (N, z_dim), frozen Kairos/Sensenova/world-model features
action   required, (N, action_dim), behavior labels
reward   optional, (N,), scalar reward/success/hindsight score
episode  optional, (N,), trajectory id
step     optional, (N,), timestep inside trajectory
task_id  optional, (N,), integer task id
```

For SOAR, `scripts/export_soar_sequence_cache.py` maps ZIP-resident robot episodes into this schema.

## Trainer

Implementation:

```text
src/sensenova_drone/midtraining.py
scripts/train_behavior_cloning_midtraining.py
```

The model uses:

```text
frozen latent context z[t-C+1:t]
previous action context a[t-C:t-1]
task token
agent token
small transformer encoder
multi-token action heads
multi-token reward heads
value head
```

This is a practical version of the paper's phase-2 objective:

```text
L = sum_n loss(action[t+n] | h_t) + sum_n loss(reward[t+n] | h_t)
```

For continuous robot actions, action loss is MSE on normalized actions.

## Smoke Test

Create a synthetic cache:

```bash
python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
  --make-smoke-cache sensenova_drone_agent/output/bc_midtraining_smoke/cache.npz
```

Inspect the cache without torch:

```bash
python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
  --sequence-cache sensenova_drone_agent/output/bc_midtraining_smoke/cache.npz \
  --out-dir sensenova_drone_agent/output/bc_midtraining_smoke \
  --dry-run
```

Train in the GPU container:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
    --sequence-cache sensenova_drone_agent/output/bc_midtraining_smoke/cache.npz \
    --out-dir sensenova_drone_agent/output/bc_midtraining_smoke/train \
    --context-len 4 \
    --mtp-horizon 2 \
    --hidden-dim 64 \
    --num-layers 1 \
    --num-heads 4 \
    --epochs 2 \
    --batch-size 64 \
    --device cuda"
```

Verified smoke result:

```text
valid anchors: 336
train anchors: 302
val anchors: 34
epoch 1 val loss: 2.4782
epoch 2 val loss: 1.5940
checkpoint: output/bc_midtraining_smoke/train/best.pt
```

## SOAR Path

SOAR numpy data can be used without full extraction:

```text
SOAR frames/actions/rewards
  -> zip-native SOAR schema adapter
  -> sequence cache .npz
  -> BC midtraining
  -> behavioral prior checkpoint
  -> imagination RL
```

Exporter:

```text
scripts/export_soar_sequence_cache.py
```

The first verified exporter uses an RGB-flat placeholder feature:

```text
frame -> resize to 32x32 -> flatten normalized RGB -> z
```

This is intentionally not the final Kairos/Sensenova latent path. It verifies the real-data phase-2
pipeline before replacing `z` with frozen world-model latents.

The exporter now also supports frozen Kairos/Wan VAE features:

```text
--feature kairos_vae       frame -> Wan VAE latent -> channel mean/std pooled z
--feature kairos_vae_flat  frame -> Wan VAE latent -> flattened full latent z
```

## SOAR Smoke Test

Downloaded ZIP:

```text
data/robotics/soar/soar-dataset-numpy.zip
```

Download verification:

```text
size: 25.31 GiB
zip entries: 347703
first entry: soar-dataset-local/
trajectory count discovered by exporter: 31812
```

Export command:

```bash
python3 sensenova_drone_agent/scripts/export_soar_sequence_cache.py \
  --out sensenova_drone_agent/output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz \
  --summary-json sensenova_drone_agent/output/soar_sequence_cache_smoke/summary.json \
  --max-trajectories 6 \
  --max-steps-per-trajectory 32 \
  --frame-size 32 \
  --seed 7
```

Export result:

```text
selected trajectories: 6
exported steps: 192
feature: rgb_flat
frame size: 32
z_dim: 3072
action_dim: 7
episodes: 6
tasks: 5
success trajectories represented: 3
valid anchors, C=8/L=8: 96
```

Inspection command:

```bash
python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
  --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz \
  --out-dir sensenova_drone_agent/output/soar_bc_midtraining_smoke/inspect_compact \
  --context-len 8 \
  --mtp-horizon 8 \
  --dry-run
```

Training command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
    --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz \
    --out-dir sensenova_drone_agent/output/soar_bc_midtraining_smoke/train \
    --context-len 8 \
    --mtp-horizon 4 \
    --hidden-dim 128 \
    --num-layers 1 \
    --num-heads 4 \
    --epochs 3 \
    --batch-size 32 \
    --device cuda \
    --num-workers 0"
```

Training result:

```text
valid anchors: 120
train anchors: 108
val anchors: 12
epoch 1 val loss: 0.8057
epoch 2 val loss: 0.7646
epoch 3 val loss: 0.7188
checkpoint: output/soar_bc_midtraining_smoke/train/best.pt
```

Interpretation:

```text
SOAR zip -> frames/actions/tasks/rewards -> generic sequence cache -> BC midtraining works.
```

The next gate is to export the same SOAR trajectories with Kairos/Sensenova frozen latents instead
of RGB-flat features, then scale the trajectory count and run action-shuffle/control baselines.

## Strict Phase-2 Validation

The trainer now includes the corrections needed to make this closer to the Dreamer-style phase-2 target:

```text
agent-token isolation mask
episode-heldout validation
task-stratified episode validation
MTP horizon 8
positive-reward-only BC action loss
metric-specific checkpoints
early stopping
control perturbations
```

The larger task-balanced SOAR cache is:

```text
output/soar_sequence_cache_kairos_task_balanced_512/soar_kairos_flat128_512traj32_trajectory_success.npz
```

Cache summary:

```text
episodes: 512
tasks: 64
steps: 16384
feature: kairos_vae_flat
z_dim: 4096
valid anchors, C=8/L=8: 8192
```

Best current positive-BC checkpoint:

```text
output/soar_bc_midtraining_kairos_task_balanced_512_positive_bc/train_seed2_earlystop/best_bc_action_mse.pt
```

Validation report:

```text
output/soar_midtraining_validation_v5_task_balanced_512_positive_bc/report.md
```

Current result:

```text
normal seed mean best BC action MSE: 0.7945
normal seed std: 0.0294
positive-reward mean-action baseline: 0.8875
positive-reward repeat-previous-action baseline: 1.2130
shuffle_targets ratio vs normal: 1.022
zero_prev_actions ratio vs normal: 1.069
zero_z_context ratio vs normal: 0.996
training duration status: overfit_after_best
reward/value heads validated: false
ready for imagination RL: false
```

Interpretation:

```text
The phase-2 structure is now close to the theoretical target.
The action prior learns something from successful SOAR behavior and previous-action context matters.
The visual latent signal is not yet proven because zero/shuffled z controls are about tied.
Trajectory-level success labels are too weak for a reliable reward/value model under held-out episodes.
More epochs are harmful; use early stopping around the best BC action epoch.
```

Do not use this reward/value head for imagination RL yet. The next useful step is better supervision:

```text
1. Keep BC action loss on successful/relevant trajectories.
2. Replace trajectory-level success reward with denser event/progress labels where possible.
3. Prefer coherent task families with repeated demonstrations over broad one-off tasks.
4. Re-run the same validation suite and require controls to be clearly worse.
```

## SOAR Medium RGB-Flat Scaling Test

Export command:

```bash
python3 sensenova_drone_agent/scripts/export_soar_sequence_cache.py \
  --out sensenova_drone_agent/output/soar_sequence_cache_medium/soar_rgb32_32traj.npz \
  --summary-json sensenova_drone_agent/output/soar_sequence_cache_medium/summary.json \
  --max-trajectories 32 \
  --max-steps-per-trajectory 32 \
  --frame-size 32 \
  --seed 11
```

Export result:

```text
selected trajectories: 32
exported steps: 1024
feature: rgb_flat
z_dim: 3072
action_dim: 7
episodes: 32
tasks: 23
success trajectories represented: 16
valid anchors, C=8/L=8: 512
```

Training command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
    --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_medium/soar_rgb32_32traj.npz \
    --out-dir sensenova_drone_agent/output/soar_bc_midtraining_medium_rgb32/train \
    --context-len 8 \
    --mtp-horizon 4 \
    --hidden-dim 128 \
    --num-layers 1 \
    --num-heads 4 \
    --epochs 5 \
    --batch-size 64 \
    --device cuda \
    --num-workers 0"
```

Training result:

```text
valid anchors: 640
train anchors: 576
val anchors: 64
best epoch: 3
best val loss: 1.3315
best val action MSE: 0.6572
last epoch train loss: 1.0825
last epoch val loss: 1.3344
last epoch val action MSE: 0.6198
checkpoint: output/soar_bc_midtraining_medium_rgb32/train/best.pt
```

Interpretation:

```text
The pipeline scales past smoke size. Action prediction improves, but sparse success reward/value
learning is still noisy at this small validation size.
```

## SOAR Kairos/Wan VAE-Flat Smoke Test

Export command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/export_soar_sequence_cache.py \
    --feature kairos_vae_flat \
    --max-trajectories 2 \
    --max-steps-per-trajectory 16 \
    --kairos-height 128 \
    --kairos-width 128 \
    --out sensenova_drone_agent/output/soar_sequence_cache_kairos_smoke/soar_kairos_flat128_2traj16.npz \
    --summary-json sensenova_drone_agent/output/soar_sequence_cache_kairos_smoke/summary.json \
    --seed 13"
```

Export result:

```text
selected trajectories: 2
exported steps: 32
feature: kairos_vae_flat
latent shape per frame: [1, 16, 1, 16, 16]
z_dim: 4096
action_dim: 7
episodes: 2
tasks: 2
valid anchors, C=8/L=4: 8
```

Trainer inspection:

```bash
python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
  --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_kairos_smoke/soar_kairos_flat128_2traj16.npz \
  --out-dir sensenova_drone_agent/output/soar_bc_midtraining_kairos_smoke/inspect \
  --context-len 8 \
  --mtp-horizon 4 \
  --dry-run
```

Tiny training command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
    --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_kairos_smoke/soar_kairos_flat128_2traj16.npz \
    --out-dir sensenova_drone_agent/output/soar_bc_midtraining_kairos_smoke/train \
    --context-len 8 \
    --mtp-horizon 4 \
    --hidden-dim 64 \
    --num-layers 1 \
    --num-heads 4 \
    --epochs 2 \
    --batch-size 8 \
    --device cuda \
    --num-workers 0"
```

Tiny training result:

```text
valid anchors: 8
train anchors: 7
val anchors: 1
epoch 1 val loss: 0.4646
epoch 2 val loss: 0.3994
checkpoint: output/soar_bc_midtraining_kairos_smoke/train/best.pt
```

Interpretation:

```text
SOAR video -> Kairos/Wan VAE-flat latents -> generic sequence cache -> BC midtraining works.
```

This is a plumbing proof only. The cache is too small to claim a learned policy.

## SOAR Kairos/Wan VAE-Flat Medium Midtraining

Export command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/export_soar_sequence_cache.py \
    --feature kairos_vae_flat \
    --max-trajectories 32 \
    --max-steps-per-trajectory 32 \
    --kairos-height 128 \
    --kairos-width 128 \
    --out sensenova_drone_agent/output/soar_sequence_cache_kairos_medium/soar_kairos_flat128_32traj32.npz \
    --summary-json sensenova_drone_agent/output/soar_sequence_cache_kairos_medium/summary.json \
    --seed 17"
```

Export result:

```text
selected trajectories: 32
exported steps: 1002
feature: kairos_vae_flat
latent shape per frame: [1, 16, 1, 16, 16]
z_dim: 4096
action_dim: 7
episodes: 32
tasks: 20
valid anchors, C=8/L=4: 620
```

Training command:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc "python3 sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py \
    --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_kairos_medium/soar_kairos_flat128_32traj32.npz \
    --out-dir sensenova_drone_agent/output/soar_bc_midtraining_kairos_medium/train \
    --context-len 8 \
    --mtp-horizon 4 \
    --hidden-dim 128 \
    --num-layers 2 \
    --num-heads 4 \
    --epochs 25 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --device cuda \
    --num-workers 0"
```

Training result:

```text
train anchors: 558
val anchors: 62
best epoch: 25
epoch 1 val loss: 1.1302
epoch 1 val action MSE: 0.7620
epoch 25 val loss: 0.8486
epoch 25 val action MSE: 0.4698
epoch 25 train loss: 0.4342
checkpoint: output/soar_bc_midtraining_kairos_medium/train/best.pt
```

Interpretation:

```text
Frozen Kairos/Wan VAE latents contain enough signal for the phase-2 head to improve SOAR action prediction on held-out anchors.
```

Reward/value learning remains weak because the current reward is sparse final-step success only.
This checkpoint is a stronger behavioral prior than the tiny smoke run, but it is still not an
imagination-trained controller.

## SOAR Kairos/Wan VAE-Flat Control Baselines

The trainer supports control modes:

```text
normal              aligned latents, previous actions, and future targets
shuffle_targets     future action/reward labels come from another anchor
shuffle_z_context   visual latent context comes from another anchor
zero_z_context      visual latent context is removed
zero_prev_actions   previous-action context is removed
```

Baseline suite output:

```text
output/soar_bc_midtraining_kairos_medium_baselines/
```

Results:

```text
normal best val action MSE:              0.4698
shuffle_targets best val action MSE:     0.8158  (1.74x worse)
shuffle_z_context best val action MSE:   0.6073  (1.29x worse)
zero_z_context best val action MSE:      0.5440  (1.16x worse)
zero_prev_actions best val action MSE:   0.5525  (1.18x worse)
mean-action control val action MSE:      0.8265
repeat-last-action val action MSE:       0.9443
```

Interpretation:

```text
Aligned Kairos/Wan VAE latents improve action prediction beyond action-distribution priors and simple temporal persistence.
```

The effect is real but still modest. Previous-action context also carries signal, and reward/value
learning is not decisive yet because the current reward is sparse final-step success only.

## SOAR Success/Failure Reward Labels

SOAR provides trajectory-level labels:

```text
success.txt       true/false per trajectory
language_task.txt task string per trajectory
```

The exporter supports:

```text
--reward-mode final_success            reward 1 only on the final exported frame of successful trajectories
--reward-mode trajectory_success       reward 1 on every frame of successful trajectories
--reward-mode linear_success_progress  reward ramps from 0 to 1 across successful trajectories
--reward-mode signed_trajectory_success      success=+1, failure=-1 for every frame
--reward-mode signed_final_success           success/failure label only on the final frame
--reward-mode signed_linear_success_progress signed progress ramp over the trajectory
--reward-mode linear_success_progress_with_action_penalty
```

Reward-mode comparison on the 32-trajectory Kairos/Wan VAE-flat cache:

```text
final_success:
  best val action MSE: 0.4698
  best val reward MSE: 0.1951
  best val value MSE: 0.7770

trajectory_success:
  best val action MSE: 0.4281
  best val reward MSE: 0.0015
  best val value MSE: 0.0041

linear_success_progress:
  best val action MSE: 0.4831
  best val reward MSE: 0.0048
  best val value MSE: 0.0661
```

Report:

```text
output/soar_bc_midtraining_kairos_medium_reward_modes/report.md
```

Interpretation:

```text
SOAR trajectory-level success/failure labels are directly usable for reward-head midtraining.
```

For the next larger phase-2 run, use `trajectory_success` as the default reward mode. Keep
`final_success` as the sparse-event ablation.

## SOAR Kairos/Wan VAE-Flat Large Trajectory-Success Midtraining

Export:

```text
cache: output/soar_sequence_cache_kairos_large/soar_kairos_flat128_128traj32_trajectory_success.npz
feature: kairos_vae_flat
reward mode: trajectory_success
episodes: 128
steps: 4096
tasks: 60
z_dim: 4096
action_dim: 7
reward_mean: 0.5
valid anchors, C=8/L=4: 2560
```

Training:

```text
output: output/soar_bc_midtraining_kairos_large_trajectory_success/train
checkpoint: output/soar_bc_midtraining_kairos_large_trajectory_success/train/best.pt
epochs: 30
train anchors: 2304
val anchors: 256
best epoch: 29
best val loss: 0.4559
best val action MSE: 0.4460
best val reward MSE: 0.0012
best val value MSE: 0.0072
```

Comparison to 32-trajectory trajectory-success run:

```text
32 trajectories:
  val anchors: 62
  tasks: 20
  best val action MSE: 0.4281
  best val reward MSE: 0.0015
  best val value MSE: 0.0041

128 trajectories:
  val anchors: 256
  tasks: 60
  best val action MSE: 0.4460
  best val reward MSE: 0.0012
  best val value MSE: 0.0072
```

Report:

```text
output/soar_bc_midtraining_kairos_large_trajectory_success/report.md
```

Interpretation:

```text
The larger trajectory_success run is a better phase-2 prior candidate because it validates across more tasks and anchors while keeping reward/value heads strong.
```

## Dreamer-Style Corrections

We added the four missing pieces needed before treating this as a closer Dreamer-style phase-2 stack:

```text
1. action-conditioned latent dynamics:
   z_context + action_sequence + task -> future_z

2. stronger reward/progress labels:
   trajectory success, signed success, signed progress, optional action penalty

3. task-conditioned agent heads:
   task token + agent token + policy/reward/value heads

4. better training mixture and controls:
   task-balanced SOAR sampling, relevant/uniform sampler, positive-reward-only BC option,
   held-out episode/task split, action-shuffle/zero-action controls, metric-specific checkpoints
```

Dreamer-style relevant/uniform BC sampling can be enabled with:

```text
--relevant-sample-fraction 0.5
```

This samples half of train windows from reward-relevant windows and half from the remaining data
when both groups are available.

Sampler smoke:

```text
output: output/soar_bc_mixture_sampler_smoke
requested relevant fraction: 0.5
sampler enabled: true
train relevant windows: 4380
train non-relevant windows: 4580
```

New dynamics doc:

```text
docs/ACTION_CONDITIONED_LATENT_DYNAMICS.md
```

Key result:

```text
Synthetic action-driven latent data:
  normal best val z MSE: 0.1564
  shuffled future-action best val z MSE: 0.3380
  zero future-action best val z MSE: 0.3371
  persistence MSE: 0.5135

SOAR Kairos/Wan VAE-flat 512-trajectory probe:
  normal best val z MSE: 0.141892
  shuffled future-action best val z MSE: 0.141896
  zero future-action best val z MSE: 0.141890
  persistence MSE: 0.141856
```

Interpretation:

```text
The architecture can use action tokens when action signal exists. The current SOAR Kairos/Wan
one-frame VAE-flat cache does not yet expose a useful action-conditioned transition signal.
```

## Claim Boundary

This phase can support:

```text
We can precondition policy/reward/value heads from real robot video-action data on top of frozen Kairos/Wan VAE latent sequences.
We have the action-conditioned latent dynamics scaffold required before imagination RL.
```

It does not yet support:

```text
The policy improved through imagination RL.
Kairos controls drones.
SOAR Kairos/Wan VAE-flat latents are already a validated controllable simulator.
```

## Promotion Gate

Before using a cache for policy/reward/value midtraining or imagination RL, run:

```text
scripts/run_action_conditioning_gate.py
```

Promotion requires:

```text
normal future actions beat persistence
normal future actions beat shuffled future actions
normal future actions beat zero future actions
```

Latest SOAR Kairos/Wan VAE-flat 512-trajectory gate:

```text
output: output/soar_action_conditioning_gate_kairos_task_balanced_512_offsets
ready_for_bc_or_imagination: false
passed_offsets: []
best_normal_vs_persistence_ratio: 1.000273
best_shuffle_vs_normal_ratio: 1.000047
best_zero_vs_normal_ratio: 1.000012
```

Latest RGB-flat sanity gate:

```text
output: output/soar_action_conditioning_gate_rgb32_medium
ready_for_bc_or_imagination: false
normal_vs_persistence_ratio: 1.019120
shuffle_vs_normal_ratio: 0.999323
zero_vs_normal_ratio: 0.997915
```

Best temporal/action-aggregated RGB gate:

```text
output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride8_sum_ctx4_e30
frame_stride: 8
action_aggregation: sum
ready_for_bc_or_imagination: false
normal_vs_persistence_ratio: 0.918304
shuffle_vs_normal_ratio: 1.049823
zero_vs_normal_ratio: 1.040453
```

This is a weak positive action signal, but it still fails the hard promotion threshold because zero
future actions remain too competitive.

Decision:

```text
Do not proceed to policy/reward/value promotion or imagination RL from these caches yet.
The next blocker is stronger action grounding, likely via explicit robot state/state deltas or a more
directly controllable first-person environment.
```
