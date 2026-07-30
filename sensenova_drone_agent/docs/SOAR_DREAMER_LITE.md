# SOAR Dreamer-Lite Training

This path temporarily ignores drone runtime and focuses on SOAR as the robotics control dataset.

The goal is to mimic the Dreamer-style structure with the components available here:

```text
frozen Kairos/Sensenova visual features z_t
    -> action-conditioned latent dynamics
    -> isolated agent-token policy/reward/value heads
    -> frozen-dynamics imagination training
```

## Implemented Stack

1. Keep Kairos/Sensenova as frozen visual pretraining.
2. Load SOAR sequence caches containing `z`, `action`, `reward`, `episode`, `step`, `task_id`, and `done`.
3. Train latent dynamics:

```text
z[t-C+1:t], a[t-C+1:t], a[t:t+H-1], task -> z[t+1:t+H]
```

4. Train Dreamer-style agent heads with agent-token isolation:

```text
agent token reads z/action/task context
context/world tokens cannot read agent/task tokens
```

5. Train BC/reward/value heads:

```text
h_t -> action[t:t+L]
h_t -> reward[t:t+L]
h_t -> value_t
```

6. Freeze dynamics and train action/value heads in imagined rollouts using a frozen reward/prior copy.

The dynamics and policy/reward/value outputs now support configurable MLP heads:

```text
--head-layers 2
--head-hidden-dim 1024
```

This keeps the old linear heads as the default while allowing larger heads for SOAR/Kairos ablations.

## Action-Grounding Fixes

These switches were added after the stride-8 summed-action cache still failed the dynamics gate:

```text
--future-action-offset N
```

Tests action/frame lag by conditioning on actions starting at `t + N`.

```text
--motion-filter-quantile Q
--min-motion-norm X
```

Keeps high-motion windows so static visual persistence does not dominate the loss.

```text
--delta-loss-weight W
```

Adds loss on `z[t+k] - z[t]`, making the objective care about motion rather than only reconstructing static latent content.

```text
--contrastive-loss-weight W
--contrastive-margin M
```

Adds a true-vs-counterfactual action loss:

```text
true action prediction must beat shuffled-action prediction by margin M
true action prediction must beat zero-action prediction by margin M
```

This directly trains the same property measured by the action-conditioning gate.

## Main Script

```bash
python sensenova_drone_agent/scripts/train_soar_dreamer_lite.py \
  --stage all \
  --sequence-cache sensenova_drone_agent/output/soar_sequence_cache_rgb32_task_balanced_512_stride8_sum/soar_rgb32_512traj160_stride8_sum_trajectory_success.npz \
  --out-dir sensenova_drone_agent/output/soar_dreamer_lite_stride8_control_smoke_v2 \
  --context-len 4 \
  --prediction-horizon 4 \
  --mtp-horizon 4 \
  --hidden-dim 128 \
  --num-layers 1 \
  --num-heads 4 \
  --bc-epochs 3 \
  --imagination-epochs 1 \
  --imagination-horizon 4 \
  --batch-size 128 \
  --device cuda
```

Use Docker when the host Python environment lacks Torch:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-pybullet-drones-gpu:local \
  bash -lc 'python sensenova_drone_agent/scripts/train_soar_dreamer_lite.py ...'
```

## Outputs

Each run writes:

```text
config.json
summary.json
report.md
best_dynamics_bc.pt
best_imagination.pt
last.pt
```

The summary includes action-conditioning controls:

```text
normal_mse
shuffle_future_actions_mse
zero_future_actions_mse
persistence_mse
normal_over_persistence
shuffle_over_normal
zero_over_normal
strict_gate_passed
action_conditioning_strength
```

## Current Evidence

Synthetic action-causal cache:

```text
normal_over_persistence = 0.856
shuffle_over_normal     = 1.162
zero_over_normal        = 1.166
strict_gate_passed      = true
```

SOAR RGB stride-8 smoke cache:

```text
normal_over_persistence = 0.995
shuffle_over_normal     = 1.002
zero_over_normal        = 1.002
strict_gate_passed      = false
action_conditioning_strength = weak
```

Long Kairos/Wan VAE-flat tuned run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_kairos_tuned_v1
bc_epochs = 120
imagination_epochs = 20
learning_rate = 0.0002
imagination_learning_rate = 0.00003
prior_loss_weight = 3.0
normal_over_persistence = 0.984
shuffle_over_normal     = 1.027
zero_over_normal        = 0.994
strict_gate_passed      = false
action_conditioning_strength = none
```

Kairos/Wan VAE-flat dynamics-focused run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_kairos_dynamics_focused_v1
dynamics_loss_weight = 5.0
action/reward/value_loss_weight = 0.0
normal_over_persistence = 0.970
shuffle_over_normal     = 1.034
zero_over_normal        = 0.997
strict_gate_passed      = false
action_conditioning_strength = none
```

Interpretation: the training scaffold works, and lower LR plus stronger prior regularization prevents the imagination policy from drifting as badly. But the current Kairos/Wan SOAR cache still does not pass the action-conditioning gate because zero future actions are not worse than normal future actions. The next target is data/action alignment: export a Kairos/Wan VAE cache with temporal stride and summed action intervals, then rerun the same gate.

Kairos/Wan VAE-flat stride-8 summed-action cache:

```text
cache = sensenova_drone_agent/output/soar_sequence_cache_kairos_task_balanced_512_stride8_sum/soar_kairos_flat128_512traj320_stride8_sum_trajectory_success.npz
steps = 8281
episodes = 489
tasks = 64
action_mean_abs = 0.717
reward_mean = 0.285
```

Large dynamics gate, context 16:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_kairos_stride8_sum_big_dynamics_v1
hidden_dim = 512
num_layers = 4
num_heads = 8
head_layers = 2
head_hidden_dim = 1024
valid_anchors = 580
train_dynamics_mse = 0.178
val_dynamics_mse = 0.357
normal_over_persistence = 1.236
shuffle_over_normal = 1.001
zero_over_normal = 0.970
strict_gate_passed = false
action_conditioning_strength = none
```

Large dynamics diagnostic, context 8:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_kairos_stride8_sum_big_dynamics_ctx8_v1
valid_anchors = 1747
train_dynamics_mse = 0.165
val_dynamics_mse = 0.335
normal_over_persistence = 1.150
shuffle_over_normal = 1.036
zero_over_normal = 0.919
strict_gate_passed = false
action_conditioning_strength = none
```

Latest interpretation:

```text
MLP heads and a larger transformer increase train-set fit but do not produce a reliable validation-time
action-conditioned simulator. Persistence still beats learned dynamics, and zero future actions remain
too competitive. Do not resume policy/reward/value BC or imagination RL from this cache.
```

Action-grounding smoke:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_smoke/action_grounding_smoke_run
motion_filter_quantile = 0.2
delta_loss_weight = 0.5
contrastive_loss_weight = 0.5
normal_over_persistence = 0.996
shuffle_over_normal = 1.063
zero_over_normal = 1.052
action_conditioning_strength = weak
```

This confirms the new loss path executes and can penalize shuffled/zero future actions.

Action-grounded SOAR/Kairos dynamics run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3
stage = dynamics_bc
future_action_offset = 0
motion_filter_quantile = 0.50
dynamics_loss_weight = 8.0
delta_loss_weight = 1.0
contrastive_loss_weight = 0.75
contrastive_margin = 0.02
valid_anchors = 874
normal_over_persistence = 0.940
shuffle_over_normal = 1.065
zero_over_normal = 1.408
strict_gate_passed = true
action_conditioning_strength = strong
```

This is the first SOAR/Kairos cache run where true future actions beat persistence, shuffled future
actions, and zero future actions.

Gated full BC/imagination run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_action_grounding_full_ctx8_v1
stage = all
imagination_epochs = 20
imagination_val_mean_return0 = 7.043
imagination_val_prior_mse = 0.396
normal_over_persistence = 1.004
shuffle_over_normal = 1.000
zero_over_normal = 1.000
strict_gate_passed = false
```

Important note:

```text
Joint dynamics + policy/reward/value BC can select a checkpoint that loses action grounding.
The next protocol should train dynamics to strict gate pass, freeze/load that dynamics checkpoint,
then train policy/reward/value heads separately.
```

Frozen-dynamics agent BC/imagination run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_frozen_agent_bc_imagination_ctx8_v1
stage = agent_bc_imagination
checkpoint = sensenova_drone_agent/output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3/best_dynamics_bc.pt
agent_bc_epochs = 120
imagination_epochs = 20
```

Dynamics control gate before agent training:

```text
normal_over_persistence = 0.940
shuffle_over_normal = 1.063
zero_over_normal = 1.408
strict_gate_passed = true
action_conditioning_strength = strong
```

Dynamics control gate after frozen-dynamics agent BC:

```text
normal_over_persistence = 0.940
shuffle_over_normal = 1.061
zero_over_normal = 1.408
strict_gate_passed = true
action_conditioning_strength = strong
```

Agent BC and imagination result:

```text
best_agent_bc_epoch = 1
best_agent_bc_val_action_mse = 0.875
final_agent_bc_val_action_mse = 1.101
final_agent_bc_val_reward_mse = 0.602
final_agent_bc_val_value_mse = 48.852
final_imagination_val_mean_return0 = 3.884
final_imagination_val_prior_mse = 0.253
```

Interpretation:

```text
The split protocol fixes the dynamics-regression bug: policy/reward/value training no longer destroys
the action-conditioned latent simulator. The agent head still overfits the small filtered validation split,
so promotion now depends on stronger agent-head regularization, earlier stopping, or a larger SOAR
cache/split before treating imagination returns as policy-quality evidence.
```

Regularized frozen-dynamics agent BC/imagination run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_frozen_agent_bc_regularized_ctx8_v1
stage = agent_bc_imagination
checkpoint = sensenova_drone_agent/output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3/best_dynamics_bc.pt
val_ratio = 0.20
dropout = 0.25
learning_rate = 0.00005
weight_decay = 0.01
agent_bc_metric = loss
agent_bc_early_stop_patience = 12
agent_bc_min_delta = 0.001
```

Dynamics control gate before regularized agent training:

```text
normal_over_persistence = 0.900
shuffle_over_normal = 1.100
zero_over_normal = 1.473
strict_gate_passed = true
action_conditioning_strength = strong
```

Dynamics control gate after regularized agent training:

```text
normal_over_persistence = 0.900
shuffle_over_normal = 1.108
zero_over_normal = 1.473
strict_gate_passed = true
action_conditioning_strength = strong
```

Agent BC and imagination result:

```text
agent_bc_epochs_run = 29
agent_bc_early_stop_triggered = true
best_agent_bc_epoch = 17
best_agent_bc_val_loss = 1.602
best_agent_bc_val_action_mse = 0.924
best_agent_bc_val_reward_mse = 0.427
best_agent_bc_val_value_mse = 29.660
final_agent_bc_val_action_mse = 1.005
final_agent_bc_val_reward_mse = 0.371
final_agent_bc_val_value_mse = 29.276
final_imagination_val_mean_return0 = 10.608
final_imagination_val_prior_mse = 0.899
```

Interpretation:

```text
Early stopping and stronger regularization reduced agent-head overfit and preserved the strict
action-grounded dynamics gate. The remaining gap is reward/value calibration: the value head still has
large validation error, so imagined returns are useful for plumbing and relative ablations but are not yet
strong enough evidence for a policy-quality claim.
```

Reward-calibrated long frozen-dynamics run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2
stage = agent_bc_imagination
checkpoint = sensenova_drone_agent/output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3/best_dynamics_bc.pt
motion_filter_quantile = 0.25
valid_anchors = 1310
train_anchors = 1046
val_anchors = 264
reward_target_mode = raw
reward_loss_type = bce
value_target_mode = raw_discounted_sum
value_loss_type = huber
value_loss_weight = 0.5
bc_epochs = 500
agent_bc_early_stop_patience = 60
agent_bc_epochs_run = 63
imagination_epochs = 100
prior_loss_weight = 5.0
elapsed_s = 240
```

Best agent BC checkpoint:

```text
best_agent_bc_epoch = 3
best_agent_bc_val_loss = 1.324
best_agent_bc_val_action_mse = 0.837
best_agent_bc_val_reward_loss = 0.158
best_agent_bc_val_reward_brier = 0.047
best_agent_bc_val_reward_accuracy = 0.936
best_agent_bc_val_value_mse = 3.228
best_agent_bc_val_value_mae = 0.866
best_agent_bc_val_value_corr = 0.721
```

Reward/value calibration:

```text
reward_target_mean = 0.091
reward_pred_mean = 0.132
reward_brier = 0.047
reward_ece_10 = 0.071
reward_accuracy = 0.936
value_target_mean = 0.808
value_pred_mean = 0.690
value_mse = 3.228
value_mae = 0.866
value_corr = 0.721
```

Dynamics gate after calibrated agent training:

```text
normal_over_persistence = 0.852
shuffle_over_normal = 1.100
zero_over_normal = 1.628
strict_gate_passed = true
action_conditioning_strength = strong
```

Long imagination result:

```text
final_imagination_val_mean_return0 = 19.514
final_imagination_val_mean_reward = 0.119
final_imagination_val_value_mse = 1.189
final_imagination_val_prior_mse = 0.0116
```

Interpretation:

```text
The calibration patch fixed the reward scale problem: reward predictions are now bounded probabilities
instead of normalized unbounded scores, and value error is much lower than the earlier normalized-value
runs. The policy can now run a long frozen-dynamics imagination phase without destroying action
grounding or drifting far from the prior. The remaining proof gap is transfer: we still need a held-out
closed-loop or counterfactual rollout evaluation showing that higher imagined return corresponds to
better real SOAR/drone behavior.
```

SOAR learned-model transfer eval:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2/soar_transfer_eval_h16
bc_checkpoint = best_agent_bc.pt
imagination_checkpoint = best_imagination.pt
rollout_horizon = 16
val_anchors = 264
strict_dynamics_gate = true
```

Closed-loop learned-model rollout:

```text
zero_action_model_return = 0.811
zero_action_mean_reward = 0.051

bc_prior_model_return = 1.243
bc_prior_mean_reward = 0.079
bc_prior_action_norm = 0.426

after_imagination_model_return = 1.737
after_imagination_mean_reward = 0.111
after_imagination_prior_plan_mse = 0.009
after_imagination_action_norm = 0.420

return_delta_after_minus_bc = 0.494
return_ratio_after_over_bc = 1.397
model_transfer_improved = true
prior_constrained = true
```

Open-loop held-out action/value fit:

```text
bc_prior_action_mse = 0.837
after_imagination_action_mse = 0.844
bc_prior_value_mse = 3.228
after_imagination_value_mse = 328.584
bc_prior_value_corr = 0.721
after_imagination_value_corr = -0.002
```

Interpretation:

```text
The policy head improved under closed-loop rollouts in the learned SOAR dynamics while staying close
to the BC prior. This is the first SOAR-only transfer signal for the imagination-trained policy. However,
the value head drifted badly after imagination training on held-out real SOAR contexts, so the next
fix should regularize or freeze/calibrate value during imagination rather than treating the value head
itself as a reliable deployment score.
```

Freeze-value imagination run:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1
stage = imagination
checkpoint = sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2/best_agent_bc.pt
imagination_epochs = 100
imagination_horizon = 8
imagination_train_value_head = false
imagination_value_loss_weight = 0.0
real_value_replay_loss_weight = 0.0
prior_loss_weight = 5.0
elapsed_s = 223
```

Final imagination metrics:

```text
final_val_mean_return0 = 1.962
final_val_mean_reward = 0.138
final_val_prior_mse = 0.0344
final_val_value_loss_weight = 0.0
final_val_real_value_replay_loss = 0.656
```

Dynamics gate:

```text
normal_over_persistence = 0.852
shuffle_over_normal = 1.092
zero_over_normal = 1.628
strict_gate_passed = true
action_conditioning_strength = strong
```

Freeze-value transfer eval:

```text
output = sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/soar_transfer_eval_h16
rollout_horizon = 16
val_anchors = 264

zero_action_model_return = 0.811
bc_prior_model_return = 1.243
after_imagination_model_return = 2.099
return_delta_after_minus_bc = 0.856
return_ratio_after_over_bc = 1.689

bc_prior_mean_reward = 0.079
after_imagination_mean_reward = 0.134
after_imagination_prior_plan_mse = 0.031
model_transfer_improved = true
prior_constrained = true
```

Open-loop held-out fit after freeze-value imagination:

```text
bc_prior_action_mse = 0.837
after_imagination_action_mse = 0.846
bc_prior_value_mse = 3.228
after_imagination_value_mse = 3.228
bc_prior_value_corr = 0.721
after_imagination_value_corr = 0.721
```

Interpretation:

```text
Freezing the value head fixes the previous value-drift failure mode. The imagination-trained action
heads now improve closed-loop return inside the learned SOAR dynamics while preserving the BC
reward/value calibration. This is stronger evidence for moving toward imagination training, but it is
still a learned-model result: the next proof step is to evaluate the resulting action policy against
held-out real SOAR sequences or a real/sim environment that supplies fresh observations.
```
