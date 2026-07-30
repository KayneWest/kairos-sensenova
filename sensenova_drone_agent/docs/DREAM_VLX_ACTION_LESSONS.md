# Dream-VLX Lessons for Action Grounding

Date: `2026-05-18`

Purpose: extract implementation lessons from the cloned `Dream-VLX` repo that are directly relevant to our action-conditioned dynamics and imagination-training path.

## Relevant Dream-VLX Findings

Dream-VLA is not using generic latent-dynamics imagination training. It is primarily a VLA action-prediction stack trained on curated robot demonstrations, with diffusion/flow action heads.

Important implementation details:

- LIBERO fine-tuning uses `_no_noops` datasets. Their README explicitly says near-zero/no-op actions are filtered before training.
- Their LIBERO regeneration script filters unsuccessful demonstrations and no-op transitions. A no-op is defined as near-zero non-gripper motion with unchanged gripper action.
- Their dataset transform predicts action chunks, not a single action. For LIBERO, the default is `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`.
- Only action tokens contribute supervised loss. Prompt/vision/language tokens are masked out of the loss.
- Dream-VLA predicts normalized raw robot actions. It does not train the policy head to emit handcrafted expanded action features.
- Their strongest recipe uses flow matching over continuous action chunks: sample noise, interpolate between noise and ground-truth action chunk, then predict velocity toward the ground-truth action.
- Their fine-tuning recipe uses wrist image and proprio when available.
- Their published fine-tuning script uses LoRA rank `32`, learning rate `1e-4`, image augmentation, and long schedules around `200k` steps for LIBERO.

## Difference From Our Current Native Imagination Path

Our current game-action native dynamics run used expanded action features:

```text
current,prev,delta,mean4,norm
```

For the game-action run, this expanded the action input to `61` dimensions. That is useful for making the dynamics model action-sensitive, but it is not the correct executable policy output.

The current imagination script also sets:

```text
feature_dim = 2 * z_dim + config.action_dim
policy output dim = config.action_dim
```

So when `action_dim=61`, the policy learns to output the expanded dynamics feature vector itself. This is probably the wrong abstraction. A deployable policy should output raw environment actions, and a deterministic adapter should expand those raw actions into the dynamics-conditioning representation.

## Practical Changes We Should Make

1. Split raw actions from dynamics action features.

```text
raw_action_dim: executable action size
dynamics_action_dim: expanded action-conditioning size
```

The policy/prior/action head should predict `raw_action_dim` or `action_chunk * raw_action_dim`. The dynamics model can still receive `current,prev,delta,mean4,norm` features produced by a deterministic adapter.

2. Add action-chunk behavior cloning.

Instead of only predicting one aligned action per latent state, add an action head that predicts chunks:

```text
z/context/task -> raw actions[t : t + chunk]
```

Start with L1/MSE for continuous actions and cross-entropy for one-hot discrete actions. Then add flow matching over raw action chunks for continuous-control datasets.

3. Add no-op filtering and no-op-aware evaluation.

For robot datasets, filter near-zero/no-op transitions where the state/action does not actually change. For game datasets, either exclude explicit no-op actions during policy training or keep them only for a controlled no-op baseline.

4. Keep expanded action features only inside dynamics.

The dynamics input should receive the expanded action-conditioning features. The policy should not output those features directly.

5. Re-run the action-grounding gate after the split.

Required pass criteria:

```text
normal actions beat shuffled actions
normal actions beat zero/no-op actions
policy raw actions beat zero/no-op under learned reward
policy raw actions can be rolled through the dynamics adapter without shape hacks
```

## Why This Matters

Our current result shows that the dynamics model can become action-conditioned, but the policy/reward/imagination loop is still weak against zero-action. Dream-VLX points to three likely causes:

- no-op transitions dilute the action signal;
- policy output is not cleanly separated from dynamics-conditioning features;
- action prediction should be chunked and trained as a first-class objective.

The next serious run should therefore be a Dream-VLA-style midtraining pass, not another reward-tuning-only run.

## Next Implementation Target

Build a new native action-head training path:

```text
dataset raw actions
  -> raw action chunk labels
  -> policy/action head predicts raw action chunks
  -> deterministic feature adapter expands predicted raw actions
  -> frozen action-conditioned dynamics rolls future latents
  -> reward/value/imagination training operates on executable raw actions
```

This preserves the useful action-sensitive dynamics we already trained while making the agent output space match the actual control interface.

## Implemented Retrofit Pass

Date: `2026-05-18`

Code changes:

- `dreamer4/dreamer4/wm_dataset.py` now returns raw executable actions separately from expanded dynamics features:
  - `raw_act`
  - `raw_act_mask`
  - existing `act` remains the expanded dynamics action-feature tensor.
- `WMDataset` now accepts:
  - `raw_action_dim`
  - `require_non_noop`
  - `no_op_threshold`
  - `min_non_noop_steps`
- `sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py` now supports:
  - `--policy-action-source raw`
  - `--raw-action-dim`
  - `--action-chunk-len`
  - raw-action BC targets with future action chunks.
  - deterministic raw-action to expanded-dynamics-action conversion during imagination rollout.
- `sensenova_drone_agent/scripts/experiments/launch_game_actions_native_imagination.sh` and payload now pass the raw-action settings.

Validated smoke:

```bash
RUN_ID=raw_policy_smoke_v1 \
BC_STEPS=2 \
IMAGINATION_UPDATES=1 \
EVAL_BATCHES=1 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
ACTION_CHUNK_LEN=4 \
POLICY_ACTION_SOURCE=raw \
RAW_ACTION_DIM=15 \
./sensenova_drone_agent/scripts/experiments/launch_game_actions_native_imagination.sh
```

Output:

```text
sensenova_drone_agent/output/dreamer4_game_actions_imagination_raw_policy_smoke_v1
```

Smoke result:

```text
policy_action_source: raw
action_chunk_len: 4
policy_return_delta: +0.0006
policy_prior_mse_delta: -0.0026
runtime/shape validation: pass
```

Interpretation: this smoke validates the Dream-VLX-inspired plumbing, not policy quality. The policy now outputs raw executable actions, while the frozen dynamics still receives expanded action features. The next meaningful run is a longer raw-policy run with enough BC steps for the chunk prior to become competent before imagination updates.

## Raw-Policy Full Run

Date: `2026-05-18`

Output:

```text
sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_v1
```

Configuration:

```text
policy_action_source: raw
raw_action_dim: 15
dynamics_action_dim: 61
action_features: current,prev,delta,mean4,norm
action_chunk_len: 4
bc_steps: 1200
imagination_updates: 400
eval_batches: 64
```

BC result:

```text
action_mse step 1:    0.1618
action_mse step 1200: 0.0489
```

Held-out learned-return result:

```text
before policy-minus-BC:   +0.0071
before policy-minus-zero: -0.0014
after policy-minus-BC:    -0.0009
after policy-minus-zero:  -0.0093
policy_return_delta:      -0.0080
policy_prior_mse_delta:   -0.0048
```

Interpretation:

- The raw-action policy/chunk path trains and rolls through the frozen dynamics correctly.
- The BC prior learns a usable raw-action chunk predictor under supervised loss.
- The imagination update still does not improve the held-out policy; it moves closer to the BC prior but lowers learned return.
- This points back to reward/advantage quality rather than action-output representation as the immediate blocker.

Next required experiment: run a `no_update` control and then try reward/advantage changes on the raw-policy path before scaling. The current raw-policy result is an implementation pass, not an imagination-training performance pass.

## Raw-Policy No-Update Control

Date: `2026-05-18`

Output:

```text
sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_no_update_v1
```

Result:

```text
before policy-minus-BC:   +0.0071
after policy-minus-BC:    +0.0071
before policy-minus-zero: -0.0014
after policy-minus-zero:  -0.0014
policy_return_delta:       0.0000
policy_prior_mse_delta:    0.0000
```

Interpretation: the raw-policy evaluation path is deterministic and stable. The degradation in `blocks_raw_policy_v1` is caused by the imagination update, not by evaluation drift or the raw-action adapter.

## Reward-Filtered Raw-Policy Run

Date: `2026-05-18`

Motivation from Dream-VLX/Dream-VLA: train on demonstrations with clear useful behavior rather than mostly background/no-op transitions. We added dataset-level reward-signal filtering:

```text
--reward-filter-mode none|positive_sum|abs_sum|any_positive|any_abs
--reward-signal-threshold 0.0
--min-reward-signal-steps 1
```

Run filter:

```text
reward_filter_mode: any_positive
reward_signal_threshold: 0.0
min_reward_signal_steps: 1
```

Filtered data size:

```text
procgen-bigfish-random: 154 windows
procgen-coinrun-random: 0 windows
procgen-jumper-random: 1 window
vizdoom-basic-random: 8 windows
total: 163 windows
train split: 145 windows
eval split: 18 windows
```

No-update control:

```text
output: sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_reward_any_positive_no_update_v1
action_mse step 1200: 0.0355
policy-minus-BC:       +0.0533
policy-minus-zero:     +0.4758
policy_return_delta:    0.0000
```

Imagination update:

```text
output: sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_reward_any_positive_v1
action_mse step 1200:      0.0355
before policy-minus-BC:   +0.0533
after policy-minus-BC:    +0.0066
after policy-minus-zero:  +0.4292
policy_return_delta:      -0.0466
policy_prior_mse_delta:   -0.0018
```

Interpretation:

- Reward filtering materially improves the supervised/raw-policy signal: the filtered BC policy beats both BC-prior and zero-action under the learned reward model.
- The current imagination update still reduces held-out learned return relative to the pre-imagination policy, even though the final policy remains better than BC-prior and zero-action.
- The filtered set is too small and too BigFish-dominated for a broad claim, but it is the first clean positive signal that “reward-clear windows” make the agent heads meaningful.
- Next step is not more RL updates on this tiny subset; it is collecting/adding more reward-identifiable trajectories or using a stricter success-demo dataset, then repeating this gate.

## Balanced Reward Mixture Sampler

Date: `2026-05-18`

Motivation: Dreamer4-HF has many positive windows but no negative reward windows. A literal SMOTE-style interpolation is a bad fit for trajectory data because it fabricates invalid action/state sequences. The safer alternative is balanced resampling across source and reward buckets so the policy/reward/value heads see success windows, mixed positive windows, and low/zero-return controls in known proportions.

Added training options:

```text
--train-sampling-mode shuffle|dreamer4_reward_mixture
--train-balance-spec hf_expert_positive=0.25,hf_mixed_positive=0.25,hf_mixed_zero=0.25,soar_game_positive=0.25
--train-balance-return-threshold 0.0
--train-balanced-samples 0
--train-balance-seed 0
```

Reusable launcher:

```text
sensenova_drone_agent/scripts/experiments/launch_all_data_balanced_imagination.sh
sensenova_drone_agent/scripts/experiments/all_data_balanced_imagination_payload.sh
```

Smoke output:

```text
sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_smoke_launcher_v1
```

Sampler availability after episode holdout:

```text
hf_expert_positive: 322,135 windows, 37 tasks
hf_mixed_positive: 2,127,492 windows, 36 tasks
hf_mixed_zero: 811,123 windows, 21 tasks
soar_game_positive: 12,709 windows, 64 tasks
```

Launcher smoke result:

```text
train_sampling_mode: dreamer4_reward_mixture
sample_count: 64
episode-holdout train windows: 3,294,587
episode-holdout eval windows: 366,848
policy-minus-BC: +0.000092
policy-minus-zero: -0.000902
exit code: 0
```

Decision: the sampler and launcher are ready for long midtraining runs. Do not treat this as an imagination-training result yet. The current all-data native dynamics checkpoint is still a smoke checkpoint and fails the action-conditioning gate:

```text
direct_action_conditioning_detected: false
autoregressive_action_conditioning_detected: false
native_dynamics_ready_for_imagination: false
```

Next gate: train or continue an action-conditioned all-data dynamics model until normal actions beat shuffled and zero actions. Then run this balanced sampler with `IMAGINATION_MODE=train`.

## Balanced Midtraining Run

Date: `2026-05-18`

Run:

```text
container: sda-dreamer4-all-data-balanced-imagination-balanced_midtrain_v1
output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_balanced_midtrain_v1
mode: no_update
```

Training setup:

```text
bc_steps: 1200
batch_size: 4
train_balanced_samples: 6400
eval_batches: 64
eval_samples: 256
policy_action_source: raw
action_chunk_len: 4
target_normalization: per_task
```

Balanced sample allocation:

```text
hf_expert_positive: 1624 sampled windows from 322,135 available windows across 37 tasks
hf_mixed_positive: 1568 sampled windows from 2,127,492 available windows across 36 tasks
hf_mixed_zero: 1539 sampled windows from 811,123 available windows across 21 tasks
soar_game_positive: 1669 sampled windows from 12,709 available windows across 64 tasks
```

BC training checkpoints:

```text
step 1:    loss 3.7792, action_mse 0.1026, reward_mse 3.3867, value_mse 1.8726
step 120:  loss 0.3583, action_mse 0.0635, reward_mse 0.2039, value_mse 0.2741
step 480:  loss 0.3642, action_mse 0.0887, reward_mse 0.1787, value_mse 0.0796
step 1200: loss 2.7194, action_mse 0.2508, reward_mse 2.1818, value_mse 0.3592
```

Held-out evaluation:

```text
zero-action learned return: 1.2405
BC-prior learned return:   1.3787
policy learned return:     1.3530
policy-minus-BC:          -0.0256
policy-minus-zero:        +0.1125
mean per-task policy-minus-BC: -0.0275
```

Interpretation: balanced midtraining produced a nontrivial policy that beats zero-action under the learned reward model, but it still does not beat the BC prior. This is acceptable as a calibration run because `IMAGINATION_MODE=no_update` intentionally avoided RL updates. It suggests the heads are learning useful signal, but not enough to justify scaling imagination RL until the dynamics checkpoint passes the action-conditioning gate.

## Action-Conditioned Dynamics Gate

Date: `2026-05-19`

Run:

```text
container: sda-dreamer4-all-data-action_cond_gate20k_v1
output: sensenova_drone_agent/output/dreamer4_all_data_native_action_cond_gate20k_v1
base dynamics: sensenova_drone_agent/output/dreamer4_all_data_native_smoke/dynamics_ckpts/latest.pt
steps: 20,000
action_contrast_start: 500
action_contrast_weight: 1.0
```

Training-log action contrast tail:

```text
tail mean shuffle/normal: 1.1536
tail mean zero/normal:    1.2418
```

Post-run held-out dynamics eval:

```text
direct normal:  0.03196
direct shuffle: 0.03275  (shuffle/normal 1.0248)
direct zero:    0.03207  (zero/normal 1.0036)

autoregressive normal:      0.02576
autoregressive shuffle:     0.02660  (shuffle/normal 1.0324)
autoregressive zero:        0.02515  (zero/normal 0.9763)
autoregressive persistence: 0.02811  (normal/persistence 0.9164)
```

Decision:

```text
direct_action_conditioning_detected: true
autoregressive_action_conditioning_detected: true
autoregressive_beats_persistence: true
native_dynamics_ready_for_imagination: true
```

Interpretation: the formal gate passed mainly on shuffled-action degradation and beating persistence. Zero-action remains mixed in autoregressive rollout, so this is a sufficient gate for an imagination-RL attempt, not yet a strong action-conditioned simulator claim.

## Balanced Imagination Training with Gated Dynamics

Date: `2026-05-19`

Run:

```text
container: sda-dreamer4-all-data-balanced-imagination-balanced_imagination_train_gate20k_v1
output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_balanced_imagination_train_gate20k_v1
native_run: sensenova_drone_agent/output/dreamer4_all_data_native_action_cond_gate20k_v1
mode: train
bc_steps: 1200
imagination_updates: 400
train_balanced_samples: 6400
eval_samples: 256
```

Balanced sample allocation:

```text
hf_expert_positive: 1624 sampled windows
hf_mixed_positive: 1568 sampled windows
hf_mixed_zero: 1539 sampled windows
soar_game_positive: 1669 sampled windows
```

Held-out evaluation before imagination:

```text
zero-action learned return: 4.8531
BC-prior learned return:   4.9634
policy learned return:     4.9714
policy-minus-BC:          +0.0080
policy-minus-zero:        +0.1183
```

Held-out evaluation after imagination:

```text
zero-action learned return: 4.8531
BC-prior learned return:   4.9634
policy learned return:     4.9641
policy-minus-BC:          +0.0007
policy-minus-zero:        +0.1110
mean per-task policy-minus-BC: +0.0002
policy_return_delta:      -0.0073
```

Interpretation: this is the first all-data run where the final imagination-trained policy beats both the BC prior and zero-action on held-out learned reward. The margin over BC is tiny, and the RL phase reduced the pre-imagination policy score. The result supports the weaker claim that action-gated dynamics plus balanced midtraining can make imagination training non-destructive enough to remain above BC, but not yet the stronger claim that imagination RL improves the policy beyond its pre-RL checkpoint.

Next gate: tune the imagination objective for positive `policy_return_delta`, likely by reducing updates, checkpoint-selecting the best policy during imagination, lowering `IMAGINATION_LEARNING_RATE`, or using stronger value/reward calibration.

## Best-Selected Imagination Tuning Run

Date: `2026-05-19`

Run:

```text
container: sda-dreamer4-all-data-balanced-imagination-balanced_imagination_selectbest_gate20k_lr1e5_v1
output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_balanced_imagination_selectbest_gate20k_lr1e5_v1
native_run: sensenova_drone_agent/output/dreamer4_all_data_native_action_cond_gate20k_v1
mode: train
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
select_best_imagination: true
imagination_eval_every: 40
best_metric: policy_minus_bc
train_balanced_samples: 6400
eval_samples: 256
```

Selection trace:

```text
update 0:   policy-minus-BC +0.0080, policy-minus-zero +0.1183
update 40:  policy-minus-BC +0.0026, policy-minus-zero +0.1130
update 80:  policy-minus-BC -0.0002, policy-minus-zero +0.1101
update 120: policy-minus-BC +0.0017, policy-minus-zero +0.1120
update 160: policy-minus-BC +0.0031, policy-minus-zero +0.1134
selected: update 0
```

Final held-out evaluation after restoring the selected checkpoint:

```text
zero-action learned return: 4.8531
BC-prior learned return:   4.9634
policy learned return:     4.9714
policy-minus-BC:          +0.0080
policy-minus-zero:        +0.1183
policy_return_delta:       0.0000
```

Interpretation: checkpoint selection works as a safety mechanism and prevents destructive imagination updates, but the RL objective still did not improve beyond the BC-initialized policy. Lowering the imagination learning rate made updates less destructive than the 400-update run, but the best policy remained the pre-RL checkpoint. The next technical blocker is not checkpoint selection; it is reward/value calibration or advantage construction strong enough to make imagined updates reliably improve the held-out learned return.

## Score-Function PMPO Fix

Date: `2026-05-19`

Issue found: the PMPO-style policy loss was computing `log_prob()` on a reparameterized `Normal.rsample()` action without detaching the sampled action. For a Normal distribution, `log_prob(mean + std * eps)` cancels most of the direct score-function gradient to the policy mean. In practice, this made imagination updates mostly act through variance and regularization rather than reliably moving the action mean toward positive-advantage samples.

Fix:

```text
log_prob = dist.log_prob(raw_action_flat.detach())
```

This keeps the sampled action fixed for the likelihood update, matching the intended score-function behavior.

Run:

```text
container: sda-dreamer4-all-data-balanced-imagination-balanced_imagination_scorefn_gate20k_lr1e5_v1
output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_balanced_imagination_scorefn_gate20k_lr1e5_v1
native_run: sensenova_drone_agent/output/dreamer4_all_data_native_action_cond_gate20k_v1
mode: train
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
select_best_imagination: true
imagination_eval_every: 40
best_metric: policy_minus_bc
train_balanced_samples: 6400
eval_samples: 256
```

Selection trace:

```text
update 0:   policy-minus-BC +0.0080, policy-minus-zero +0.1183
update 40:  policy-minus-BC +0.0200, policy-minus-zero +0.1304
update 80:  policy-minus-BC +0.0225, policy-minus-zero +0.1329
update 120: policy-minus-BC +0.0113, policy-minus-zero +0.1217
update 160: policy-minus-BC +0.0080, policy-minus-zero +0.1183
selected: update 80
```

Final held-out evaluation after restoring selected update 80:

```text
zero-action learned return: 4.8531
BC-prior learned return:   4.9634
policy learned return:     4.9859
policy-minus-BC:          +0.0225
policy-minus-zero:        +0.1329
policy_return_delta:      +0.0145
policy_prior_mse_delta:   -0.0002
policy_action_abs_delta:  +0.0041
```

Interpretation: this is the first clean held-out learned-return result where imagination RL improves over both the BC prior and the pre-imagination policy checkpoint. The result is still an internal learned-reward/dynamics result, not a real-environment success claim, but it validates the basic midtraining-plus-imagination loop under the current action-gated dynamics checkpoint.

## Score-Function Repeatability, Larger Eval

Date: `2026-05-19`

Suite:

```text
summary: sensenova_drone_agent/output/all_data_scorefn_repeatability_eval256_v1/summary.json
report: sensenova_drone_agent/output/all_data_scorefn_repeatability_eval256_v1/report.md
seeds: 20260518, 20260519, 20260520
eval_batches: 256
eval_samples: 1024
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
select_best_imagination: true
imagination_eval_every: 40
best_metric: policy_minus_bc
```

Per-seed results:

```text
seed 20260518: selected update 160, policy-minus-BC +0.0261, policy-return-delta +0.0183
seed 20260519: selected update 160, policy-minus-BC +0.0140, policy-return-delta +0.0040
seed 20260520: selected update 0,   policy-minus-BC +0.0160, policy-return-delta +0.0000
```

Aggregate:

```text
completed: 3
strict passes: 2/3
repeatability_pass: true
mean policy-minus-BC:      +0.0187
mean policy-minus-zero:    +0.1617
mean policy-return-delta:  +0.0075
mean per-task policy-minus-BC: +0.0191
mean policy-prior-MSE after: 0.00077
```

Interpretation: the corrected PMPO score-function path repeats under larger held-out evaluation. Two of three seeds improve over their pre-imagination checkpoint; the third does not improve with RL but remains safely above BC and zero-action because checkpoint selection restores update 0. This is enough to move from smoke validation to ablations, but not enough to claim solved control: the claim remains internal to the learned dynamics and learned reward model.

## Score-Function Ablation Controls

Date: `2026-05-19`

Matched seed/control settings:

```text
seed: 20260518
eval_batches: 256
eval_samples: 1024
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
native_run: sensenova_drone_agent/output/dreamer4_all_data_native_action_cond_gate20k_v1
```

Control runs:

```text
no_update:
  output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_ablate_no_update_gate20k_seed20260518_eval256_v1
  selected update: n/a
  policy-minus-BC: +0.0078
  policy-return-delta: 0.0000

old no-detach PMPO path:
  output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_ablate_nodetach_gate20k_seed20260518_eval256_v1
  selected update: 0
  policy-minus-BC: +0.0078
  policy-return-delta: 0.0000

corrected score-function PMPO path:
  output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_repeat_scorefn_gate20k_seed20260518_eval256_v1
  selected update: 160
  policy-minus-BC: +0.0261
  policy-return-delta: +0.0183
```

Old no-detach selection trace:

```text
update 0:   policy-minus-BC +0.0078
update 40:  policy-minus-BC +0.0022
update 80:  policy-minus-BC +0.0026
update 120: policy-minus-BC +0.0018
update 160: policy-minus-BC +0.0025
selected: update 0
```

Interpretation: the ablation supports the score-function diagnosis. With the old non-detached `log_prob(rsample)` path, imagination updates do not improve beyond the no-update baseline and checkpoint selection restores update 0. With detached log-prob actions, the same seed/config improves to update 160 and gains +0.0183 learned return over the pre-imagination policy.

## Full Causal Ablation Sweep

Date: `2026-05-19`

Aggregate files:

```text
sensenova_drone_agent/output/all_data_causal_ablation_eval256_v1/summary.json
sensenova_drone_agent/output/all_data_causal_ablation_eval256_v1/report.md
```

Matched evaluation:

```text
eval_batches: 256
eval_samples: 1024
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
```

Results:

```text
baseline score-function, seed 20260518:
  selected update: 160
  policy-minus-BC: +0.0261
  policy-return-delta: +0.0183

no update, seed 20260518:
  policy-minus-BC: +0.0078
  policy-return-delta: +0.0000

old no-detach log_prob path, seed 20260518:
  selected update: 0
  policy-minus-BC: +0.0078
  policy-return-delta: +0.0000

weak native dynamics checkpoint, seed 20260518:
  selected update: 160
  policy-minus-BC: +0.0481
  policy-return-delta: +0.0754

unbalanced sampler, seed 20260518:
  selected update: 80
  policy-minus-BC: +0.0029
  policy-return-delta: +0.0026

zero action sent to dynamics during RL, seed 20260518:
  selected update: 160
  policy-minus-BC: +0.0251
  policy-return-delta: +0.0173

shuffled action sent to dynamics during RL, seed 20260518:
  selected update: 160
  policy-minus-BC: +0.0271
  policy-return-delta: +0.0193

zero action history in policy/reward/value features during RL, seed 20260518:
  selected update: 80
  policy-minus-BC: +0.0186
  policy-return-delta: +0.0108

shuffled action history in policy/reward/value features during RL, seed 20260518:
  selected update: 160
  policy-minus-BC: +0.0214
  policy-return-delta: +0.0136

best-selected baseline, seed 20260520:
  selected update: 0
  policy-minus-BC: +0.0160
  policy-return-delta: +0.0000

final checkpoint without best selection, seed 20260520:
  policy-minus-BC: +0.0059
  policy-return-delta: -0.0101
```

Interpretation:

```text
Score-function PMPO is causal for the observed gain:
  no-update and old no-detach stay at +0.0078;
  corrected score-function PMPO reaches +0.0261.

Balanced/reward-mixture sampling is causal:
  unbalanced shuffle sampling drops to +0.0029.

Exact action conditioning through the dynamics is not yet causal under this metric:
  zero/shuffled dynamics actions still reach +0.0251/+0.0271.

Action-history features explain part, but not all, of the internal signal:
  zero/shuffled agent action context still select positive checkpoints.

Best-checkpoint selection matters:
  seed 20260520 degrades at the final checkpoint without selection.
```

Claim boundary: these ablations support the narrower claim that the current pipeline has a repeatable learned-reward improvement signal that depends on score-function PMPO and balanced reward sampling. They do not yet support a strong Dreamer-style claim that the policy improves because exact actions causally steer a reliable learned world simulator.

## Reward/Value Action-Context Ablation

Date: `2026-05-19`

Aggregate files:

```text
sensenova_drone_agent/output/all_data_causal_reward_value_eval256_v1/summary.json
sensenova_drone_agent/output/all_data_causal_reward_value_eval256_v1/report.md
```

Question: does the previous positive imagination signal survive if reward/value heads cannot directly read action-history features?

Implementation:

```text
reward_value_action_context_mode: zero
```

This leaves the policy/prior path unchanged, but zeroes action context for reward/value features during behavior cloning, imagination rollouts, and evaluation.

Matched evaluation:

```text
seed: 20260518
eval_batches: 256
eval_samples: 1024
bc_steps: 1200
imagination_updates: 160
imagination_learning_rate: 1e-5
select_best_imagination: true
```

Results:

```text
real dynamics actions:
  selected update: 80
  policy-minus-BC: +0.0028
  policy-return-delta: +0.0018
  policy-minus-zero: +0.0063

zero dynamics actions:
  selected update: 80
  policy-minus-BC: +0.0034
  policy-return-delta: +0.0024
  policy-minus-zero: +0.0069
```

Interpretation:

```text
The earlier seed-20260518 baseline had policy-minus-BC +0.0261.
With reward/value action context removed, the signal collapses to +0.0028.
Zero dynamics actions slightly outperform real dynamics actions.
```

Claim boundary update: the previous positive signal was not a valid Dreamer-style action-causal result. It was compatible with reward/value/action-context leakage or non-causal learned-reward optimization. The next fix should target dynamics/reward identifiability directly: wrong actions must lead to visibly/latently worse predicted futures and worse learned reward, otherwise imagination RL has no reliable simulator to optimize inside.

## Causal Identifiability and Reward-Blind Imagination

Date: `2026-05-19`

Aggregate files:

```text
sensenova_drone_agent/output/all_data_causal_ident_rewardblind_eval256_v1/summary.json
sensenova_drone_agent/output/all_data_causal_ident_rewardblind_eval256_v1/report.md
```

Implementation changes:

```text
Dynamics contrast:
  per-frame active-action contrast instead of global sequence contrast
  negatives: shuffle, zero, time-shift
  active action threshold: 0.001
  first-frame contrast skipped

Dynamics eval:
  strict gate requires both zero and shuffled actions to be worse
  reports direct and autoregressive pair-pass fractions

Agent midtraining:
  reward/value features can be action-blind via reward_value_action_context_mode=zero
  reward counterfactual contrast compares true-action futures to zero/shuffled-action futures
```

Dynamics continuation:

```text
run: dreamer4_all_data_native_causal_ident_gate20k_continue_25k_v1
resume: dreamer4_all_data_native_action_cond_gate20k_v1 at step 20000
target step: 25000
strict gate: passed

direct shuffle/normal:         1.1908
direct zero/normal:            1.1836
autoregressive shuffle/normal: 1.1836
autoregressive zero/normal:    1.1805
autoregressive normal/persist: 0.7499
```

Matched reward-blind imagination:

```text
real dynamics actions:
  selected update: 80
  policy-minus-BC: +0.0060
  policy-minus-zero: +0.0072

zero dynamics actions:
  selected update: 120
  policy-minus-BC: +0.0053
  policy-minus-zero: +0.0065

shuffled dynamics actions:
  selected update: 120
  policy-minus-BC: +0.0090
  policy-minus-zero: +0.0102
```

Interpretation:

```text
Positive:
  The retrofit dynamics now passes a strict held-out action-conditioning gate.

Negative:
  The downstream imagination gain is still not action-causal.
  Shuffled dynamics actions outperform real dynamics actions on the matched seed.
```

Claim boundary update: we can now claim progress on retrofitting a pretrained visual world model into an action-conditioned latent predictor. We still cannot claim Dreamer-style policy improvement caused by exact action-conditioned imagination. The next fix needs to make the reward/value objective depend more sharply on action-conditioned state changes, likely via event/milestone windows, inverse dynamics, or contrastive reward prediction on counterfactual futures.

## All-Six Causal Imagination Pass

Date: `2026-05-19 local / 2026-05-20 UTC logs`

Goal:

```text
Convert the previous reward-blind imagination result into a stricter action-causal test.
The success metric is no longer just policy > BC. The policy gain must survive learned
dynamics counterfactuals where the same policy rollout is evaluated with zero/shuffled
dynamics actions.
```

Implemented changes:

```text
1. Causal reward contrast:
   Reward contrast now rolls out multiple future dynamics steps and compares true-action
   reward returns against zero/shuffled-action returns.

2. Causal PMPO / counterfactual advantage gate:
   Imagination training can replace or gate advantages using the return gap between
   real-action rollouts and counterfactual zero/shuffled-action rollouts.

3. Causal checkpoint selection:
   Best imagination checkpoint can be selected by causal_policy_gain, defined as
   policy_return - max(policy_return_with_zero_dynamics_actions,
                      policy_return_with_shuffled_dynamics_actions).

4. Longer reward counterfactuals:
   reward_contrast_horizon is configurable. The main run uses horizon 4; the stricter
   companion run uses horizon 8.

5. Harder action-identifiable sampling:
   Balanced reward mixture buckets now support positive_active windows, requiring both
   reward/event signal and nontrivial action magnitude.

6. Auxiliary inverse/action-effect losses:
   Agent heads now include optional inverse dynamics and action-effect prediction losses
   as diagnostics and regularizers for action-relevant latent features.
```

Validation:

```text
python3 -m py_compile sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py
bash -n sensenova_drone_agent/scripts/experiments/all_data_balanced_imagination_payload.sh \
  sensenova_drone_agent/scripts/experiments/launch_all_data_balanced_imagination.sh
```

Smoke run:

```text
run: dreamer4_all_data_balanced_imagination_causal_all6_smoke_v1
BC steps: 2
imagination updates: 2
eval batches: 1
selected metric: causal_policy_gain
selected update: 1

after policy-minus-BC:          +0.0021
after policy-minus-zero:        +0.0035
after policy-minus-dyn-zero:    +0.0022
after policy-minus-dyn-shuffle: +0.0231
after causal-policy-gain:       +0.0022
```

Long runs launched:

```text
main:
  container: sda-dreamer4-all-data-balanced-imagination-causal_all6_seed20260518_eval256_v1
  output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_causal_all6_seed20260518_eval256_v1
  GPU: 0
  BC steps: 1200
  imagination updates: 160
  reward contrast horizon: 4
  causal policy margin: 0.0

strict companion:
  container: sda-dreamer4-all-data-balanced-imagination-causal_all6_strict_seed20260518_eval256_v1
  output: sensenova_drone_agent/output/dreamer4_all_data_balanced_imagination_causal_all6_strict_seed20260518_eval256_v1
  GPU: 1
  BC steps: 1200
  imagination updates: 160
  reward contrast horizon: 8
  causal policy margin: 0.005
```

Interpretation target:

```text
If policy_minus_bc is positive but causal_policy_gain is not, the run is still reward-model
optimization rather than Dreamer-style action-causal imagination. If causal_policy_gain is
positive and stable across main/strict settings, the retrofit path has a materially stronger
claim: policy improvement depends on the action-conditioned dynamics, not just learned
reward/value shortcuts.
```

Final aggregate files:

```text
sensenova_drone_agent/output/all_data_causal_all6_eval256_v1/summary.json
sensenova_drone_agent/output/all_data_causal_all6_eval256_v1/report.md
```

Main result:

```text
run: dreamer4_all_data_balanced_imagination_causal_all6_seed20260518_eval256_v1
selected update: 160

before:
  policy-minus-BC:          +0.003492
  policy-minus-dyn-shuffle: -0.000264
  causal-policy-gain:       -0.000264

after:
  policy-minus-BC:          +0.004212
  policy-minus-dyn-shuffle: +0.000019
  causal-policy-gain:       +0.000019
```

Strict companion result:

```text
run: dreamer4_all_data_balanced_imagination_causal_all6_strict_seed20260518_eval256_v1
selected update: 0

after:
  policy-minus-BC:          -0.001627
  policy-minus-dyn-zero:    +0.005089
  policy-minus-dyn-shuffle: +0.003080
  causal-policy-gain:       +0.003080
```

Interpretation:

```text
The main run is the first all-six configuration that satisfies the weak both-positive
criterion: policy-minus-BC > 0 and causal-policy-gain > 0. The causal margin is
extremely small, so the result is positive but not yet robust.

The strict run gives cleaner causal counterfactual margins but fails to beat BC and
selects the initial checkpoint. Stricter gating preserves causal robustness but does not
yet produce imagination improvement.
```

Claim boundary update: this is no longer the previous "nothing burger" where shuffled dynamics won. The full causal patch set produces a measurable action-causal signal. However, the margin is too thin for a strong Dreamer-style claim. The next target is repeatability and margin expansion, not just another proof-of-plumbing run.

## Continued Action-Conditioned World-Model Data

We are moving from retrofit-only policy/value tricks to continued dynamics training with explicit action tokens. The current audited data collection is recorded in:

```text
sensenova_drone_agent/docs/ACTION_WORLD_MODEL_DATA_COLLECTION.md
sensenova_drone_agent/data/action_world_model_continue_v1/manifest.json
sensenova_drone_agent/data/action_world_model_continue_v1/report.md
```

Ready sources:

```text
Dreamer4-HF expert/mixed-small/mixed-large: strongest action-causality anchor
SOAR robotics task-balanced: real manipulation actions plus success/failure labels
RoboNet sample: extra robot action-video replay with zero reward placeholders
```

The collection is ready for continued dynamics pretraining with `ACTION_DIM=49`, richer action features, and `shuffle,zero,time_shift` action contrast. Claim boundary remains unchanged until the continued dynamics run passes strict held-out causal gates.

Smoke validation:

```text
run: dreamer4_all_data_native_continued_action_wm_train_smoke_v1
global steps: 25000 -> 25002
valid training windows: 3669335 across 109 tasks
direct shuffle/normal: 1.0739
direct zero/normal: 1.0512
autoregressive shuffle/normal: 1.3050
autoregressive zero/normal: 1.2226
normal/persistence: 0.7059
strict gate passed: true
```

This confirms the continued action-conditioned dynamics path can consume the collected corpus and still exposes counterfactual action sensitivity after a tiny continuation. The next run should be a real multi-GPU continuation, not another smoke.

Additional real-action data expansion is now in progress:

```text
script: sensenova_drone_agent/scripts/download_robot_action_hf_datasets.py
profile: oxe-compact
mode: paired video parquets only
target remote size: 17.51 GiB
sources: compact DROID, Fractal/Google Robot, BridgeData-style LeRobot mirrors
pid: sensenova_drone_agent/logs/data_downloads/oxe_compact_hf_download.pid
log: sensenova_drone_agent/logs/data_downloads/oxe_compact_hf_download.log
auth: HF_TOKEN environment variable
retry policy: --repo-retries 20 --retry-sleep-s 90
```

These sources still need conversion via:

```text
sensenova_drone_agent/scripts/export_lerobot_hf_dreamer4_dataset.py
```
