# SOAR Retrofit Dynamics Ablations

Date: 2026-05-13

This note records the current retrofit-dynamics evidence for frozen Kairos/Sensenova SOAR latents.
The question was whether we can improve the external Dreamer-style dynamics adapter enough to support longer imagination rollouts without changing the Kairos world model itself.

## Current Best Baseline

Reference run:

```text
sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1
```

Reference checkpoint:

```text
sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/best_imagination.pt
```

This remains the best current retrofit path.

| Eval | normal/persistence | shuffled/normal | zero/normal | strict gate |
|---|---:|---:|---:|---|
| single-pass | 0.852 | 1.092 | 1.628 | pass |
| autoregressive h4 | 0.922 | 1.058 | 2.668 | pass |
| autoregressive h8 | 0.947 | 1.091 | 6.197 | pass |
| autoregressive h16 | 1.061 | 1.087 | 19.445 | fail |

Interpretation: the current frozen-Kairos retrofit dynamics support short-horizon imagination/replanning at h4/h8. They do not support h16 open-loop claims.

## Tested Knobs

All runs used the SOAR Kairos stride8/summed-action cache unless noted:

```text
sensenova_drone_agent/output/soar_sequence_cache_kairos_task_balanced_512_stride8_sum/soar_kairos_flat128_512traj320_stride8_sum_trajectory_success.npz
```

### Naive Autoregressive Rollout MSE

Runs:

```text
sensenova_drone_agent/output/soar_retrofit_rollout_h8_w05_ctx8_v1
sensenova_drone_agent/output/soar_retrofit_rollout_h16_w05_ctx8_v1
```

Result: negative. The model became conservative and action contrast largely collapsed.

| Run | h4 strict | h8 strict | h16 strict | Key failure |
|---|---|---|---|---|
| rollout MSE h8 | fail | fail | fail | normal/persistence near 1.0, weak shuffle/zero separation |
| rollout MSE h16 | fail | fail | fail | normal/persistence above or near 1.0, weak shuffle separation |

Conclusion: direct multi-step MSE encourages persistence-like predictions more than controllable action-conditioned dynamics.

### Longer Direct Prediction Horizon

Runs:

```text
sensenova_drone_agent/output/soar_retrofit_pred8_ctx8_contrastive_v1
sensenova_drone_agent/output/soar_retrofit_pred8_ctx16_contrastive_v1
```

Result: negative.

| Run | direct normal/persistence | direct shuffled/normal | direct zero/normal | strict gate |
|---|---:|---:|---:|---|
| pred8 ctx8 | 0.990 | 1.021 | 1.234 | fail |
| pred8 ctx16 | 1.005 | 1.000 | 1.000 | fail |

Conclusion: increasing direct prediction horizon and context length does not extend action grounding by itself. Context 16 was worse than context 8 for this adapter.

### Autoregressive Rollout Contrast

Runs:

```text
sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h8_w05_ctx8_v1
sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h16_w05_ctx8_v1
sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h8_w01_conservative_ctx8_v1
```

Result: negative.

| Run | h4 strict | h8 strict | h16 strict | Key failure |
|---|---|---|---|---|
| h8 contrast w0.5 | fail | fail | fail | high-LR rollout contrast weakened short-horizon grounding |
| h16 contrast w0.5 | fail | fail | fail | severe autoregressive drift, no useful shuffle separation |
| h8 contrast w0.1 conservative | fail | fail | fail | selected rollout-MSE checkpoint still collapsed toward persistence |

The conservative run used:

```text
learning_rate=5e-5
weight_decay=0.01
dropout=0.25
dynamics_bc_metric=autoregressive_rollout_mse
dynamics_bc_early_stop_patience=12
```

It improved checkpoint selection mechanics but not the dynamics gate.

## Code Changes From This Ablation Pass

Trainer:

```text
sensenova_drone_agent/scripts/train_soar_dreamer_lite.py
```

Added rollout-training knobs:

```text
--dynamics-rollout-loss-weight
--dynamics-rollout-contrastive-loss-weight
--dynamics-rollout-contrastive-margin
--dynamics-rollout-horizon
```

Added checkpoint-selection knobs:

```text
--dynamics-bc-metric
--dynamics-bc-early-stop-patience
--dynamics-bc-min-delta
```

Evaluator:

```text
sensenova_drone_agent/scripts/eval_soar_learned_dynamics.py
```

This evaluator is now the gate for direct and autoregressive dynamics quality.

## Current Conclusion

The retrofit path is real but short-horizon:

```text
frozen Kairos/Sensenova latents + external dynamics adapter + short-horizon action conditioning works at h4/h8
```

The following do not currently improve the retrofit path:

```text
naive rollout MSE
longer direct prediction horizon
longer context length
aggressive rollout contrast
weak conservative rollout contrast selected by rollout MSE
```

The strongest practical path remains:

```text
use h4/h8 short-horizon model-based replanning
avoid h16 open-loop imagination claims
keep value head frozen during near-term imagination policy optimization
```

## Recommended Next Knobs

1. Add a gate-aware checkpoint evaluator that selects checkpoints by strict control metrics, not just scalar validation losses.
2. Try action-lag and action-window alignment sweeps under the current h4/h8 setup before changing capacity.
3. Try curriculum dynamics training: first direct h4 action contrast, then h8 autoregressive eval only, not h8 rollout loss.
4. Add an action-gated residual dynamics adapter where actions modulate a residual over persistence instead of predicting the full latent delta.
5. Treat h16+ imagination as requiring native action-token dynamics or continued world-model training, not just external adapter tuning.

## 2026-05-14 Alignment, Residual, Action-Token, and Native-WM Preflight

### New Retrofit Runs

All runs used:

```text
context_len=8
prediction_horizon=4
mtp_horizon=8
hidden_dim=256
num_layers=2
num_heads=4
dropout=0.25
motion_filter_quantile=0.25
learning_rate=5e-5
weight_decay=0.01
dynamics_bc_metric=dynamics_mse
dynamics_bc_early_stop_patience=15
```

Runs:

```text
sensenova_drone_agent/output/soar_retrofit_align_offset_m1_window1_v1
sensenova_drone_agent/output/soar_retrofit_align_offset_p1_window1_v1
sensenova_drone_agent/output/soar_retrofit_align_offset0_window2_mean_v1
sensenova_drone_agent/output/soar_retrofit_action_gated_residual_ctx8_v1
sensenova_drone_agent/output/soar_retrofit_action_query_tokens_ctx8_v1
```

Built-in single-pass gate:

| Run | normal/persistence | shuffled/normal | zero/normal | strength | strict gate |
|---|---:|---:|---:|---|---|
| offset -1 | 1.000 | 1.000 | 1.000 | none | fail |
| offset +1 | 1.000 | 1.000 | 1.000 | none | fail |
| window 2 mean | 1.000 | 1.000 | 1.000 | none | fail |
| action-gated residual | 0.997 | 1.003 | 1.002 | weak | fail |
| action-query tokens | 1.000 | 1.000 | 1.000 | none | fail |

Autoregressive eval:

| Run | h4 normal/persistence | h4 shuffled/normal | h8 normal/persistence | h8 shuffled/normal | h16 normal/persistence | h16 shuffled/normal |
|---|---:|---:|---:|---:|---:|---:|
| offset -1 | 1.003 | 1.000 | 1.009 | 1.000 | 1.043 | 0.998 |
| offset +1 | 1.010 | 1.000 | 1.032 | 1.000 | 1.100 | 0.998 |
| window 2 mean | 1.013 | 1.000 | 1.036 | 0.999 | 1.107 | 0.999 |
| action-gated residual | 1.016 | 1.002 | 1.048 | 1.005 | 1.123 | 1.008 |
| action-query tokens | 1.001 | 1.000 | 1.005 | 1.000 | 1.018 | 1.000 |

Interpretation:

```text
alignment/window sweeps did not reveal a simple action-frame lag bug.
action-query tokens alone did not improve action grounding.
action-gated residual created weak action sensitivity, but its autoregressive normal rollouts were worse than persistence.
none of these runs beat the existing h4/h8 baseline.
```

The current best remains:

```text
sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1
```

Reference comparison:

| Eval | normal/persistence | shuffled/normal | zero/normal | strict gate |
|---|---:|---:|---:|---|
| baseline h4 | 0.922 | 1.058 | 2.668 | pass |
| baseline h8 | 0.947 | 1.091 | 6.197 | pass |
| baseline h16 | 1.061 | 1.087 | 19.445 | fail |

### Native Dreamer4-Style SOAR World-Model Preflight

Added:

```text
sensenova_drone_agent/scripts/export_soar_dreamer4_dataset.py
```

Patched the local unofficial Dreamer4 reproduction to support local SOAR smoke tests:

```text
dreamer4/dreamer4/sharded_frame_dataset.py
dreamer4/dreamer4/train_tokenizer.py
dreamer4/dreamer4/train_dynamics.py
```

Changes:

```text
--tasks_from_data
--tasks
--wandb_mode
optional no-op wandb fallback
clean max_steps exit in tokenizer/dynamics training loops
```

Smoke dataset:

```text
sensenova_drone_agent/data/robotics/soar/dreamer4_soar_smoke
```

Smoke outputs:

```text
sensenova_drone_agent/output/dreamer4_soar_native_smoke_v2/tokenizer_ckpts/latest.pt
sensenova_drone_agent/output/dreamer4_soar_native_smoke_v2/dynamics_ckpts/latest.pt
```

Smoke result:

```text
SOAR ZIP -> Dreamer4 raw task .pt files + frame shards works.
Tiny tokenizer training ran for 2 steps.
Tiny action-conditioned Dreamer4 dynamics training ran for 2 steps.
This validates plumbing only; it is not a meaningful world-model result.
```

Native world-model training is now possible in principle, but the real run still requires:

```text
full or large SOAR conversion
proper tokenizer training or a valid pretrained tokenizer checkpoint
long action-conditioned dynamics training
held-out action-shuffle/zero-action eval
```

The README checkpoint names for the cloned Dreamer4 repo were not present at the advertised Hugging Face paths during this check, so the smoke used a tiny local tokenizer instead of a pretrained Dreamer4 checkpoint.

## 2026-05-14 Native SOAR Dreamer4 Run v1

Run:

```text
sensenova_drone_agent/output/dreamer4_soar_native_v1
```

Dataset:

```text
sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v1
```

Dataset summary:

| Field | Value |
|---|---:|
| selected trajectories | 127 |
| tasks | 16 |
| exported steps | 6066 |
| success episodes | 58 |
| failure episodes | 69 |
| frame size | 128 |
| frame stride | 2 |

Training:

| Component | Steps | Model |
|---|---:|---|
| tokenizer | 3000 | d_model=128, depth=3, n_latents=16, d_bottleneck=32, patch=8 |
| dynamics | 5000 | d_model=128, depth=3, k_max=8, self_fraction=0.25, bootstrap_start=1000 |

Artifacts:

```text
sensenova_drone_agent/output/dreamer4_soar_native_v1/tokenizer_ckpts/latest.pt
sensenova_drone_agent/output/dreamer4_soar_native_v1/tokenizer_ckpts/final_step_0003000.pt
sensenova_drone_agent/output/dreamer4_soar_native_v1/dynamics_ckpts/latest.pt
sensenova_drone_agent/output/dreamer4_soar_native_v1/dynamics_ckpts/final_step_0005000.pt
sensenova_drone_agent/output/dreamer4_soar_native_v1/native_dynamics_eval_h8.json
```

Native dynamics eval at ctx8/h8:

| Metric | Value |
|---|---:|
| direct normal MSE | 0.0233 |
| direct shuffled/normal | 1.0004 |
| direct zero/normal | 0.9996 |
| autoregressive normal MSE | 0.0622 |
| autoregressive persistence MSE | 0.3016 |
| autoregressive normal/persistence | 0.2061 |
| autoregressive shuffled/normal | 1.0014 |
| autoregressive zero/normal | 0.9730 |

Decision:

```text
native dynamics learned visual latent dynamics: yes
native dynamics beats persistence: yes
native dynamics learned measurable action grounding: no
native dynamics ready for imagination RL: no
```

Interpretation:

The native Dreamer4-style run is now operational and learns future latent prediction from pixels.
However, the h8 control eval shows almost no difference between normal and shuffled actions, and zero actions are slightly better than normal actions.
This means the first native run learned a scene-motion prior more than controllable action-conditioned dynamics.

Most likely next fixes:

```text
increase action-paired data substantially
train dynamics longer after tokenizer stabilizes
increase action salience via action dropout/zero-vs-normal contrast
evaluate/repair action-frame alignment in the native WMDataset path
use a stronger tokenizer or train it longer before dynamics
```

## 2026-05-14 Native SOAR Dreamer4 Run v2 Action-Contrast

Purpose:

```text
Convert the v1 scene-motion prior into a controllable action-conditioned simulator by increasing
paired SOAR data scale and adding explicit pressure for normal actions to outperform shuffled and
zeroed actions.
```

Implementation changes:

```text
dreamer4/dreamer4/train_dynamics.py
  - action_frame_offset controls action-to-frame alignment
  - action_contrast_weight enables normal-vs-shuffled/zero action contrast
  - action_contrast_margin controls the required MSE gap
  - action_contrast_signal controls the corruption level used for the contrast pass
  - resume_reset_optim allows continued training from weights with a fresh optimizer/LR

dreamer4/dreamer4/train_tokenizer.py
  - resume_reset_optim allows continued tokenizer training with a fresh optimizer/LR

sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py
  - action-frame-offset evaluates timestamp alignment hypotheses
```

Run:

```text
sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast
```

Dataset:

```text
sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast
```

Launcher:

```bash
sensenova_drone_agent/scripts/experiments/launch_soar_native_action_contrast_v2.sh
```

Monitoring:

```bash
docker logs -f soar_dreamer4_native_v2_action_contrast
tail -f sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast/native_run.log
```

Decision gate:

```text
Pass requires normal actions to beat shuffled/zero actions and autoregressive rollout to beat persistence.
Failure means the system remains a scene-motion prior, not yet a controllable simulator.
```

Result:

| offset | h8 normal/persistence | h8 shuffled/normal | h8 zero/normal | ready |
|---:|---:|---:|---:|---|
| -2 | 0.1859 | 1.0019 | 0.9633 | false |
| -1 | 0.1695 | 0.9958 | 0.9688 | false |
| 0 | 0.1674 | 0.9951 | 0.9675 | false |
| 1 | 0.1793 | 0.9950 | 0.9629 | false |
| 2 | 0.1605 | 0.9994 | 0.9723 | false |

Conclusion:

```text
V2 retains strong latent future prediction but still fails action identity grounding on held-out eval.
The training-time contrast objective made zero-action metrics move in logs, but this did not translate
to a reliable eval-time normal-vs-shuffled/zero action gap.
```

Do not proceed to imagination RL from this checkpoint. The next pass should make action identity unavoidable, for example by predicting action-conditioned deltas, using contrastive pairs in-batch with identical observation contexts, or adding a dedicated action-token/policy-token architecture rather than relying on a single action embedding next to a strong visual prior.

## Paper-Level Interpretation

This supports a cautious claim:

```text
Frozen video-world-model features can support a lightweight short-horizon action-conditioned control stack.
```

It does not support:

```text
The frozen Kairos/Sensenova generator has been converted into a long-horizon Dreamer-style world simulator.
```

For Dreamer-like h16+ imagination, the missing ingredient is still native action-conditioned dynamics, not merely more rollout loss on top of frozen latents.
