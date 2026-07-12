# Game Action Dreamer4 Data

Purpose: create action-identifiable pixel/action sequences for native Dreamer4-style tokenizer and dynamics training.

This data is intentionally different from SOAR/RoboNet. The game environments expose short-horizon counterfactual structure: different discrete actions should produce visibly different futures from the same recent scene context.

## Collector

Script:

```bash
sensenova_drone_agent/scripts/collect_game_action_dreamer4_dataset.py
```

Docker wrapper:

```bash
./sensenova_drone_agent/scripts/run_game_action_collector.sh
```

The collector writes the existing native `WMDataset` layout:

```text
<out>/
  raw/<task>.pt
  frames/<task>/<task>_shard0000.pt
  tasks.json
  summary.json
  report.md
  previews/<task>.png
```

Each `raw/<task>.pt` contains:

```text
episode: int64 [N]
action:  float32 [N, action_dim]
reward:  float32 [N]
```

The first row of each episode has a zero action/reward. Row `t+1` stores the action and reward for the transition from frame `t` to frame `t+1`, matching `WMDataset` alignment.

The collector supports two policies:

```text
random        sample a fresh random action each step
action_blocks sample one random action and repeat it for --action-block-steps frames
```

For action-grounding experiments, prefer `action_blocks` because repeated actions make causal consequences easier to identify than one-step random twitching.

## Current V1 Batch

Path:

```text
sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1
```

Command:

```bash
./sensenova_drone_agent/scripts/run_game_action_collector.sh \
  --source all \
  --episodes 16 \
  --max-steps 128 \
  --procgen-envs coinrun,bigfish,jumper \
  --vizdoom-scenarios basic \
  --out sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1 \
  --overwrite \
  --validate \
  --validate-seq-len 16
```

Result:

```text
tasks: 4
total frames: 6357
valid WMDataset windows at seq_len=16: 5368
frame size: 128x128
max action_dim: 15
```

## Stronger Action-Grounding Variant

For a stronger action signal:

```bash
./sensenova_drone_agent/scripts/run_game_action_collector.sh \
  --source all \
  --episodes 16 \
  --max-steps 128 \
  --policy action_blocks \
  --action-block-steps 8 \
  --procgen-envs coinrun,bigfish,jumper \
  --vizdoom-scenarios basic \
  --out sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1 \
  --overwrite \
  --validate \
  --validate-seq-len 16
```

Result:

```text
path: sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1
tasks: 4
total frames: 6531
valid WMDataset windows at seq_len=16: 5535
frame size: 128x128
max action_dim: 15
```

## Initial Probe Result

The existing linear action-identifiability probe was run on both game batches:

```text
sensenova_drone_agent/output/action_identifiability_audit_game_actions_v1
sensenova_drone_agent/output/action_identifiability_audit_game_actions_blocks_v1
```

Both passed dataset loading and action-variance checks, but the simple ridge probe did not mark the data as action-identifiable. Treat this as a probe limitation rather than a final verdict: it mixes tasks with different action semantics and uses a low-capacity linear predictor over downsampled pixel deltas. The next meaningful test is native dynamics training with normal vs shuffled/zero action evaluation.

## Native Dynamics Result

Run:

```bash
./sensenova_drone_agent/scripts/experiments/launch_game_actions_native_dreamer4.sh
```

Output:

```text
sensenova_drone_agent/output/dreamer4_game_actions_native_blocks_v1
```

Configuration:

```text
dataset: dreamer4_game_actions_blocks_v1
tokenizer: copied from dreamer4_all_data_native_v1
dynamics steps: 20000
GPUs: 2
action_features: current,prev,delta,mean4,norm
expanded action_dim: 61
action_contrast_weight: 1.0
action_contrast_start: 1000
```

Evaluation:

```text
sensenova_drone_agent/output/dreamer4_game_actions_native_blocks_v1/native_dynamics_eval_h8_game_actions.json
```

Result:

```json
{
  "direct_action_conditioning_detected": true,
  "autoregressive_action_conditioning_detected": true,
  "autoregressive_beats_persistence": true,
  "native_dynamics_ready_for_imagination": true
}
```

Metrics:

```text
direct shuffle_over_normal: 1.2077
direct zero_over_normal:    1.0905
autoregressive normal_over_persistence: 0.8082
autoregressive shuffle_over_normal:     1.1573
autoregressive zero_over_normal:        1.2363
```

Interpretation: this is the first local native Dreamer4-style run where the learned dynamics both beat persistence and measurably depend on action tokens in autoregressive rollout evaluation. This does not yet prove a useful policy, but it clears the dynamics action-conditioning gate for a small game-action domain.

Tasks:

```text
procgen-bigfish-random
procgen-coinrun-random
procgen-jumper-random
vizdoom-basic-random
```

## Native Imagination Result

Run:

```bash
RUN_ID=blocks_v1 \
BC_STEPS=1200 \
IMAGINATION_UPDATES=400 \
EVAL_BATCHES=64 \
LOG_STD_INIT=-2.5 \
./sensenova_drone_agent/scripts/experiments/launch_game_actions_native_imagination.sh
```

Output:

```text
sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_v1
```

Smoke comparison:

```text
default log_std_init=-1.0:
  policy_return_delta: +0.0036
  after policy-minus-BC: -0.0008
  after policy-minus-zero: -0.0027

lower log_std_init=-2.5:
  policy_return_delta: +0.0048
  after policy-minus-BC: +0.0004
  after policy-minus-zero: -0.0015
```

Full result:

```text
BC steps: 1200
imagination updates: 400
held-out eval windows: 800
balanced eval samples: 256
policy_return_delta: +0.0274
after policy-minus-BC: +0.0004
after policy-minus-zero: -0.0234
policy_prior_mse_delta: -0.0018
```

Decision:

```text
native dynamics action-conditioning: pass
BC/reward/value + imagination plumbing: pass
policy improves over pre-imagination policy: pass
policy beats BC prior: weak pass
policy beats zero-action baseline: fail
```

Interpretation: the Dreamer-style loop is now operational on a small action-conditioned game dataset. It can train policy/reward/value heads and apply constrained imagination updates without drifting away from the BC prior. The remaining blocker is reward/task quality: under the learned reward, zero-action still scores higher than the post-imagination policy, so this is not yet a strong imagination-training success claim.

Next gate: collect or curate tasks with clearer positive progress signals, then re-run the same native dynamics and imagination pipeline. The model needs data where useful actions visibly change future state and are rewarded more than no-op behavior.

## Raw-Action Policy Retrofit

Dream-VLX/Dream-VLA highlighted that policy heads should predict executable raw action chunks, not handcrafted expanded dynamics features. We added that split on `2026-05-18`.

The dynamics model can still use:

```text
current,prev,delta,mean4,norm
```

but the policy can now use:

```text
--policy-action-source raw
--raw-action-dim 15
--action-chunk-len 4
```

Smoke run:

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

Decision: the raw-policy/chunk-action plumbing works. This does not supersede the earlier full result yet; it only clears the implementation gate for a longer raw-action run.

Full raw-policy run:

```text
sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_v1
```

Result:

```text
action_mse step 1:    0.1618
action_mse step 1200: 0.0489
before policy-minus-BC:   +0.0071
after policy-minus-BC:    -0.0009
after policy-minus-zero:  -0.0093
policy_return_delta:      -0.0080
policy_prior_mse_delta:   -0.0048
```

Interpretation: raw-action chunk BC trains, and the policy stays tightly prior-constrained, but imagination still hurts held-out learned return. The remaining blocker is not just policy output representation; it is reward/advantage quality or task reward design.

No-update control:

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

Interpretation: the raw-policy eval path is stable. The full-run degradation comes from the imagination update itself, not from nondeterministic before/after evaluation.

## Reward-Signal Filtering

Added on `2026-05-18` to match the lesson from Dream-VLX/Dream-VLA: training should emphasize successful or clearly rewarded demonstrations, not random background windows.

Supported native dataset/training options:

```text
--reward-filter-mode none|positive_sum|abs_sum|any_positive|any_abs
--reward-signal-threshold 0.0
--min-reward-signal-steps 1
```

Initial filtered run:

```text
reward_filter_mode: any_positive
output: sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_reward_any_positive_v1
control: sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_raw_policy_reward_any_positive_no_update_v1
```

Filtered window counts:

```text
procgen-bigfish-random: 154
procgen-coinrun-random: 0
procgen-jumper-random: 1
vizdoom-basic-random: 8
total: 163
train: 145
eval: 18
```

Result:

```text
BC action_mse step 1200: 0.0355

no-update policy-minus-BC:    +0.0533
no-update policy-minus-zero:  +0.4758

after-RL policy-minus-BC:     +0.0066
after-RL policy-minus-zero:   +0.4292
policy_return_delta from RL:  -0.0466
```

Decision: reward filtering gives a real positive BC/eval signal; imagination training still harms the already-good BC policy on this tiny filtered subset. This is evidence that reward/task curation matters more than further tuning the current RL update. The current dataset is too small and too skewed toward BigFish to support a strong claim.

## Training Hook

Use these paths for native Dreamer4 training:

```text
--data_dirs  sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1/raw
--frame_dirs sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1/frames
--tasks_json sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1/tasks.json
--action_dim 15
```

For an action-conditioning test, train dynamics with `--use_actions`, then evaluate normal actions against shuffled and zero actions. This batch should be a stronger action-grounding sanity check than SOAR/RoboNet because actions are expected to causally affect near-future pixels.

## Notes

- Procgen emits one-hot discrete actions with 15 choices.
- ViZDoom emits one-hot action-choice vectors over noop plus available single-button actions.
- Large sources such as CARLA, MineRL, and Habitat are intentionally not pulled until the collector/training/eval loop is proven on lightweight sources.
