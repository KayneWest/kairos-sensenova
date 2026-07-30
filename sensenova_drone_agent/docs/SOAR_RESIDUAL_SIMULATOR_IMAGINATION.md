# SOAR Residual Simulator Imagination Track

Status: active as of 2026-05-27.

## Current Simulator Artifact

The current frozen controllable simulator artifact is:

`sensenova_drone_agent/output/controllable_soar_simulator_v1/manifest.json`

It combines:

- Kairos/Sensenova-style visual pretraining via the tokenizer checkpoint.
- Continued Dreamer4-style dynamics from the source-weighted SOAR+DROID run.
- A residual action adapter selected from held-out causal controls.

Selected adapter:

`sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt`

Selection summary:

`sensenova_drone_agent/output/residual_action_adapter_selection_v1/selection_summary.json`

Key held-out metrics:

- AR normal/persistence: `0.5390`
- AR effect-far-shuffle/normal: `1.0252`
- Direct effect-far-shuffle/normal: `1.0030`
- AR zero/normal: `22.4219`
- AR temporal-min/normal: `29.7049`

Interpretation: this is the best current retrofit simulator. It passes the AR rollout causal gate and preserves rollout quality, but direct one-step cross-trajectory action identity is still weak.

## Implemented Steps

- Checkpoint selection: `select_residual_adapter_checkpoint.py` ranks completed residual adapters and writes a stable selection summary.
- Simulator promotion: `promote_controllable_soar_simulator.py` writes `controllable_soar_simulator_v1` with component paths and claim boundary.
- Runtime wrapper: `residual_adapter_runtime.py` loads the adapter and wraps frozen dynamics for downstream imagination training.
- Midtraining/imagination integration: `train_native_dreamer4_imagination.py` accepts `--residual-adapter-ckpt` and keeps dynamics frozen while training BC/reward/value/policy heads.
- Launch path: `launch_soar_residual_adapter_imagination.sh` runs SOAR+DROID only, not drone/game data.

## Active Runs

Smoke test:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_smoke_v1`

Result: completed end-to-end.

Long run:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_v1`

Configuration:

- BC steps: `2400`
- Imagination updates: `800`
- Batch size: `4`
- Eval batches: `64`
- Data: SOAR native contrast + DROID LeRobot export
- Simulator: frozen dynamics + residual adapter
- Causal policy mode: `advantage_gate`
- Reward contrast: enabled with zero/shuffle negatives

Residual adapter eval grid:

`sensenova_drone_agent/output/residual_adapter_eval_grid_v2`

Purpose: compare `effect`, `far`, and `random` adapters at horizon 16, plus per-source and selected per-task probes for the winning adapter.

Grid result snapshot:

| Eval | AR/Persist | AR Cross | Direct Cross | AR Zero | Interpretation |
|---|---:|---:|---:|---:|---|
| `effect_all_h16` | 0.475 | 1.092 | 0.995 | 30.6 | Strong global horizon-16 result |
| `far_all_h16` | 0.596 | 1.021 | 1.013 | 30.2 | Passes but weaker than effect globally |
| `random_all_h16` | 0.616 | 0.993 | 1.017 | 14.8 | Does not pass AR cross-action gate |
| `effect_droid_h16` | 0.548 | 0.986 | 0.981 | 22.7 | Source-level DROID cross gate weak |
| `effect_task_droid_h16` | 0.652 | 1.037 | 1.007 | 18.2 | DROID task-level cross gate passes |
| `effect_soar_h16` | 0.572 | 0.998 | 1.003 | 29.1 | Source-level SOAR cross gate weak |
| `effect_task_open_drawer_h16` | 1.376 | 0.980 | 1.001 | 19.8 | Fails persistence and cross gates |
| `effect_task_red_cloth_h16` | 0.688 | 0.984 | 0.996 | 82.4 | Beats persistence; cross gate weak |

## Causal Imagination Results

Follow-up runs used `BEST_IMAGINATION_METRIC=policy_minus_dyn_shuffle`,
`CAUSAL_POLICY_MODE=advantage_gate`, `REWARD_CONTRAST_WEIGHT=2.0`,
`REWARD_CONTRAST_HORIZON=4`, `BC_STEPS=2400`, and frozen dynamics plus the selected
residual adapter.

| Run | Train balance | Selected update | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Reading |
|---|---|---:|---:|---:|---:|---:|---|
| `soar_residual_adapter_imagination_causal_v1` | SOAR positive 0.35, SOAR active 0.25, DROID active 0.40 | 100 | +0.0012 | +0.1595 | +0.0985 | +0.0008 | First positive but very small shuffle-causal margin |
| `soar_residual_adapter_imagination_causal_seed_20260528` | SOAR positive 0.35, SOAR active 0.25, DROID active 0.40 | 0 | -0.0016 | +0.1578 | +0.1112 | +0.0045 | Repeat seed shows stronger causal margin, but selected checkpoint does not beat BC |
| `soar_residual_adapter_imagination_causal_droidheavy_v1` | DROID active 0.80, SOAR positive 0.10, SOAR active 0.10 | 100 | -0.0021 | +0.1387 | +0.1087 | +0.0092 | DROID-heavy weighting substantially strengthens action-causal margin |

Important non-selected DROID-heavy checkpoints:

| Update | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Reading |
|---:|---:|---:|---:|---:|---|
| 200 | +0.0062 | +0.1471 | +0.1153 | +0.0043 | Beats BC and keeps positive causal margin |
| 300 | +0.0086 | +0.1494 | +0.1182 | +0.0044 | Better BC gain with stable causal margin |
| 600 | +0.0105 | +0.1514 | +0.1199 | +0.0040 | Stronger BC gain, causal margin still positive |
| 800 | +0.0180 | +0.1589 | +0.1271 | +0.0026 | Best BC gain, but causal margin decays |

Interpretation:

- The weak `causal_v1` signal was not a one-off; repeat runs keep `policy_minus_dyn_shuffle` positive.
- DROID-heavy/source weighting makes the causal gate much stronger, supporting the hypothesis that action-identifiable data matters.
- Selecting only by `policy_minus_dyn_shuffle` is too blunt: it can choose an early checkpoint that maximizes causal margin but fails to beat BC.
- Next selection should use a composite gate: require `policy_minus_bc > 0`, require `policy_minus_dyn_shuffle > 0`, then maximize either `policy_minus_bc` or a weighted combination of BC gain and causal margin.

Composite metric follow-up:

The trainer now supports `BEST_IMAGINATION_METRIC=policy_minus_bc_plus_dyn_shuffle`,
implemented as `policy_minus_bc + policy_minus_dyn_shuffle`.
It also supports `BEST_IMAGINATION_METRIC=policy_minus_bc_causal_gate`, which requires
`policy_minus_bc > 0` and `policy_minus_dyn_shuffle >= CAUSAL_POLICY_MIN_MARGIN`, then
selects by `policy_minus_bc`.

| Run | Train balance | Selected update | Metric | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `soar_residual_adapter_imagination_causal_composite_mixed_v1` | SOAR positive 0.35, SOAR active 0.25, DROID active 0.40 | 300 | +0.0263 | +0.0146 | +0.1165 | +0.1118 | +0.0117 | Best current mixed-data result: improves over BC and keeps a nontrivial causal margin |
| `soar_residual_adapter_imagination_causal_composite_droidheavy_v1` | DROID active 0.80, SOAR positive 0.10, SOAR active 0.10 | 600 | +0.0158 | +0.0079 | +0.1048 | +0.0826 | +0.0079 | Best current DROID-heavy result: cleaner than pure causal selection, with both BC and shuffle gates positive |

Interpretation update:

- Composite selection fixed the immediate checkpoint-selection problem.
- The strongest current result is the mixed composite run: `policy_minus_bc=+0.0146` and `policy_minus_dyn_shuffle=+0.0117`.
- DROID-heavy improves causal identifiability but can reduce absolute learned-return scale and requires composite selection to avoid early negative-BC checkpoints.
- The next strict selection rule should be a hard gate rather than only a sum: require `policy_minus_bc > 0` and `policy_minus_dyn_shuffle >= 0.002`, then maximize `policy_minus_bc`.

## Hard-Gated Policy Artifact

Hard-gated follow-up used `BEST_IMAGINATION_METRIC=policy_minus_bc_causal_gate`
with `CAUSAL_POLICY_MIN_MARGIN=0.002`. The selection rule rejects checkpoints unless
they beat BC and clear the dynamics-shuffle margin, then maximizes `policy_minus_bc`.

| Run | Train balance | Selected update | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Reading |
|---|---|---:|---:|---:|---:|---:|---|
| `soar_residual_adapter_imagination_causal_hardgate_mixed_v1` | SOAR positive 0.35, SOAR active 0.25, DROID active 0.40 | 300 | +0.0059 | +0.1074 | +0.0728 | +0.0003 | Beats BC, but fails the strict `0.002` shuffle gate |
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_v1` | DROID active 0.80, SOAR positive 0.10, SOAR active 0.10 | 500 | +0.0088 | +0.1661 | +0.1177 | +0.0021 | Current promoted policy: beats BC and clears hard causal gate |

The promoted imagination policy artifact is:

`sensenova_drone_agent/output/controllable_soar_imagination_policy_v1/manifest.json`

It points to:

- Policy checkpoint: `soar_residual_adapter_imagination_causal_hardgate_droidheavy_v1/after_imagination.pt`
- BC prior checkpoint: `soar_residual_adapter_imagination_causal_hardgate_droidheavy_v1/bc_prior.pt`
- Frozen tokenizer, continued dynamics, and selected residual action adapter.

## Repeatability

DROID-heavy hard-gated repeatability used three independent seeds:
`20260605`, `20260606`, and `20260607`.

| Seed run | Selected update | Policy - BC | Policy - dyn-shuffle | Pass |
|---|---:|---:|---:|---|
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_seed_20260605` | 600 | +0.0043 | +0.0023 | yes |
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_seed_20260606` | 200 | +0.0025 | +0.0061 | yes |
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_seed_20260607` | 0 | +0.0087 | +0.0194 | yes |

Repeatability aggregate:

- Pass count: `3/3`
- Mean `policy_minus_bc`: `+0.0052`
- Mean `policy_minus_dyn_shuffle`: `+0.0093`

Note: selected update `0` denotes the post-BC/pre-imagination checkpoint. For seed
`20260607`, the post-BC checkpoint already passed the hard gate and later imagination
checkpoints also remained gate-positive with slightly lower `policy_minus_bc`.

## Breakdown Eval

Breakdown eval for the promoted DROID-heavy artifact:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_causal_hardgate_droidheavy_v1/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0045 | +0.0615 | +0.0383 | +0.0007 | +0.0020 | +0.0007 |
| all | 8 | +0.0088 | +0.1661 | +0.1177 | +0.0021 | +0.0100 | +0.0022 |
| all | 16 | +0.0403 | +0.2997 | +0.2573 | +0.0107 | +0.0285 | +0.0085 |
| soar | 4 | +0.0040 | +0.0572 | +0.0364 | +0.0007 | +0.0025 | +0.0003 |
| soar | 8 | +0.0086 | +0.1592 | +0.1141 | +0.0011 | +0.0119 | +0.0034 |
| soar | 16 | +0.0381 | +0.2924 | +0.2541 | +0.0123 | +0.0331 | +0.0104 |
| droid | 4 | +0.0025 | +0.0404 | +0.0288 | -0.0000 | -0.0122 | -0.0017 |
| droid | 8 | +0.0050 | +0.1073 | +0.0865 | +0.0020 | -0.0274 | -0.0029 |
| droid | 16 | +0.0316 | +0.1755 | +0.1713 | -0.0043 | -0.0714 | -0.0215 |

Interpretation:

- The all-source eval passes the hard gate at horizon 8 and strengthens at horizon 16.
- SOAR-only strengthens clearly at horizon 16.
- DROID-only beats BC and zero-action controls, but time-shift and far-shuffle controls can beat the learned policy. This means exact action timing is not uniformly proven across all source slices.
- Paper wording should say the learned-simulator policy improves over BC and zero/shuffle controls under the headline all-source gate, with source-level caveats. It should not claim solved action causality across every source/control.

Paper-ready tables:

`sensenova_drone_agent/paper/soar_imagination_results.md`

## Post-RL-Only Selection

To separate behavior-cloning strength from actual imagination-training improvement, the trainer now
supports `MIN_IMAGINATION_SELECTION_UPDATE`. With `MIN_IMAGINATION_SELECTION_UPDATE=100`,
the update `0` post-BC checkpoint remains in diagnostic history but cannot be selected.

Implementation:

- `train_native_dreamer4_imagination.py` records `eligible` for every selection eval row.
- `best_imagination_selection.json` records `min_selection_update`.
- `launch_soar_residual_adapter_imagination.sh` and its payload forward `MIN_IMAGINATION_SELECTION_UPDATE`.

Post-RL-only DROID-heavy repeatability used:

- `BEST_IMAGINATION_METRIC=policy_minus_bc_causal_gate`
- `CAUSAL_POLICY_MIN_MARGIN=0.002`
- `MIN_IMAGINATION_SELECTION_UPDATE=100`
- Same DROID-heavy mix: DROID active 0.80, SOAR positive 0.10, SOAR active 0.10

| Seed run | Selected update | Policy - BC | Policy - dyn-shuffle | Pass |
|---|---:|---:|---:|---|
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_postrl_seed_20260605` | 600 | +0.0043 | +0.0023 | yes |
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_postrl_seed_20260606` | 200 | +0.0025 | +0.0061 | yes |
| `soar_residual_adapter_imagination_causal_hardgate_droidheavy_postrl_seed_20260607` | 800 | +0.0078 | +0.0103 | yes |

Post-RL-only aggregate:

- Pass count: `3/3`
- Mean `policy_minus_bc`: `+0.0048`
- Mean `policy_minus_dyn_shuffle`: `+0.0062`

Promoted post-RL-only artifact:

`sensenova_drone_agent/output/controllable_soar_imagination_policy_postrl_v1/manifest.json`

Paper-ready post-RL table:

`sensenova_drone_agent/paper/soar_imagination_postrl_results.md`

Post-RL breakdown for the promoted seed `20260607`:

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0018 | +0.0238 | +0.0112 | +0.0009 | -0.0112 | -0.0019 |
| all | 8 | +0.0078 | +0.0728 | +0.0534 | +0.0103 | -0.0237 | -0.0011 |
| all | 16 | +0.0187 | +0.2077 | +0.2138 | +0.0067 | +0.0479 | -0.0116 |
| soar | 4 | +0.0021 | +0.0244 | +0.0124 | +0.0008 | -0.0111 | -0.0008 |
| soar | 8 | +0.0073 | +0.0742 | +0.0564 | +0.0098 | -0.0235 | +0.0011 |
| soar | 16 | +0.0185 | +0.2093 | +0.2151 | +0.0127 | +0.0461 | -0.0072 |
| droid | 4 | +0.0021 | +0.0213 | +0.0171 | +0.0033 | -0.0078 | +0.0013 |
| droid | 8 | +0.0077 | +0.0616 | +0.0652 | +0.0157 | -0.0181 | +0.0055 |
| droid | 16 | +0.0163 | +0.1925 | +0.2270 | +0.0180 | +0.0537 | +0.0031 |

Interpretation:

- The BC-only loophole is closed: all three selected checkpoints are actual imagination updates.
- The post-RL artifact improves over BC and zero/shuffle dynamics controls across all source/horizon slices.
- Time-shift remains an adversarial weakness at horizon 4/8, especially on all/SOAR/DROID slices, but DROID horizon 16 clears it.
- The most defensible claim is now: post-BC imagination training yields repeatable learned-simulator gains under the hard BC-plus-shuffle gate.

## Claim Boundary

We can claim that we have a Dreamer-style offline imagination-training pipeline over a learned SOAR/DROID latent simulator.

We can also claim a repeatable learned-simulator causal signal under the all-source hard gate: DROID-heavy training passed the gate in the main run, in `3/3` repeat seeds, and in `3/3` post-RL-only repeat seeds where update `0` was ineligible.

We should not yet claim native Dreamer4-equivalent action-token dynamics or real-world control. The unresolved gap is exact action identity and timing across every source/control slice, especially DROID-only time-shift/far-shuffle controls. The parallel research track remains native continued dynamics with action tokens rather than only a residual adapter.

## Native Action-Token Dynamics Track

Started a native continued-dynamics run to test the stronger paper claim: the world-model dynamics itself, not only a residual adapter or policy head, becomes action-conditioned.

Run:

`sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2`

Container:

`sda-dreamer4-native-actionfocus-m1-325k-v2`

Launch intent:

- Resume tokenizer and dynamics from `dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1`.
- Continue dynamics from roughly step `275k` to `325k`.
- Use explicit dynamics action tokens with `ACTION_FEATURES=current,prev,delta,mean4,norm` and `ACTION_FRAME_OFFSET=-1`.
- Weight clear action sources: SOAR `6x`, DROID `6x`, Dreamer4 mixed-large `2x`, plus Dreamer4 expert, RoboNet, Fractal, and Bridge at `1x`.
- Filter to action/motion-heavy windows with `REQUIRE_NON_NOOP=1`, `NO_OP_THRESHOLD=0.2`, `MIN_NON_NOOP_STEPS=4`, `REQUIRE_VISUAL_DELTA=1`, `VISUAL_DELTA_THRESHOLD=0.01`, and `MIN_VISUAL_DELTA_STEPS=4`.
- Use stronger but bounded contrast: `ACTION_CONTRAST_WEIGHT=1.0`, `ACTION_CONTRAST_MARGIN=0.001`, `ACTION_CONTRAST_SIGNAL=0.25`, negatives `shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse`, high-action weighting `1.5`, latent-delta weighting `2.0`, and weight clip `6.0`.

Implementation notes:

- Added launcher/payload support for visual-delta filtering in `all_data_native_dreamer4_payload.sh` and `launch_all_data_native_dreamer4.sh`.
- Fixed `WMDataset` visual-delta start indexing so tasks with shorter motion arrays trim candidate starts instead of crashing.
- Installed `tensordict` into `.pydeps`; without it, the Dreamer4 HF control tasks were skipped by the action dataset loader.

Gate after completion:

- `native_dynamics_eval_h8_all_data.json` must show normal actions beating shuffle, zero, and temporal counterfactuals under the `1.02x` causal ratio.
- If it passes, rerun the post-RL-only imagination policy using this native dynamics checkpoint with no residual adapter.
- If it only passes shuffle/zero but not temporal negatives, claim remains limited to residual/post-RL learned-simulator gains, with native dynamics listed as an ablation rather than the main claim.

Result:

- Container exited cleanly with code `0`.
- Final checkpoint: `dynamics_ckpts/final_step_0325000.pt`.
- Filtered eval data: `3,843,555` valid windows across `100` tasks.
- The Dreamer4 HF control tasks loaded after adding `tensordict`; static/low-motion tasks were filtered out by the visual-delta gate.

Native dynamics eval:

`sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2/native_dynamics_eval_h8_all_data.json`

| Eval | Normal MSE | Shuffle / normal | Zero / normal | Time-shift / normal | Time-shift2 / normal | Time-shift4 / normal | Time-shift8 / normal | Time-perm / normal | Time-reverse / normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 0.01374 | 1.048 | 1.043 | 1.056 | 1.061 | 1.051 | 1.052 | 1.044 | 1.050 |
| Autoregressive h8 | 0.01853 | 1.148 | 1.362 | 1.011 | 1.018 | 1.035 | 1.092 | 1.037 | 1.051 |

Decision:

- Direct action conditioning: `pass`.
- Autoregressive beats persistence: `pass` (`normal_over_persistence=0.607`).
- Autoregressive action conditioning: `fail` under the strict `1.02x` all-negative gate because `time_shift=1.011` and `time_shift2=1.018`.
- Native dynamics ready for imagination without residual adapter: `no`, by the strict gate.

Interpretation:

- This is a real improvement over the earlier native runs: direct dynamics now beats shuffle, zero, and all temporal counterfactuals.
- The remaining issue is not action-token absence; it is rollout-time temporal precision. In autoregressive generation, immediate neighboring actions are still close enough that one-step and two-step shifted actions can almost substitute for true actions.
- Paper-wise, this supports a strong ablation: action-token continuation can surface causal action signal in direct dynamics, but the residual/post-RL learned simulator remains the current main controllable-simulator claim until native autoregressive temporal negatives clear.

## Native Action-Token Imagination Round

Started a native-only Dreamer-style policy/imagination ablation using the action-token dynamics checkpoint above, with the residual action adapter explicitly disabled.

Run:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_actiontoken_actionfocus_postrl_v1`

Container:

`sda-soar-native-actiontoken-postrl-v1`

Launch intent:

- Use tokenizer and native dynamics from `dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2`.
- Set `RESIDUAL_ADAPTER_CKPT=none`; payload resolved this to `residual_adapter=` in the log.
- Train BC for `1200` steps and imagination for `800` updates.
- Use DROID-heavy reward mixture: `hf_robot_active=0.80,soar_game_positive=0.10,soar_game_active=0.10`.
- Use raw-policy action chunks with `ACTION_CHUNK_LEN=4`, `ACTION_DIM=49`, `RAW_ACTION_DIM=12`, `ACTION_FRAME_OFFSET=-1`, and action features `current,prev,delta,mean4,norm`.
- Select only post-update imagination checkpoints with `MIN_IMAGINATION_SELECTION_UPDATE=100`.
- Select by `policy_minus_bc_causal_gate`, requiring the policy to beat BC while clearing the dynamics-shuffle causal margin.

Initial log check:

- Dynamics checkpoint: `dynamics_ckpts/final_step_0325000.pt`.
- Residual adapter: empty.
- Policy training dataset view: `431,714` valid sequences across `65` tasks.

Gate after completion:

- Inspect `best_imagination_selection.json`, `summary.json`, and `report.md`.
- Run the dedicated breakdown eval at horizons `4`, `8`, and `16`, including zero, shuffle, time-shift, and far-shuffle controls.
- If native-only policy clears the same gates as the residual/post-RL artifact, the paper claim can shift from "residual learned simulator" toward "continued action-token world model supports Dreamer-style offline imagination training."

Result:

- Container exited cleanly with code `0`.
- Best selected eligible checkpoint: update `500`.
- Selection metric: `policy_minus_bc_causal_gate=-1000000.0027`, meaning the causal gate did not pass.
- Best checkpoint eval: `policy_minus_bc=-0.0012`, `policy_minus_zero=-0.0038`, `policy_minus_dyn_zero=-0.0142`, `policy_minus_dyn_shuffle=-0.0015`.
- BC action MSE improved from `0.1230` at step `1` to `0.0476` at step `1200`, so behavior-cloning learned action statistics, but imagination did not improve the policy under native-only dynamics.

Dedicated breakdown:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_actiontoken_actionfocus_postrl_v1/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | -0.0021 | +0.0002 | -0.0034 | -0.0012 | -0.0009 | -0.0006 |
| all | 8 | -0.0012 | -0.0038 | -0.0142 | -0.0015 | -0.0070 | +0.0022 |
| all | 16 | -0.0154 | -0.0531 | -0.0564 | -0.0124 | -0.0538 | -0.0322 |
| soar | 4 | +0.0024 | +0.0027 | -0.0018 | +0.0006 | +0.0007 | +0.0020 |
| soar | 8 | -0.0023 | -0.0064 | -0.0146 | -0.0056 | -0.0083 | -0.0014 |
| soar | 16 | +0.0011 | -0.0385 | -0.0565 | -0.0026 | -0.0368 | -0.0073 |
| droid | 4 | +0.0016 | +0.0020 | -0.0019 | +0.0006 | -0.0001 | +0.0014 |
| droid | 8 | +0.0002 | -0.0098 | -0.0111 | -0.0016 | -0.0045 | -0.0003 |
| droid | 16 | +0.0021 | -0.0364 | -0.0504 | -0.0010 | -0.0288 | -0.0061 |

Interpretation:

- Native-only action-token dynamics is not yet sufficient for Dreamer-style policy improvement.
- The policy sometimes beats BC on source-specific slices, especially SOAR/DROID h4 and h16, but it consistently loses to the dynamics-zero control at every source/horizon slice.
- This strengthens the current paper boundary: the repeatable post-RL control result is still the residual learned-simulator path, while native action-token continuation is an informative failed ablation that exposes the remaining gap between direct action-conditioned prediction and useful autoregressive imagination.

## Native Action-Token Plus Residual Imagination Round

Started an apples-to-apples hybrid ablation to test whether native action-token continued dynamics improves the already-positive residual learned-simulator path.

Run:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_actiontoken_residual_actionfocus_postrl_v1`

Container:

`sda-soar-native-actiontoken-residual-postrl-v1`

Launch intent:

- Use the same native tokenizer and dynamics as the failed native-only ablation.
- Enable residual adapter `residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt`.
- Keep the same BC/imagination schedule, DROID-heavy source mix, action features, and strict post-update `policy_minus_bc_causal_gate` selection.
- Compare directly against both the previous residual-only/post-RL result and the failed native-only result.

Initial log check:

- Dynamics checkpoint: `dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2/dynamics_ckpts/final_step_0325000.pt`.
- Residual adapter: `residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt`.
- Policy training dataset view: `431,714` valid sequences across `65` tasks.

Result:

- Container exited cleanly.
- Selected update: `200`.
- BC final action MSE: `0.0368588`.
- Before imagination: `policy_minus_bc=-0.0004`, `policy_minus_zero=+0.1884`, `policy_minus_dyn_zero=+0.1725`, `policy_minus_dyn_shuffle=+0.0046`.
- After selected imagination update: `policy_minus_bc=+0.0002`, `policy_minus_zero=+0.1890`, `policy_minus_dyn_zero=+0.1730`, `policy_minus_dyn_shuffle=+0.0039`.

Dedicated breakdown:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_actiontoken_residual_actionfocus_postrl_v1/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0004 | +0.0777 | +0.0662 | +0.0011 | +0.0086 | +0.0006 |
| all | 8 | +0.0002 | +0.1890 | +0.1730 | +0.0039 | +0.0274 | +0.0022 |
| all | 16 | +0.0010 | +0.2685 | +0.2590 | +0.0116 | +0.0214 | +0.0058 |
| soar | 4 | +0.0002 | +0.0809 | +0.0706 | +0.0011 | +0.0101 | +0.0004 |
| soar | 8 | +0.0003 | +0.1992 | +0.1848 | +0.0021 | +0.0308 | +0.0016 |
| soar | 16 | +0.0023 | +0.2897 | +0.2801 | +0.0126 | +0.0348 | +0.0072 |
| droid | 4 | +0.0008 | +0.0388 | +0.0456 | +0.0000 | -0.0095 | -0.0005 |
| droid | 8 | +0.0002 | +0.0970 | +0.1170 | -0.0014 | -0.0163 | -0.0027 |
| droid | 16 | +0.0004 | +0.1336 | +0.1684 | -0.0067 | -0.0453 | -0.0117 |

Interpretation:

- Adding the residual adapter recovers a positive imagination result on top of the native action-token dynamics checkpoint.
- The all-source and SOAR slices clear BC, zero-action, dynamics-zero, dynamics-shuffle, time-shift, and far-shuffle controls at horizons `4`, `8`, and `16`.
- DROID still beats BC/zero controls but fails dynamics-shuffle and temporal controls at horizons `8` and `16`, so this is not yet a uniform source-level causality result.
- Paper-wise, the hybrid run is a useful ablation: native action-token continuation helps as a dynamics base, but the residual learned-simulator path is still the stronger headline artifact until native closed-loop temporal controls pass without the residual crutch.

## Native Closed-Loop Dynamics Continuation

Implemented a closed-loop dynamics auxiliary in `dreamer4/dreamer4/train_dynamics.py`.

Purpose:

- Fix the specific native-dynamics failure where direct action conditioning passes but autoregressive rollouts fail strict temporal controls.
- Train the dynamics model on its own predicted past latents rather than only teacher-forced dataset latents.
- Add closed-loop counterfactual action negatives so wrong actions must produce worse future-latent predictions during rollout.

Implementation:

- New `closed_loop_rollout_loss()` starts from clean context latents, predicts future latents autoregressively, and uses the predicted latents as subsequent history.
- The current-frame query uses a configurable shortcut signal level rather than a full expensive K-step sampler.
- History defaults to detached predictions to keep memory bounded.
- Optional closed-loop contrast supports `shuffle`, `far_shuffle`, `zero`, `time_shift`, `time_shift2`, `time_reverse`, and `time_perm`.
- `all_data_native_dreamer4_payload.sh` and `launch_all_data_native_dreamer4.sh` now expose default-off `CLOSED_LOOP_*` flags.

Run:

`sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1`

Container:

`sda-dreamer4-all-data-closedloop-m1-v1`

Configuration:

- Resume dynamics: `dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2/dynamics_ckpts/final_step_0325000.pt`.
- Frozen tokenizer: `dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2/tokenizer_ckpts/latest.pt`.
- Target max step: `375000`, adding `50000` dynamics steps.
- LR: `2e-5`.
- Source weights: Dreamer4 expert `1`, Dreamer4 mixed-large `2`, SOAR `6`, RoboNet `1`, DROID `6`, Fractal `1`, Bridge `1`.
- Action offset: `-1`.
- Action features: `current,prev,delta,mean4,norm`, action dim `49`.
- Closed-loop objective: weight `0.25`, context `8`, horizon `3`, signal `0.1`.
- Closed-loop contrast: weight `1.0`, margin `0.01`, negatives `shuffle,time_shift,time_shift2`.
- Existing one-step action contrast remains enabled with negatives `shuffle,zero,time_shift,time_shift2`.

Initial log:

- Step `325000`: `loss=0.016965`, `act_contrast=0.003291`, `act_shuffle=1.182`, `act_zero=1.237`, `act_time=1.820`.
- Closed loop: `closed_loop=0.051921`, `cl_shuffle=1.000`, `cl_time=1.014`.

Reading:

- The run is applying pressure to the exact AR causal failure without changing tokenizer or policy training.
- If the final native dynamics eval clears rollout time-shift and far-shuffle controls, the next step is rerunning native-only imagination without the residual adapter.
- If it still fails temporal controls, we keep this as a principled negative ablation and continue treating the residual learned-simulator route as the paper's main artifact.

Final result:

- Container exited cleanly with code `0`.
- Final dynamics checkpoint: `dynamics_ckpts/final_step_0375000.pt`.
- Eval: `native_dynamics_eval_h8_all_data.json`.
- Decision: `strict_gate_passed=true`, `native_dynamics_ready_for_imagination=true`.

Before/after native dynamics comparison:

| Run | Strict gate | AR normal | AR/persistence | AR shuffle/normal | AR zero/normal | AR time-shift/normal | AR time-shift2/normal | AR far-shuffle/normal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| action-focus 325k | no | 0.018528 | 0.607 | 1.148 | 1.362 | 1.011 | 1.018 | n/a |
| closed-loop 375k | yes | 0.007510 | 0.314 | 1.803 | 1.350 | 5.406 | 6.465 | 2.179 |

Closed-loop 375k direct dynamics:

| Control | MSE / normal | Pair pass |
|---|---:|---:|
| shuffle | 1.328 | 0.828 |
| zero | 1.087 | 0.875 |
| time_shift | 6.316 | 1.000 |
| time_shift2 | 7.183 | 1.000 |
| far_shuffle | 1.487 | 0.984 |

Closed-loop 375k autoregressive dynamics:

| Control | MSE / normal | Pair pass |
|---|---:|---:|
| shuffle | 1.803 | 0.844 |
| zero | 1.350 | 0.820 |
| time_shift | 5.406 | 1.000 |
| time_shift2 | 6.465 | 1.000 |
| far_shuffle | 2.179 | 0.949 |

Interpretation:

- This is the first native action-token dynamics checkpoint that passes the strict autoregressive causal gate without the residual adapter.
- The closed-loop objective directly fixed the previous failure mode: temporally wrong actions now produce substantially worse AR futures.
- Next functional step is rerunning native-only imagination with `final_step_0375000.pt` and no residual adapter. If that passes the same policy gates, the paper can upgrade from "residual learned simulator" to "native action-token world model supports offline Dreamer-style imagination training."

## Native Closed-Loop Dynamics Imagination Round

Ran a native-only Dreamer-style policy/imagination ablation using the closed-loop dynamics checkpoint, with the residual adapter disabled.

Run:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_postrl_v1`

Container:

`sda-soar-native-closedloop-postrl-v1`

Configuration:

- Tokenizer and dynamics from `dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1`.
- Dynamics checkpoint: `dynamics_ckpts/final_step_0375000.pt`.
- Residual adapter: disabled.
- BC steps: `1200`.
- Imagination updates: `800`.
- Train balance: `hf_robot_active=0.80,soar_game_positive=0.10,soar_game_active=0.10`.
- Selection: post-update only with `MIN_IMAGINATION_SELECTION_UPDATE=100`.
- Selection metric: `policy_minus_bc_causal_gate` with `CAUSAL_POLICY_MIN_MARGIN=0.002`.
- Eval batches: `96`.

Result:

- Container exited cleanly with code `0`.
- Selected update: `300`.
- Before imagination: `policy_minus_bc=+0.0006`, `policy_minus_zero=-0.0180`, `policy_minus_dyn_zero=-0.0226`, `policy_minus_dyn_shuffle=+0.0037`.
- After selected imagination update: `policy_minus_bc=+0.0151`, `policy_minus_zero=-0.0035`, `policy_minus_dyn_zero=-0.0087`, `policy_minus_dyn_shuffle=+0.0021`.
- The selected policy improves over BC and dynamics-shuffle, but it still loses to zero-action and dynamics-zero controls at the default horizon `8`.

Dedicated breakdown:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_postrl_v1/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0029 | +0.0015 | +0.0008 | +0.0003 | +0.0109 | +0.0006 |
| all | 8 | +0.0151 | -0.0035 | -0.0087 | +0.0021 | +0.0558 | -0.0006 |
| all | 16 | +0.0395 | -0.0110 | +0.0605 | +0.0028 | +0.0749 | +0.0116 |
| soar | 4 | +0.0026 | +0.0022 | +0.0013 | +0.0004 | +0.0114 | +0.0008 |
| soar | 8 | +0.0137 | -0.0027 | -0.0090 | +0.0014 | +0.0599 | +0.0005 |
| soar | 16 | +0.0385 | -0.0089 | +0.0688 | +0.0027 | +0.0902 | +0.0013 |
| droid | 4 | +0.0023 | +0.0030 | +0.0022 | +0.0007 | +0.0131 | -0.0001 |
| droid | 8 | +0.0091 | +0.0056 | -0.0010 | -0.0015 | +0.0668 | -0.0007 |
| droid | 16 | +0.0347 | +0.0631 | +0.1279 | -0.0076 | +0.1112 | -0.0048 |

Interpretation:

- Closed-loop native dynamics materially improves the native-only imagination path compared to the prior action-token run.
- The policy now improves over BC across every source/horizon slice and clears time-shift controls everywhere.
- The result is not yet a clean native-only headline because zero/dyn-zero and shuffle/far-shuffle controls are not uniformly beaten, especially at horizon `8` and on DROID.
- Paper-wise, this is now a strong ablation: native closed-loop action-token dynamics can support post-BC imagination gains, but the promoted residual/post-RL artifact remains the cleaner headline until native-only clears all hard controls.

## Native Closed-Loop Zero-Aware Imagination Round

Implemented a stricter native-only selection path to target the remaining zero-action failure.

Implementation:

- Added `BEST_IMAGINATION_METRIC=policy_minus_bc_zero_causal_gate`.
- The new gate requires `policy_minus_bc > 0`, `policy_minus_zero >= 0`, `policy_minus_dyn_zero >= CAUSAL_POLICY_MIN_MARGIN`, and `policy_minus_dyn_shuffle >= CAUSAL_POLICY_MIN_MARGIN`.
- Exposed `TRAIN_VALUE_DURING_IMAGINATION` through the launcher so the value head can be updated during imagination, closer to the Dreamer training loop.

Run:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_v1`

Container:

`sda-soar-native-closedloop-zeroaware-postrl-v1`

Configuration:

- Tokenizer and dynamics from `dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1`.
- Dynamics checkpoint: `dynamics_ckpts/final_step_0375000.pt`.
- Residual adapter: disabled.
- BC steps: `2400`.
- Imagination updates: `800`.
- Value training during imagination: enabled.
- Train balance: `hf_robot_active=0.80,soar_game_positive=0.10,soar_game_active=0.10`.
- Selection: post-update only with `MIN_IMAGINATION_SELECTION_UPDATE=100`.
- Selection metric: `policy_minus_bc_zero_causal_gate` with `CAUSAL_POLICY_MIN_MARGIN=0.002`.
- Reward contrast: weight `2.0`, horizon `4`, negatives `zero,zero,shuffle`.
- Eval batches: `96`.

Result:

- Container exited cleanly with code `0`.
- Selected update: `350`.
- Before imagination: `policy_minus_bc=+0.0169`, `policy_minus_zero=+0.0440`, `policy_minus_dyn_zero=-0.0296`, `policy_minus_dyn_shuffle=+0.0059`.
- After selected imagination update: `policy_minus_bc=+0.0369`, `policy_minus_zero=+0.0640`, `policy_minus_dyn_zero=+0.0026`, `policy_minus_dyn_shuffle=+0.0136`.
- This is the first native-only checkpoint to pass the full default-horizon hard gate without a residual adapter.

Dedicated breakdown:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_v1/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0049 | +0.0224 | +0.0075 | +0.0032 | +0.0338 | +0.0039 |
| all | 8 | +0.0369 | +0.0640 | +0.0026 | +0.0136 | +0.1518 | +0.0118 |
| all | 16 | +0.0093 | +0.0995 | +0.0173 | +0.0558 | -0.0137 | +0.0305 |
| soar | 4 | +0.0036 | +0.0246 | +0.0084 | +0.0034 | +0.0352 | +0.0025 |
| soar | 8 | +0.0339 | +0.0684 | +0.0029 | +0.0111 | +0.1587 | +0.0111 |
| soar | 16 | +0.0006 | +0.1134 | +0.0397 | +0.0518 | +0.0170 | +0.0217 |
| droid | 4 | +0.0035 | +0.0121 | +0.0046 | -0.0004 | +0.0337 | -0.0001 |
| droid | 8 | +0.0291 | +0.0381 | -0.0080 | -0.0019 | +0.1502 | +0.0016 |
| droid | 16 | +0.0031 | +0.1859 | +0.1224 | +0.0412 | +0.0342 | +0.0135 |

Repeat seeds:

Same configuration as above, with fixed held-out split/eval seeds and independent train/balance seeds.

| Seed | Selected update | Pass default h8 strict gate | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Causal gain |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 20260529 | 350 | yes | +0.0369 | +0.0640 | +0.0026 | +0.0136 | +0.0026 |
| 20260530 | 550 | yes | +0.0335 | +0.0670 | +0.0232 | +0.0088 | +0.0088 |
| 20260531 | 750 | no | +0.0192 | -0.0088 | -0.0066 | +0.0124 | -0.0066 |
| 20260532 | 700 | no | +0.0533 | -0.0286 | -0.0585 | +0.0056 | -0.0585 |
| 20260533 | 400 | yes | +0.0243 | +0.0999 | +0.0153 | +0.0040 | +0.0040 |

Pass count:

- Including the original zero-aware run: `3/5`.
- New repeat seeds only: `2/4`.
- All `5/5` selected policies improve over BC.
- The two failures are not BC failures; they are zero/dyn-zero causal-control failures.

Passing repeat breakdowns:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_seed_20260530/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0188 | +0.0383 | +0.0036 | +0.0018 | +0.0216 | +0.0032 |
| all | 8 | +0.0335 | +0.0670 | +0.0232 | +0.0088 | +0.0660 | +0.0061 |
| all | 16 | +0.0326 | +0.1670 | +0.1288 | +0.0064 | +0.1784 | +0.0145 |
| soar | 4 | +0.0165 | +0.0391 | +0.0046 | +0.0029 | +0.0235 | +0.0017 |
| soar | 8 | +0.0287 | +0.0658 | +0.0163 | +0.0066 | +0.0645 | +0.0056 |
| soar | 16 | +0.0339 | +0.1742 | +0.1322 | +0.0082 | +0.1815 | +0.0113 |
| droid | 4 | +0.0168 | +0.0355 | +0.0030 | -0.0005 | +0.0267 | +0.0005 |
| droid | 8 | +0.0347 | +0.0671 | +0.0144 | -0.0035 | +0.0835 | +0.0000 |
| droid | 16 | +0.0279 | +0.1951 | +0.1434 | +0.0004 | +0.1789 | +0.0078 |

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_seed_20260533/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0028 | +0.0327 | +0.0037 | +0.0015 | +0.0192 | +0.0019 |
| all | 8 | +0.0243 | +0.0999 | +0.0153 | +0.0040 | +0.1171 | +0.0042 |
| all | 16 | -0.0264 | +0.2442 | +0.1476 | +0.0116 | +0.1569 | +0.0182 |
| soar | 4 | +0.0027 | +0.0326 | +0.0048 | +0.0015 | +0.0205 | +0.0004 |
| soar | 8 | +0.0236 | +0.0997 | +0.0139 | +0.0022 | +0.1209 | +0.0040 |
| soar | 16 | -0.0333 | +0.2520 | +0.1657 | +0.0147 | +0.1802 | +0.0140 |
| droid | 4 | +0.0015 | +0.0277 | +0.0024 | -0.0015 | +0.0197 | -0.0013 |
| droid | 8 | +0.0149 | +0.0875 | +0.0042 | -0.0063 | +0.1127 | -0.0030 |
| droid | 16 | -0.0384 | +0.3182 | +0.2228 | +0.0029 | +0.1928 | +0.0063 |

Failing repeat breakdowns:

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_seed_20260531/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0091 | +0.0045 | -0.0010 | +0.0037 | +0.0257 | +0.0043 |
| all | 8 | +0.0192 | -0.0088 | -0.0066 | +0.0124 | +0.0514 | +0.0162 |
| all | 16 | +0.0137 | -0.0897 | -0.1472 | +0.0329 | -0.1216 | +0.0181 |
| soar | 4 | +0.0077 | +0.0048 | +0.0009 | +0.0047 | +0.0244 | +0.0035 |
| soar | 8 | +0.0177 | -0.0090 | -0.0115 | +0.0116 | +0.0474 | +0.0119 |
| soar | 16 | +0.0177 | -0.0849 | -0.1475 | +0.0361 | -0.1245 | +0.0235 |
| droid | 4 | +0.0075 | -0.0040 | -0.0029 | +0.0003 | +0.0259 | +0.0010 |
| droid | 8 | +0.0173 | -0.0195 | -0.0148 | +0.0010 | +0.0702 | +0.0039 |
| droid | 16 | +0.0100 | -0.0210 | -0.0607 | +0.0177 | -0.0281 | +0.0216 |

`sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_seed_20260532/breakdown_eval/breakdown_summary.json`

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0151 | +0.0017 | -0.0050 | +0.0001 | +0.0091 | +0.0007 |
| all | 8 | +0.0533 | -0.0286 | -0.0585 | +0.0056 | +0.0362 | +0.0066 |
| all | 16 | +0.0805 | -0.1080 | -0.1490 | +0.0387 | -0.2012 | +0.0206 |
| soar | 4 | +0.0154 | +0.0030 | -0.0048 | +0.0005 | +0.0092 | +0.0003 |
| soar | 8 | +0.0549 | -0.0260 | -0.0580 | +0.0052 | +0.0398 | +0.0062 |
| soar | 16 | +0.0837 | -0.0992 | -0.1341 | +0.0372 | -0.1782 | +0.0203 |
| droid | 4 | +0.0085 | +0.0005 | -0.0046 | -0.0007 | +0.0107 | -0.0009 |
| droid | 8 | +0.0355 | -0.0259 | -0.0507 | -0.0029 | +0.0486 | -0.0026 |
| droid | 16 | +0.0502 | -0.0101 | -0.0408 | +0.0083 | -0.1400 | -0.0033 |

Repeatability analysis artifact:

- `sensenova_drone_agent/output/native_zeroaware_repeatability_analysis/native_zeroaware_repeatability_analysis.md`
- `sensenova_drone_agent/output/native_zeroaware_repeatability_analysis/native_zeroaware_repeatability_analysis.json`
- Summary: pass count `3/5`, mean policy-minus-BC `+0.0335`.
- Failure signature: fail seeds still improve over BC, but the selected policies underperform zero or dyn-zero controls at the default horizon and at horizon 16.

Interpretation:

- The default all-source horizon-8 hard gate passes natively in `3/5` seeds without the residual adapter.
- The repeatable claim is not yet as clean as the residual/post-RL artifact, because two native seeds lose to zero/dyn-zero controls.
- The strongest defensible native claim is now: zero-aware native Dreamer-style imagination can beat BC and all default h8 causal controls in a majority of seeds, but remains seed-sensitive.
- SOAR is more stable than DROID. DROID h8 still tends to fail the shuffle control even in passing all-source seeds.
- Paper-wise, this result upgrades native-only from a single-run breakthrough to a promising but not fully robust ablation. The residual/post-RL artifact remains the cleaner repeatable headline unless we improve native zero/dyn-zero stability.

## Source-Aware Native Zero-Aware Stability Sweep

Follow-up fix started from the repeatability analysis:

- Script: `sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_stability_sweep.sh`.
- Variant 1: `soar_residual_adapter_imagination_native_closedloop_zeroaware_sourceaware_shortfall_seed_20260534`.
- Variant 2: `soar_residual_adapter_imagination_native_closedloop_zeroaware_sourceaware_low_lr_shortfall_seed_20260535`.
- Selection metric: `policy_minus_bc_zero_causal_gate_source_aware`.
- Source eval: `all,soar,droid` with `32` source batches per eval.
- Hard source gates: `all,soar`.
- Soft source gate: `droid` with minimum dyn-control margin `-0.005`.
- Causal shortfall policy penalty: weight `0.5`, margin `0.002`.
- Reward contrast: weight `2.5`, negatives `zero,zero,zero,shuffle`.
- Low-LR variant: imagination LR `1e-5`.

Purpose:

- Do not select checkpoints that only win in aggregate while failing SOAR-specific zero/dyn-zero controls.
- Penalize imagined policy updates that leave causal advantages below the strict margin.
- Measure whether the native result can be moved from `3/5` seed-sensitive to a stable native-only claim.

Final sweep result:

| Variant | Seed | Selected update | Source gate pass | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |
|---|---:|---:|:---:|---:|---:|---:|---:|
| source-aware shortfall | 20260534 | 100 | no | +0.0294 | +0.1024 | +0.0243 | +0.0006 |
| source-aware low-LR shortfall | 20260535 | 800 | no | +0.0496 | +0.1688 | +0.0234 | -0.0057 |

Selected source slices:

| Variant | Source | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |
|---|---|---:|---:|---:|---:|
| source-aware shortfall | all | +0.0294 | +0.1024 | +0.0243 | +0.0006 |
| source-aware shortfall | soar | +0.0354 | +0.1139 | +0.0322 | +0.0110 |
| source-aware shortfall | droid | +0.0356 | +0.0728 | +0.0161 | -0.0080 |
| source-aware low-LR shortfall | all | +0.0496 | +0.1688 | +0.0234 | -0.0057 |
| source-aware low-LR shortfall | soar | +0.0565 | +0.1849 | +0.0395 | +0.0091 |
| source-aware low-LR shortfall | droid | +0.0542 | +0.1701 | +0.0457 | -0.0065 |

Interpretation:

- The source-aware/shortfall penalty did not produce a clean strict-gated native checkpoint.
- Both selected policies improved over BC and zero by large margins, so the optimizer is not inert.
- The remaining failure is the dyn-shuffle control: SOAR clears it, while DROID remains negative; the low-LR run also fails the aggregate dyn-shuffle margin.
- This argues against claiming robust native action-causal imagination yet. The result instead supports a narrower ablation claim: native action-token imagination improves the policy, but action-causal reliability is still source-sensitive and control-sensitive.

## SOAR-Only Strict Native Sweep And DROID Audit

We split the next diagnostic into clean-source and weak-source tests.

SOAR-only strict sweep:

- Script: `sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_soar_only_sweep.sh`.
- Payload selector: `DATA_SOURCES=soar`.
- Seeds: `20260536`, `20260537`.
- Source gates: hard `all,soar`; no DROID soft gate.
- Training mix: `soar_game_positive=0.50,soar_game_active=0.50`.
- Selection metric: `policy_minus_bc_zero_causal_gate_source_aware`.
- Logs confirm `data_sources=soar` and `33635` valid SOAR windows.

Final SOAR-only sweep result:

| Variant | Seed | Selected update | Source gate pass | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |
|---|---:|---:|:---:|---:|---:|---:|---:|
| SOAR-only strict | 20260536 | 800 | yes | +0.0278 | +0.2103 | +0.0964 | +0.0027 |
| SOAR-only strict low-LR | 20260537 | 750 | no | +0.0150 | +0.1457 | +0.0917 | -0.0016 |

Selected source slices:

| Variant | Source | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |
|---|---|---:|---:|---:|---:|
| SOAR-only strict | all | +0.0278 | +0.2103 | +0.0964 | +0.0027 |
| SOAR-only strict | soar | +0.0280 | +0.2146 | +0.0974 | +0.0060 |
| SOAR-only strict low-LR | all | +0.0150 | +0.1457 | +0.0917 | -0.0016 |
| SOAR-only strict low-LR | soar | +0.0183 | +0.1519 | +0.0953 | +0.0035 |

Breakdown for the passing SOAR-only strict checkpoint:

| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4 | +0.0006 | +0.0575 | +0.0187 | -0.0018 | +0.0528 | -0.0010 |
| all | 8 | +0.0278 | +0.2101 | +0.0952 | +0.0021 | +0.3044 | -0.0010 |
| all | 16 | +0.0478 | +0.7763 | +0.7117 | +0.0130 | +0.8489 | +0.0088 |
| soar | 4 | +0.0006 | +0.0575 | +0.0187 | -0.0018 | +0.0528 | -0.0010 |
| soar | 8 | +0.0278 | +0.2101 | +0.0952 | +0.0021 | +0.3044 | -0.0010 |
| soar | 16 | +0.0478 | +0.7763 | +0.7117 | +0.0130 | +0.8489 | +0.0088 |

Interpretation:

- The SOAR-only strict run provides the cleanest native action-token imagination result so far: it clears BC, zero, dyn-zero, and dyn-shuffle controls at the default h8 gate without the residual adapter.
- The larger breakdown confirms the h8 and h16 gates, but h4 remains weak on shuffle and far-shuffle is slightly negative at h8. The safe claim is therefore default-horizon learned-simulator improvement, not universal action-causal dominance across every horizon/control.
- Because the run is SOAR-only, the `all` and `soar` breakdown rows are identical; the earlier all-vs-SOAR discrepancy came from different eval samplers, not hidden DROID contamination.

DROID-only identifiability audit:

- Script: `sensenova_drone_agent/scripts/experiments/audit_action_identifiability.py`.
- Output: `sensenova_drone_agent/output/action_identifiability_audit_droid_only_v1`.
- Sampled windows: `4096` from `398079` DROID windows.
- Active action dims: `29`; nonzero action fraction: `0.913`.
- Reward signal: mean `0.0`, std `0.0`, positive fraction `0.0`.
- Best action incremental R2 vs scene: `-0.0058`.
- Best action-only R2 vs mean: `-0.0097`.
- Decision: `data_action_signal_detected=false`.

Interpretation:

- DROID actions are numerically present, but this audit cannot find action-predictive visual dynamics beyond scene history.
- DROID currently has no usable reward labels in our export, so DROID-specific reward calibration is not meaningful yet.
- The DROID calibration branch is therefore gated off until we add better task/reward labels or a stronger action-effect target. Running reward/value imagination on this source now would mostly train against zero reward and ambiguous visual causality.
- If SOAR-only passes, the paper claim should separate a positive SOAR result from a DROID negative/diagnostic transfer result.

## SOAR-Only Regular-LR Repeat Hardening

We started the repeatability hardening pass after the first SOAR-only strict win.

- Launcher: `sensenova_drone_agent/scripts/experiments/launch_native_zeroaware_soar_only_regular_lr_repeats.sh`.
- Queue helper: `sensenova_drone_agent/scripts/experiments/queue_remaining_soar_only_repeats.sh`.
- Breakdown evaluator: `sensenova_drone_agent/scripts/experiments/eval_soar_only_regular_lr_repeat_breakdowns.sh`.
- Seeds: `20260601,20260602,20260603,20260604,20260605`.
- Config: same regular-LR SOAR-only strict setup as the passing seed `20260536`.
- Parallelism: initially two training containers across GPUs 0 and 1; after confirming memory headroom, all five repeats were launched concurrently.
- Larger eval target after completion: passing checkpoints get `h4,h8,h16` breakdowns with `256` eval batches by default.
- Final status: all five repeat seeds completed.
- Repeat outcome: strict selector passes were `20260601` and `20260604`; strict selector failures were `20260602`, `20260603`, and `20260605`.
- Combined with the original strict seed `20260536` and low-LR diagnostic `20260537`, the pass count is `3/7`; excluding the low-LR diagnostic, regular-LR SOAR-only is `3/6`.

The acceptance bar for paper repeatability is the strict h8 gate: policy must beat BC, zero-action dynamics, dyn-zero, and dyn-shuffle with the causal margin. Horizon `4/8/16` breakdowns are used to scope the claim rather than select the checkpoint during training.

Final repeat summary:

| Seed | Selected update | Strict selector | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |
|---:|---:|:---:|---:|---:|---:|---:|
| 20260536 | 800 | yes | +0.0278 | +0.2103 | +0.0964 | +0.0027 |
| 20260537 | 750 | no | +0.0150 | +0.1457 | +0.0917 | -0.0016 |
| 20260601 | 700 | yes | +0.0212 | +0.2151 | +0.1059 | +0.0029 |
| 20260602 | 750 | no | +0.0226 | +0.3501 | +0.2308 | -0.0062 |
| 20260603 | 600 | no | +0.0441 | +0.2241 | +0.1382 | -0.0024 |
| 20260604 | 750 | yes | +0.0176 | +0.2240 | +0.1226 | +0.0058 |
| 20260605 | 300 | no | +0.0284 | +0.5867 | +0.3814 | -0.0070 |

Large-eval hardening:

- Seed `20260601` does not survive the larger `256`-batch breakdown: h8 dyn-shuffle is only `+0.0012`, below the `+0.002` margin, and h16 falls below BC.
- Seed `20260604` survives the default h8 large eval with policy-BC `+0.0157`, dyn-zero `+0.1244`, and dyn-shuffle `+0.0027`, but h16 falls below BC.
- Seed `20260536` remains the strongest result: default h8 passes, and h16 also beats BC and dyn-shuffle.

Claim boundary:

- Safe: native SOAR action-token imagination can produce default-horizon learned-simulator improvements over BC, zero-action, dyn-zero, and dyn-shuffle controls.
- Unsafe: a seed-stable, horizon-universal, or real-world control claim.

Decision note: do not chase lucky seeds.

- Current evidence is real but fragile: several seeds clear the default h8 gate, but the pass rate is low and some selector passes fail larger-eval hardening.
- Additional seed sweeps can estimate variance, but they should not be used to find a prettier headline checkpoint.
- Treat the strongest native checkpoints as ablations that motivate the next method, not as the main result.
- The next main technical push should improve the action-causal simulator itself: continued action-conditioned dynamics training with true-vs-shuffle/time-shift/far-shuffle losses, mixed horizons, held-out large-eval selection, and preservation of the original visual prior.

## Continued Action-Conditioned Dynamics Training

We added a method-level follow-up that targets the root failure mode instead of searching for lucky imagination seeds.

Scripts:

- Payload: `sensenova_drone_agent/scripts/experiments/soar_action_conditioned_dynamics_continuation_payload.sh`.
- Launcher: `sensenova_drone_agent/scripts/experiments/launch_soar_action_conditioned_dynamics_continuation.sh`.
- Selector: `sensenova_drone_agent/scripts/select_dreamer4_soar_dynamics_checkpoint.py`.

Design:

- Freeze the tokenizer by default and reuse `dreamer4_soar_native_v2_action_contrast/tokenizer_ckpts/latest.pt`.
- Resume the prior SOAR dynamics checkpoint shape-safely, preserving compatible world-model weights while allowing larger action features.
- Expand action inputs to `current,prev,delta,mean4,norm` with padded `ACTION_DIM=64`.
- Train on visible/actionful SOAR windows with non-noop and visual-delta filters.
- Add single-step action contrast over `shuffle,zero,time_shift,time_shift2,far_shuffle,effect_far_shuffle`.
- Add closed-loop latent rollout loss with the same counterfactual controls.
- Select checkpoints with a held-out h8/h16 dynamics gate, not by latest checkpoint or training loss.

Hard dynamics gate:

- Evaluates saved checkpoints with `select_dreamer4_soar_dynamics_checkpoint.py`.
- Default eval uses horizons `8,16`, `256` batches, and controls `shuffle,zero,time_shift,far_shuffle`.
- A checkpoint passes only if direct and autoregressive true-action predictions beat all controls by the configured ratio and the autoregressive rollout beats persistence.

Intended paper role:

- If this passes robustly, it becomes the main methodological improvement after the fragile retrofit ablation.
- If it fails, it gives a stronger negative result: action-token retrofit plus continued dynamics contrast is still insufficient at SOAR scale.
