# Native Dreamer4 Imagination Result

## Result
- Date: `2026-05-15`
- Best run: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2`
- Summary: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2/summary.json`
- Report: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2/report.md`
- Status: first positive learned-dynamics imagination result.

## Claim Boundary
- This is an internal learned-dynamics result.
- It shows policy improvement inside the frozen learned dynamics model.
- It does not yet prove real-environment transfer, SOAR robot transfer, Gazebo/PX4 transfer, or real drone control.

## What Passed
The calibrated v2 run improved the policy over both learned-dynamics controls:

- Before policy minus BC-prior: `-0.0112`
- After policy minus BC-prior: `0.0587`
- After policy minus zero-action: `0.2369`
- Per-task mean policy-minus-BC: `0.0479`
- Tasks evaluated: `36`

The result also stayed prior-constrained:

- Policy prior MSE before: `0.0022`
- Policy prior MSE after: `0.0012`
- Policy prior MSE delta: `-0.0010`

## Configuration That Worked
- Target normalization: `per_task`
- Reward clip: `5`
- Value clip: `5`
- Advantage mode: `centered_sign`
- Advantage clip: `2`
- Prior weight: `1.0`
- Sample prior hinge: `25 @ 0.008`
- Mean-prior weight: `10`
- Mean-prior hinge: `100 @ 0.004`
- Imagination learning rate: `3e-5`
- Value head frozen during imagination: true

## What Failed Before This
The first uncalibrated imagination run failed because the policy drifted away from the BC prior and reduced learned return:

- After policy minus BC-prior: `-0.0545`
- After policy minus zero-action: `-0.0140`
- Policy prior MSE increased: `0.0065 -> 0.0127`

The calibrated v1 run fixed reward/value scale but still failed because deterministic policy drift remained too large:

- After policy minus BC-prior: `-0.1002`
- After policy minus zero-action: `-0.0755`
- Policy prior MSE increased: `0.0026 -> 0.0394`

## Interpretation
The useful recipe was not simply "run imagination RL." The passing run required:

- Calibrating reward/value targets per task.
- Using centered-sign advantages to avoid all-positive or all-negative PMPO collapse.
- Freezing the value head during imagination.
- Reducing imagination learning rate.
- Penalizing deterministic policy-mean drift, not only sampled action drift.

This supports the narrower claim that the learned dynamics can provide a training signal for policy improvement when the policy is constrained to stay near the behavior prior.

## Next Validation
The next required gate is external validation:

- Re-run v2 with at least two additional seeds.
- Evaluate whether the after-imagination policy still beats BC under held-out contexts.
- Add a replay/visualization report showing representative imagined rollouts.
- Only after repeatability should we try environment transfer or a SOAR-specific policy path.

## Repeatability Gate v1
- Date: `2026-05-15`
- Suite: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1`
- Report: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1/report.md`
- Summary: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1/summary.json`
- Result: repeatability pass `false`

The calibrated recipe did not reliably beat the BC prior across seeds:

- Seed `31`: pass, after policy-minus-BC `0.0587`, after policy-minus-zero `0.2369`
- Seed `37`: fail, after policy-minus-BC `-0.0190`, after policy-minus-zero `0.1043`
- Seed `43`: fail, after policy-minus-BC `-0.0100`, after policy-minus-zero `0.1111`

Aggregate result:

- Pass fraction: `1/3`
- Mean after policy-minus-BC: `0.0099`
- Mean after policy-minus-zero: `0.1508`
- Mean policy return delta: `0.0716`
- Mean policy prior MSE after imagination: `0.00094`

Interpretation:

- The policy update consistently improves over zero-action and improves the policy return from its own pre-imagination state.
- The improvement is not yet strong enough to reliably outperform the supervised BC prior.
- The prior constraint is working; the remaining issue is advantage/reward signal quality, not uncontrolled policy drift.

Updated next gate:

- Do not advance to transfer or drone claims from this result.
- Add a held-out evaluation split and a no-policy-update ablation.
- Tune the reward/value/imagination objective until at least `2/3` seeds beat BC prior.
- Add rollout visualization after the repeatability gate passes.

## Held-Out Evaluation and Gated-Policy Update v1
- Date: `2026-05-15`
- No-update control: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_no_update_smoke_v2`
- Gated smoke: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_gated_smoke_v1`
- Full seed-37 gated run: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_gated_seed37_v1`
- Result: held-out/gated imagination pass `false`

What changed:

- Added deterministic episode-level held-out splits.
- Added deterministic eval RNG so before/after no-update eval is comparable.
- Added `imagination_mode=no_update` for plumbing controls.
- Added policy-loss gating:
  - `policy_loss_min_advantage_abs`
  - `policy_loss_max_prior_mse`

Held-out split:

- Train windows: `3265020`
- Eval windows: `362780`
- Split unit: episode-level holdout
- Holdout fraction: `0.1`

No-update control:

- Policy return delta: `0.0000`
- Policy prior MSE delta: `0.0000`
- Policy action abs delta: `0.0000`
- Interpretation: deterministic held-out eval plumbing works.

Full seed-37 gated result:

- Before policy-minus-BC: `0.0099`
- Before policy-minus-zero: `-0.0283`
- After policy-minus-BC: `-0.0059`
- After policy-minus-zero: `-0.0441`
- Policy return delta: `-0.0158`
- Policy prior MSE delta: `0.0017`
- Policy action abs delta: `0.0886`

Interpretation:

- The confidence gates were active, but did not fix the objective.
- The update still increased action magnitude and reduced held-out learned return.
- The immediate issue is not evaluation plumbing; it is reward/value or policy-gradient signal quality.
- The next useful change is to add a BC-advantage baseline and only reinforce actions whose imagined return beats the BC-prior rollout from the same context.

## BC-Relative Advantage Smoke v1
- Date: `2026-05-15`
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_bc_relative_smoke_v1`
- Advantage baseline: `bc_return`
- Policy-loss min advantage abs: `0.25`
- Policy-loss max prior MSE: `0.12`
- Result: promising smoke, not yet a pass

Held-out smoke result:

- Before policy-minus-BC: `-0.0075`
- Before policy-minus-zero: `0.0118`
- After policy-minus-BC: `-0.0060`
- After policy-minus-zero: `0.0134`
- Policy return delta: `0.0016`
- Policy prior MSE delta: `-0.00085`
- Policy action abs delta: `-0.0024`

Interpretation:

- BC-relative advantages are more stable than value-baseline advantages in this setting.
- The update improved held-out learned return and reduced policy drift instead of increasing action magnitude.
- The policy still did not beat the BC prior in the short smoke run, so this is not yet sufficient evidence for imagination-training success.
- The next required test is a full seed run with the BC-relative objective, followed by multi-seed repeatability if it passes.

## Balanced Held-Out Evaluation Fix
- Date: `2026-05-15`
- Code: `sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py`
- No-update control: `sensenova_drone_agent/output/native_dreamer4_imagination_balanced_eval_no_update_smoke_v1`
- Balanced full run: `sensenova_drone_agent/output/native_dreamer4_imagination_balanced_eval_bc_relative_seed37_v1`

The previous full held-out run was too narrow because `eval_loader` used the first held-out windows in task order. With `eval_batches=64` and batch size `4`, it evaluated only task `0` (`acrobot-swingup`).

The fix creates a deterministic task-balanced held-out subset:

- Eval sampling mode: `balanced_task_round_robin_with_replacement`
- Eval samples: `256`
- Tasks sampled: `37`
- Min samples per task: `6`
- Max samples per task: `7`

No-update control after the fix:

- Tasks evaluated before/after: `37`
- Policy return delta: `0.0000`
- Policy prior MSE delta: `0.0000`
- Policy action abs delta: `0.0000`
- Interpretation: balanced evaluation is deterministic and suitable for before/after comparisons.

Balanced full seed-37 BC-relative result:

- Before policy-minus-BC: `0.0059`
- Before policy-minus-zero: `0.0405`
- After policy-minus-BC: `0.0029`
- After policy-minus-zero: `0.0375`
- Policy return delta: `-0.0030`
- Policy prior MSE delta: `-0.0064`
- Policy action abs delta: `0.0251`
- Per-task mean policy-minus-BC after: `0.0041`

Interpretation:

- The post-imagination policy still beats BC and zero-action on balanced held-out evaluation.
- However, the imagination update slightly degrades the already-good post-BC policy.
- The BC-relative objective is stable and reduces prior MSE, but it does not yet improve the policy over its pre-imagination state.
- This is not a Dreamer-style imagination-training success claim yet; it is evidence that the learned dynamics/reward path can preserve a decent policy while applying constrained updates.

Updated next gate:

- Treat balanced evaluation as mandatory for all future imagination tests.
- Change the imagination objective so it only updates when it improves over the pre-imagination policy, not merely when it stays above BC/zero.
- Add task-level diagnostics to identify whether only a subset of tasks is being degraded.

## Game-Action Native Imagination v1
- Date: `2026-05-18`
- Dynamics source: `sensenova_drone_agent/output/dreamer4_game_actions_native_blocks_v1`
- Output: `sensenova_drone_agent/output/dreamer4_game_actions_imagination_blocks_v1`
- Dataset: `sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1`
- Result: plumbing pass, weak BC-prior pass, zero-action baseline fail.

This run used the first local native dynamics checkpoint that passed action-conditioned rollout evaluation on lightweight Procgen/ViZDoom data:

- Direct action conditioning detected: `true`
- Autoregressive action conditioning detected: `true`
- Autoregressive beats persistence: `true`

The initial smoke showed that the default stochastic policy variance was too high for this small domain. Lowering `log_std_init` from `-1.0` to `-2.5` reduced sampled-action drift and made the smoke beat the BC prior.

Full run configuration:

- BC steps: `1200`
- Imagination updates: `400`
- Target normalization: `per_task`
- Advantage baseline: `bc_return`
- Advantage mode: `centered_sign`
- Action features: `current,prev,delta,mean4,norm`
- Expanded action dimension: `61`
- Initial log std: `-2.5`
- Held-out eval windows: `800`
- Balanced eval samples: `256`

Full result:

- Before policy-minus-BC: `-0.0270`
- After policy-minus-BC: `0.0004`
- Policy return delta: `0.0274`
- After policy-minus-zero: `-0.0234`
- Policy prior MSE delta: `-0.0018`

Interpretation:

- The BC/reward/value/imagination path works end-to-end on action-conditioned learned dynamics.
- The imagination update improves the policy over its own pre-imagination state and barely beats the BC prior.
- The learned reward still prefers zero-action over the learned policy, so this is not yet a strong Dreamer-style agent success.
- The next blocker is reward/task data quality, not core launcher plumbing.
