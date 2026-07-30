# Action-Conditioned Imagination Handoff

Date: 2026-07-06

This is the canonical handoff for the Kairos/Sensenova, Dreamer4, SOAR, and latent-imagination work. It consolidates the conversation history, experiment history, current artifacts, failed paths, positive signals, claim boundaries, and recommended next steps so another agent can resume without relying on chat memory.

## Fast Read

The project started with the goal of making the Kairos/Sensenova video world model control drones. After reading Dreamer4 and testing multiple retrofit paths, the claim evolved:

```text
Original goal:
Kairos/Sensenova directly controls drones.

Intermediate goal:
Retrofitted Kairos/Sensenova latents become a Dreamer-style action-conditioned simulator.

Current defensible goal:
Use action-labeled data as causal scaffolding to train inspectable latent imagination/planning modules, then show that candidate imagined futures improve decision quality over controls.
```

The central lesson is that action labels only help if the data makes actions causally identifiable:

```text
same or similar context + correct action -> visibly/latently better future
same or similar context + wrong/no-op/time-shifted action -> worse future
```

Most failures came from models learning scene-motion priors rather than exact action causality. The current best direction is not to claim that frozen Kairos is already a controllable simulator. The best direction is to measure and improve decision quality inside an explicit imagination loop.

## Current State In One Paragraph

The repo contains end-to-end infrastructure for SOAR/Dreamer4-style data export, tokenizer/dynamics training, behavior cloning, reward/value heads, PMPO-style imagination updates, residual action adapters, native action-token dynamics continuation, and a newer latent imagination planner. Several learned-simulator results are positive internally, but the strongest robust claims are still bounded to learned dynamics/reward evaluation rather than real-world control. The most recent long latent planner run produced useful positive signal but did not yet meet the strict action-causal threshold. The next agent should prioritize a decision-quality audit harness and better "thinking data" rather than running more seed sweeps.

## Repository Context

Workspace:

```text
/home/mkrzus/kairos-sensenova
```

Important local repos:

```text
dreamer4/     unofficial PyTorch Dreamer4 reproduction
dreamerv3/    official DreamerV3 reference
Dream-VLX/    cloned Dream-VL/Dream-VLA style repo for lessons
kairos/       Kairos/Sensenova stack
sensenova_drone_agent/  project code, docs, scripts, outputs
```

The worktree is dirty and contains many untracked experiment files, outputs, repos, and generated artifacts. Do not run destructive git commands. Do not reset or clean the workspace.

## Do Not Persist Secrets

The user provided a Hugging Face token during the work. Do not write that token into docs, scripts, shell history, config files, or commits. If authentication is needed again, ask the user to provide a fresh token or use the existing local cache if present.

## Conceptual History

### Dreamer4 Reading

The Dreamer4 paper motivated the project. The relevant Dreamer4 structure is:

```text
Phase 1: train tokenizer on video.
Phase 2: train action-conditioned world model / dynamics.
Phase 3: add policy, reward, and value heads.
Phase 4: train policy in imagination inside the frozen world model.
```

Key Dreamer4 design details that influenced this repo:

- The dynamics model is trained with action tokens from the start.
- Agent tokens can read world/action/task tokens, but world tokens cannot read agent/task tokens. This avoids causal confusion.
- Behavior cloning trains action and reward heads before RL.
- Imagination RL freezes the world model and updates policy/value heads.
- PMPO-style policy training uses the sign of advantage and a behavior prior KL.
- The paper's `K=4` refers to shortcut/diffusion sampling steps per generated frame, not imagination horizon.
- Minecraft success required long sequences, but the Dreamer4 robotics appendix used SOAR with 7D actions and lower FPS.

### Important Interpretation

Dreamer4 is not just "add action heads after pretraining." The core difference is that the dynamics model itself sees actions during training. Our early attempts tried to retrofit action conditioning after the fact onto frozen visual latents, which repeatedly produced scene-motion priors with weak action causality.

### Dream-VLA / Dream-VLX Lessons

Dream-VLA/Dream-VLX was inspected after the user found a related paper and Hugging Face artifacts. The important lessons:

- VLA success often comes from clean action-chunk prediction, not generic latent world-model imagination.
- LIBERO/Dream-VLA pipelines filter no-op transitions.
- They predict raw executable action chunks, not expanded handcrafted dynamics features.
- Flow matching or diffusion over continuous action chunks is often stronger than naive MSE.
- Wrist camera and proprioception matter when available.
- Long schedules and curated robotics data matter.

Changes inspired by Dream-VLA/Dream-VLX:

- Split raw executable action outputs from expanded dynamics action features.
- Add action chunking.
- Add no-op filtering.
- Add reward-signal filtering.
- Add balanced data mixture sampling.
- Treat action-identifiability as a data property, not just a model objective.

## Data Sources Collected Or Wired

### SOAR

SOAR was downloaded from:

```text
https://rail.eecs.berkeley.edu/datasets/soar_release/soar-dataset-numpy.zip
```

Local artifact:

```text
sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip
```

Download summary from the successful download:

```text
content_length: 27181294108
size: 25.31 GiB
zip entries: 347703
trajectory count recorded in docs: 31812
```

SOAR provides:

```text
RGB video
7D relative end-effector actions
language task labels
success/failure labels
```

SOAR was used for:

- Behavior cloning midtraining.
- Reward/value labels using success/failure.
- Action-conditioned dynamics probes.
- Native Dreamer4-style tokenizer/dynamics training.
- SOAR-only native closed-loop imagination repeats.

### Dreamer4 Hugging Face Dataset

Local path:

```text
sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4
```

Docs record:

```text
size: 28.79 GiB
splits: expert, mixed-small, mixed-large
```

Frame shards:

```text
sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full
```

This dataset was useful for action/reward/value machinery and control-task diversity, but it is not a direct SOAR/Minecraft reproduction.

### Game Action Sources

Purpose: create small action-identifiable pixel/action sequences where wrong game actions should visibly imply wrong futures.

Doc:

```text
sensenova_drone_agent/docs/GAME_ACTION_DREAMER4_DATA.md
```

Result: the small game-action domain was the first place where native Dreamer4-style dynamics both beat persistence and showed action dependence in autoregressive evaluation. It validated plumbing but was too small/specialized for the main robotics paper claim.

### RoboNet

RoboNet was wired as an extra robot video/action source. In our export it mostly has zero reward placeholders, so it should be used for action-conditioned dynamics or anti-forgetting, not reward claims.

Relevant docs:

```text
sensenova_drone_agent/docs/ROBOTICS_DATA_BRIDGE.md
sensenova_drone_agent/scripts/export_robonet_dreamer4_dataset.py
```

### LeRobot / Hugging Face Robotics Sources

Downloader:

```text
sensenova_drone_agent/scripts/download_robot_action_hf_datasets.py
```

Exporter:

```text
sensenova_drone_agent/scripts/export_lerobot_hf_dreamer4_dataset.py
```

Local exports include:

```text
sensenova_drone_agent/data/robotics/hf_action_exports/droid_lerobot_dreamer4
sensenova_drone_agent/data/robotics/hf_action_exports/fractal20220817_data_lerobot_dreamer4
sensenova_drone_agent/data/robotics/hf_action_exports/bridge_orig_lerobot_dreamer4
```

DROID became important for scaling, but the DROID-only action-identifiability audit found weak incremental action signal beyond scene history and no usable reward signal in our export.

## Core Code Map

### Dreamer4-Style Native World Model

Patched local Dreamer4 code:

```text
dreamer4/dreamer4/wm_dataset.py
dreamer4/dreamer4/train_tokenizer.py
dreamer4/dreamer4/train_dynamics.py
```

Important additions:

- Local task loading.
- Optional wandb fallback.
- Clean `max_steps` exits.
- Action frame offset support.
- Expanded action features.
- Raw action preservation.
- Action contrast losses.
- Visual-delta and no-op filtering.
- Closed-loop rollout auxiliary loss.
- Closed-loop action counterfactual negatives.

### SOAR Export And Evaluation

```text
sensenova_drone_agent/scripts/export_soar_dreamer4_dataset.py
sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py
sensenova_drone_agent/scripts/select_dreamer4_soar_dynamics_checkpoint.py
```

### Behavior Cloning / Midtraining

```text
sensenova_drone_agent/src/sensenova_drone/midtraining.py
sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py
sensenova_drone_agent/scripts/run_soar_midtraining_validation_suite.py
```

Docs:

```text
sensenova_drone_agent/docs/BEHAVIOR_CLONING_MIDTRAINING.md
```

### Action-Conditioned Latent Dynamics

```text
sensenova_drone_agent/scripts/train_action_conditioned_latent_dynamics.py
sensenova_drone_agent/scripts/run_action_conditioning_gate.py
```

Doc:

```text
sensenova_drone_agent/docs/ACTION_CONDITIONED_LATENT_DYNAMICS.md
```

### Native Dreamer4 Imagination

```text
sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py
sensenova_drone_agent/scripts/run_native_dreamer4_imagination_repeatability.py
sensenova_drone_agent/scripts/experiments/launch_game_actions_native_imagination.sh
sensenova_drone_agent/scripts/experiments/launch_all_data_balanced_imagination.sh
sensenova_drone_agent/scripts/experiments/launch_soar_residual_adapter_imagination.sh
```

### Residual Action Adapter

```text
sensenova_drone_agent/scripts/train_residual_action_adapter.py
sensenova_drone_agent/scripts/select_residual_adapter_checkpoint.py
sensenova_drone_agent/scripts/promote_controllable_soar_simulator.py
sensenova_drone_agent/scripts/promote_soar_imagination_policy.py
sensenova_drone_agent/scripts/residual_adapter_runtime.py
```

Doc:

```text
sensenova_drone_agent/docs/SOAR_RESIDUAL_SIMULATOR_IMAGINATION.md
```

### Latent Imagination Planner

```text
sensenova_drone_agent/scripts/train_latent_imagination_planner.py
sensenova_drone_agent/scripts/experiments/launch_latent_imagination_planner.sh
sensenova_drone_agent/scripts/experiments/latent_imagination_planner_payload.sh
```

Docs:

```text
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ADJUSTED_PLAN.md
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ALGORITHM.md
```

## Experiment Chronology And Results

### 1. Drone/PyBullet Toy Control

The repo initially built a closed-loop drone-game/PyBullet scaffold. This proved that the training/evaluation plumbing could collect transitions, train BC/DQN/world-model policies, and evaluate real simulated outcomes.

Strongest early toy result from `PAPER_READINESS.md`:

```text
Frozen world-model encoder DQN:
  success_rate: 0.671875
  collision_rate: 0.234375
  timeout_rate: 0.109375
  mean_return: 21.0176

Frozen random encoder DQN:
  success_rate: 0.515625
  collision_rate: 0.375
  timeout_rate: 0.109375
  mean_return: 15.7519

CNN DQN from scratch, 6000 steps:
  success_rate: 0.390625
  collision_rate: 0.328125
  timeout_rate: 0.28125
  mean_return: 15.9655
```

Claim boundary: this supports "predictive visual features can help toy control." It does not support Kairos/Sensenova direct drone autonomy.

### 2. SOAR Behavior Cloning And Reward/Value Midtraining

SOAR caches and midtraining were built in stages:

```text
SOAR RGB-flat smoke/medium caches
SOAR Kairos/Wan VAE-flat smoke/medium caches
SOAR task-balanced 512 cache
SOAR stride-8 summed-action caches
SOAR success/failure reward labels
```

Important result: SOAR Kairos/Wan VAE-flat medium behavior cloning predicted actions better than simple controls, but strict Dreamer-style validation did not fully pass.

Examples from `TRAINING_STATUS.md`:

```text
SOAR Kairos/Wan VAE-flat medium best action MSE: 0.4698
shuffle_targets best action MSE: 0.8158
mean-action control val action MSE: 0.8265
repeat-last-action control val action MSE: 0.9443
```

Reward labels:

```text
reward labels source: success.txt + language_task.txt
recommended reward mode: trajectory_success
```

But strict phase-2 controls remained weak:

```text
SOAR strict phase-2 controls passed: false
SOAR strict phase-2 reward/value ready: false
current SOAR phase-2 conclusion: architecture close to target, but data/reward signal not strong enough for unrestricted imagination RL
```

### 3. Early Action-Conditioned Dynamics Probes

Synthetic action-driven latent dynamics passed:

```text
normal best val z MSE: 0.1564
shuffled future-action best val z MSE: 0.3380
zero future-action best val z MSE: 0.3371
persistence MSE: 0.5135
```

Interpretation: the dynamics architecture can use future action tokens when the data has real action signal.

SOAR Kairos/Wan VAE-flat probe failed:

```text
normal best val z MSE: 0.141892
shuffled future-action best val z MSE: 0.141896
zero future-action best val z MSE: 0.141890
persistence MSE: 0.141856
```

Interpretation: the cache/window/action setup was persistence-dominated. Future actions did not matter.

Action offset gate failed:

```text
offsets tested: -2, -1, 0, 1, 2
passed_offsets: []
best normal/persistence ratio: 1.000273
best shuffle/normal ratio: 1.000047
best zero/normal ratio: 1.000012
```

RGB-flat sanity also failed:

```text
normal/persistence ratio: 1.019120
shuffle/normal ratio: 0.999323
zero/normal ratio: 0.997915
```

### 4. Temporal Aggregation And SOAR Short-Horizon Retrofit

Frame stride and summed actions exposed weak real signal.

Stride-8 summed-action RGB gate:

```text
normal_vs_persistence_ratio: 0.918304
shuffle_vs_normal_ratio: 1.049823
zero_vs_normal_ratio: 1.040453
ready_for_bc_or_imagination: false
```

The best frozen-Kairos retrofit baseline:

```text
sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1
```

Reference metrics:

```text
single-pass normal/persistence: 0.852
single-pass shuffled/normal: 1.092
single-pass zero/normal: 1.628

autoregressive h4 normal/persistence: 0.922
autoregressive h4 shuffled/normal: 1.058
autoregressive h4 zero/normal: 2.668

autoregressive h8 normal/persistence: 0.947
autoregressive h8 shuffled/normal: 1.091
autoregressive h8 zero/normal: 6.197

autoregressive h16 normal/persistence: 1.061
autoregressive h16 shuffled/normal: 1.087
autoregressive h16 zero/normal: 19.445
```

Interpretation: short-horizon h4/h8 replanning was plausible; h16 open-loop imagination was not.

### 5. Retrofit Knobs That Did Not Work

The following were tried and did not fix action grounding:

- Naive autoregressive rollout MSE.
- Longer direct prediction horizon.
- Longer context length.
- Aggressive rollout contrast.
- Conservative rollout contrast.
- Alignment offset sweeps.
- Action-window mean sweeps.
- Action-query tokens alone.
- Action-gated residual alone under the early cache.

From `SOAR_RETROFIT_DYNAMICS_ABLATIONS.md`:

```text
alignment/window sweeps did not reveal a simple action-frame lag bug.
action-query tokens alone did not improve action grounding.
action-gated residual created weak action sensitivity, but autoregressive rollouts were worse than persistence.
none beat the existing h4/h8 baseline.
```

### 6. Native Dreamer4-Style SOAR Training

Native SOAR v1:

```text
output: sensenova_drone_agent/output/dreamer4_soar_native_v1
selected trajectories: 127
tasks: 16
exported steps: 6066
tokenizer steps: 3000
dynamics steps: 5000
```

Eval:

```text
direct normal MSE: 0.0233
direct shuffled/normal: 1.0004
direct zero/normal: 0.9996
AR normal/persistence: 0.2061
AR shuffled/normal: 1.0014
AR zero/normal: 0.9730
```

Interpretation: learned visual dynamics and beat persistence, but not action identity.

Native SOAR v2 action contrast:

```text
output: sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast
```

Offsets all failed. Example best-ish rows:

```text
offset 0: h8 normal/persistence 0.1674, shuffled/normal 0.9951, zero/normal 0.9675
offset -1: h8 normal/persistence 0.1695, shuffled/normal 0.9958, zero/normal 0.9688
```

Interpretation: strong latent prediction, still scene-motion prior.

### 7. Game-Action Native Dynamics And Imagination

Game-action data provided cleaner action causality. Native dynamics passed action-conditioned rollout gates in a lightweight game domain.

But the policy/reward/imagination result was weak:

```text
after policy-minus-BC: +0.0004
policy return delta: +0.0274
after policy-minus-zero: -0.0234
```

Interpretation: plumbing worked, but reward/task quality was not sufficient for a strong agent claim.

### 8. Raw Action Split And Reward Filtering

The policy was changed to output raw executable action chunks while the dynamics still consumed expanded action features.

Validated smoke:

```text
policy_action_source: raw
action_chunk_len: 4
runtime/shape validation: pass
```

Full raw-policy run:

```text
action_mse step 1200: 0.0489
after policy-minus-BC: -0.0009
policy return delta: -0.0080
```

Reward-filtered raw-policy run:

```text
filtered data size: 163 windows
before policy-minus-BC: +0.0533
after policy-minus-BC: +0.0066
after policy-minus-zero: +0.4292
policy return delta: -0.0466
```

Interpretation: reward-clear windows make agent heads meaningful, but the tiny filtered set is too small and the RL update still degraded the pre-imagination policy.

### 9. Balanced All-Data Imagination And PMPO Score-Function Fix

Balanced data sampler:

```text
hf_expert_positive
hf_mixed_positive
hf_mixed_zero
soar_game_positive
```

Balanced no-update midtraining:

```text
zero-action learned return: 1.2405
BC-prior learned return: 1.3787
policy learned return: 1.3530
policy-minus-BC: -0.0256
policy-minus-zero: +0.1125
```

Action-conditioned dynamics gate at 20k:

```text
direct shuffle/normal: 1.0248
direct zero/normal: 1.0036
AR normal/persistence: 0.9164
AR shuffle/normal: 1.0324
AR zero/normal: 0.9763
native_dynamics_ready_for_imagination: true under the then-current gate
```

Initial balanced imagination with gated dynamics:

```text
before policy-minus-BC: +0.0080
after policy-minus-BC: +0.0007
policy_return_delta: -0.0073
```

Best-selection with low LR selected update 0, so it prevented harm but did not improve.

Critical bug found:

```text
Old PMPO used log_prob(dist.rsample()) without detaching the sampled action.
For Normal distributions, log_prob(mean + std * eps) cancels much of the score-function gradient to the mean.
Fix: log_prob = dist.log_prob(raw_action_flat.detach())
```

After score-function fix:

```text
selected update: 80
zero-action learned return: 4.8531
BC-prior learned return: 4.9634
policy learned return: 4.9859
policy-minus-BC: +0.0225
policy-minus-zero: +0.1329
policy_return_delta: +0.0145
policy_prior_mse_delta: -0.0002
```

Repeatability:

```text
seeds: 20260518, 20260519, 20260520
strict passes: 2/3
repeatability_pass: true
mean policy-minus-BC: +0.0187
mean policy-minus-zero: +0.1617
mean policy-return-delta: +0.0075
```

Causal ablation warning:

```text
Zeroing or shuffling actions sent to dynamics during RL still preserved most selected gain.
Therefore this result is not clean proof that exact policy actions causally drive imagined futures.
```

### 10. Residual Action Adapter Track

The residual simulator track selected a residual adapter on top of frozen continued dynamics.

Promoted simulator:

```text
sensenova_drone_agent/output/controllable_soar_simulator_v1/manifest.json
```

Selected adapter:

```text
sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt
```

Selection summary:

```text
sensenova_drone_agent/output/residual_action_adapter_selection_v1/selection_summary.json
```

Key held-out metrics:

```text
AR normal/persistence: 0.5390
AR effect-far-shuffle/normal: 1.0252
Direct effect-far-shuffle/normal: 1.0030
AR zero/normal: 22.4219
AR temporal-min/normal: 29.7049
```

Promoted DROID-heavy hard-gated policy:

```text
sensenova_drone_agent/output/controllable_soar_imagination_policy_v1/manifest.json
```

Main run:

```text
sensenova_drone_agent/output/soar_residual_adapter_imagination_causal_hardgate_droidheavy_v1
```

Selected update:

```text
update: 500
policy-minus-BC: +0.0088
policy-minus-zero: +0.1661
policy-minus-dyn-zero: +0.1177
policy-minus-dyn-shuffle: +0.0021
policy_return_delta: +0.0223
```

Post-RL-only repeatability closed the update-0 loophole:

```text
seeds: 20260605, 20260606, 20260607
pass count: 3/3
mean policy-minus-BC: +0.0048
mean policy-minus-dyn-shuffle: +0.0062
```

Promoted post-RL artifact:

```text
sensenova_drone_agent/output/controllable_soar_imagination_policy_postrl_v1/manifest.json
```

Claim boundary:

```text
This is a learned-simulator policy improvement result.
It is not real-world control.
It is not native Dreamer4-equivalent action-token dynamics.
```

### 11. Native Action-Token Continuation

Action-focus continuation to 325k:

```text
output: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_actionfocus_m1_to325k_v2
final checkpoint: dynamics_ckpts/final_step_0325000.pt
```

Eval:

```text
direct shuffle/normal: 1.048
direct zero/normal: 1.043
direct time_shift/normal: 1.056
AR normal/persistence: 0.607
AR shuffle/normal: 1.148
AR zero/normal: 1.362
AR time_shift/normal: 1.011
AR time_shift2/normal: 1.018
strict gate: false
```

Interpretation: direct action conditioning improved, but autoregressive temporal precision failed.

Closed-loop continuation to 375k:

```text
output: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_closedloop_m1_v1
final checkpoint: dynamics_ckpts/final_step_0375000.pt
```

Eval:

```text
direct shuffle/normal: 1.328
direct zero/normal: 1.087
direct time_shift/normal: 6.316
direct time_shift2/normal: 7.183
direct far_shuffle/normal: 1.487

AR normal/persistence: 0.314
AR shuffle/normal: 1.803
AR zero/normal: 1.350
AR time_shift/normal: 5.406
AR time_shift2/normal: 6.465
AR far_shuffle/normal: 2.179

strict_gate_passed: true
native_dynamics_ready_for_imagination: true
```

Interpretation: this was the first native action-token dynamics checkpoint that passed the strict autoregressive causal gate without the residual adapter.

### 12. Native Closed-Loop Imagination

Native-only closed-loop post-RL:

```text
output: sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_postrl_v1
```

Selected update:

```text
update: 300
policy-minus-BC: +0.0151
policy-minus-zero: -0.0035
policy-minus-dyn-zero: -0.0087
policy-minus-dyn-shuffle: +0.0021
```

Interpretation: improved over BC and shuffle, but lost to zero/dyn-zero.

Native closed-loop zero-aware:

```text
output: sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_postrl_v1
```

Selected update:

```text
update: 350
policy-minus-BC: +0.0369
policy-minus-zero: +0.0640
policy-minus-dyn-zero: +0.0026
policy-minus-dyn-shuffle: +0.0136
policy_return_delta: +0.0200
```

This was the first native-only checkpoint to pass the default-horizon hard gate without residual adapter.

Repeatability:

```text
including original zero-aware run: 3/5 pass
new repeat seeds only: 2/4 pass
all 5/5 improved over BC
failures were zero/dyn-zero causal-control failures
```

### 13. Native SOAR-Only Strict Sweep

SOAR-only strict run:

```text
output: sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_soaronly_strict_seed_20260536
selected update: 800
policy-minus-BC: +0.0278
policy-minus-zero: +0.2103
policy-minus-dyn-zero: +0.0964
policy-minus-dyn-shuffle: +0.0027
```

Repeat seed that survived larger eval:

```text
output: sensenova_drone_agent/output/soar_residual_adapter_imagination_native_closedloop_zeroaware_soaronly_strict_repl_seed_20260604
selected update: 750
policy-minus-BC: +0.0176
policy-minus-zero: +0.2240
policy-minus-dyn-zero: +0.1226
policy-minus-dyn-shuffle: +0.0058
```

Overall SOAR-only repeat result:

```text
regular-LR SOAR-only pass count: 3/6
including low-LR diagnostic: 3/7
```

Decision note from docs:

```text
Do not chase lucky seeds.
Evidence is real but fragile.
Use strongest native checkpoints as ablations, not the main headline.
Next main push should improve action-causal simulator itself.
```

### 14. DROID Audit

DROID-only identifiability audit:

```text
output: sensenova_drone_agent/output/action_identifiability_audit_droid_only_v1
sampled windows: 4096
available DROID windows: 398079
active action dims: 29
nonzero action fraction: 0.913
reward mean/std/positive fraction: 0.0 / 0.0 / 0.0
best action incremental R2 vs scene: -0.0058
best action-only R2 vs mean: -0.0097
data_action_signal_detected: false
```

Interpretation: DROID actions are present, but our current export lacks reward labels and the audit does not find incremental action-predictive visual dynamics beyond scene history. Do not make DROID-specific reward/imagination claims until labels or stronger action-effect targets are added.

### 15. Latent Imagination Planner Pivot

The current adjusted algorithm shifts from direct action-conditioned simulation to latent future proposal and decision-quality measurement.

Docs:

```text
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ADJUSTED_PLAN.md
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ALGORITHM.md
```

Model structure:

```text
frozen tokenizer
plan encoder: context + future actions -> latent plan token
future proposer: context + plan token -> future latents
trajectory scorer: context + future + plan -> return/value
inverse dynamics: context + future + plan -> action chunk
contrast losses: true actions vs zero/shuffle/time-shift/permutation/reversal
candidate sampling: sample plan tokens, score candidates, select best
```

Training script:

```text
sensenova_drone_agent/scripts/train_latent_imagination_planner.py
```

Launch script:

```text
sensenova_drone_agent/scripts/experiments/launch_latent_imagination_planner.sh
```

Long run:

```text
container: sda-latent-imagination-planner-all_data_v1
output: sensenova_drone_agent/output/latent_imagination_planner_all_data_v1
manifest: output/dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/all_data_manifest.json
dataset windows: 13,438,694
tasks: 103
max_steps: 500000
```

The GPU/CPU died twice. Last durable checkpoint:

```text
sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt
checkpoint step: 130000
```

Latest logged train row before crash:

```text
step: 139800
loss: 0.6103
future_loss: 0.00759
reward_loss: 2.2118
inverse_loss: 0.09187
shuffle_over_normal: 3.2536
zero_over_normal: 5.8477
time_shift_over_normal: 1.0504
time_shift2_over_normal: 1.0835
time_perm_over_normal: 1.0994
time_reverse_over_normal: 1.1208
```

Latest eval before crash:

```text
step: 139000
future_mse: 0.01285
inverse_mse: 0.16181
score_mse: 15.6136
score_return_corr: 0.60683
candidate_selected_minus_random: 0.18754
shuffle_over_normal: 1.56006
zero_over_normal: 4.22977
time_shift_over_normal: 1.02836
time_shift2_over_normal: 1.04278
time_perm_over_normal: 1.04383
time_reverse_over_normal: 1.10153
```

Interpretation:

```text
Positive signal exists.
score_return_corr is good.
zero-action contrast is strong.
candidate selected beats random, but margin is below preferred threshold.
shuffle and temporal contrasts remain below paper-grade action-causal thresholds.
This is working but not conclusive.
```

Recommended resume command:

```bash
cd /home/mkrzus/kairos-sensenova

RUN_ID=all_data_v1_resume_130k_save1k \
RESUME_CKPT=/workspace/sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt \
SAVE_EVERY=1000 \
TRACE_EVERY=1000 \
EVAL_EVERY=1000 \
REQUIRE_VISUAL_DELTA=0 \
GPU_SELECTOR=device=0 \
bash sensenova_drone_agent/scripts/experiments/launch_latent_imagination_planner.sh
```

Use `SAVE_EVERY=1000` or `2000`; the previous `10000` interval lost around 9800 logged steps after the crash.

## Metrics And Gates

### Dynamics Causality Gates

For dynamics/world-model claims:

```text
normal future prediction must beat persistence
wrong actions must be worse than true actions
zero actions must be worse than true actions
time-shifted actions must be worse than true actions
far-shuffled actions must be worse than true actions
```

Typical ratio interpretation:

```text
wrong_over_normal <= 1.00: no action causality
1.00-1.05: weak or ambiguous
1.05-1.20: positive but modest
>1.20: strong
>2.00: very strong
```

For strict native gates we often used:

```text
causal_min_ratio: 1.02
```

### Policy / Imagination Gates

For learned-simulator policy claims:

```text
policy_minus_bc > 0
policy_minus_zero > 0
policy_minus_dyn_zero >= margin
policy_minus_dyn_shuffle >= margin
policy_return_delta >= 0 if claiming RL improvement over pre-imagination
```

Hard margin often used:

```text
CAUSAL_POLICY_MIN_MARGIN=0.002
```

### Latent Planner Gates

For the new latent imagination planner:

```text
score_return_corr >= 0.60 consistently
candidate_selected_minus_random >= 0.20, preferably >= 0.25
shuffle_over_normal >= 2.0
zero_over_normal >= 3.0 to 4.0
future_mse stable, not much worse than baseline
inverse_mse improving or reported as inverse R2
```

Current latest eval:

```text
score_return_corr: 0.6068  good
candidate_selected_minus_random: 0.1875  positive but below target
shuffle_over_normal: 1.56  positive but not conclusive
zero_over_normal: 4.23  good
```

## Safe Claims Today

These are defensible with current artifacts:

```text
1. The repo implements a Dreamer-style offline learned-simulator training pipeline:
   tokenizer/dynamics, behavior cloning, reward/value heads, PMPO-style imagination,
   controls, checkpoint selection, and repeatability sweeps.

2. Frozen/patched Kairos/Dreamer-style latent spaces can support short-horizon learned
   simulator control stacks, but action grounding is fragile.

3. Closed-loop native action-token dynamics continuation substantially improves
   autoregressive action causality compared with teacher-forced action-focus continuation.

4. Residual-adapter learned-simulator policy training produced repeatable internal
   learned-dynamics gains under hard BC-plus-causal gates.

5. Native closed-loop SOAR-only action-token imagination can pass default-horizon
   learned-simulator gates in some seeds, but is not robust enough for a headline claim.

6. The latent planner has a positive decision-quality signal, but needs stronger
   decision audit and contrast metrics before it can be claimed as "thinking."
```

## Unsafe Claims Today

Do not claim:

```text
Kairos/Sensenova itself controls drones.
The system controls real drones.
The frozen Kairos generator is a reliable action-conditioned simulator.
The native Dreamer4-style world model is robustly action-causal across all sources.
The latent planner's imagination is conclusively useful.
DROID-specific reward/imagination results are meaningful with the current export.
Long-horizon h16+ open-loop action control is solved universally.
Seed sweeps prove robustness if they only find a lucky checkpoint.
```

## Major Failure Modes To Preserve

### Scene-Motion Prior

Many models learn:

```text
context -> likely future
```

instead of:

```text
context + action -> action-specific future
```

This appears when shuffled/zero/time-shifted actions are nearly as good as true actions.

### Persistence Trap

Low future MSE can be meaningless if persistence already predicts well. Always report normal-over-persistence.

### Zero-Action Trap

Some policies beat BC but lose to zero-action controls. This means reward/value/dynamics are not proving useful action choice.

### Time-Shift Trap

Passing shuffle/zero is not enough. If neighboring time-shifted actions perform as well as true actions, exact action timing is not grounded.

### Update-0 Loophole

Best checkpoint selection can select the post-BC checkpoint before any imagination update. For claims about RL/imagination updates, enforce:

```text
MIN_IMAGINATION_SELECTION_UPDATE=100
```

### Lucky Seed Trap

Several native SOAR-only runs pass, but pass rate is low. Do not keep running seeds to find a prettier headline. Improve the simulator/objective/data.

### DROID Reward Trap

Our DROID export has no reward signal and weak incremental action signal under current audits. Do not train reward/value claims from DROID without fixing labels.

## Decision-Quality / Thinking Plan

The user and assistant converged on this:

```text
The question is not "can the model imagine?"
The question is "can imagined alternatives be evaluated well enough that choosing among them improves action?"
```

The next method should expose:

```text
context -> candidate futures/actions -> internal score/value -> selected candidate
```

Then measure:

```text
ranking quality
counterfactual sensitivity
calibration
depth benefit
ablation sensitivity
held-out source/task generalization
```

The minimum positive decision-quality signature:

```text
selected imagined plan > random imagined plan > shuffled-action plan
```

on held-out tasks, with confidence intervals.

## Next Agent: First 30 Minutes

1. Read this handoff.
2. Read the two active planning docs:

```text
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ADJUSTED_PLAN.md
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ALGORITHM.md
```

3. Inspect the latest latent planner metrics:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/metrics.jsonl')
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print(rows[-5:])
PY
```

4. Decide whether to resume the latent planner or implement the audit first. Prefer implementing the audit first if the goal is paper clarity.

## Next Agent: Highest-Value Implementation

Create:

```text
sensenova_drone_agent/scripts/eval_latent_imagination_decision_quality.py
```

Load:

```text
sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt
```

For each held-out context, construct candidate plans:

```text
true_plan:      plan_encoder(context, true future actions)
zero_plan:      plan_encoder(context, zero actions)
shuffle_plan:   plan_encoder(context, shuffled actions)
shift_plan:     plan_encoder(context, time-shifted actions)
reverse_plan:   plan_encoder(context, reversed actions)
sampled_plans:  randomly sampled latent plan tokens
```

For each candidate, record:

```text
candidate_score
candidate_future_mse_to_real_future
candidate_predicted_return
candidate_action_norm
candidate_future_delta_norm
candidate_type
candidate_rank
selected_candidate
source
task
context_id
```

Compute:

```text
oracle_score_margin = score(true_plan) - mean(score(wrong_plans))
oracle_future_margin = mean(mse(wrong_plans)) - mse(true_plan)
oracle_rank_pct = percentile rank of true_plan among candidates
oracle_top1 = true if true_plan has highest score
selected_vs_random_future_mse = mse(random_candidate) - mse(selected_candidate)
selected_vs_random_proxy_return = proxy_return(selected) - proxy_return(random)
candidate_count_sweep: K=1,4,8,16,32,64
horizon_sweep: H=4,8,16,32
```

Pass criteria:

```text
true plans rank above zero/shuffle/time-shift controls
selected candidate beats random under external proxy
candidate search improves with larger K
effect holds on held-out source/task splits
```

## Next Agent: Data Work

The next data need is not "more video" generically. It is paired or contrastive thinking data:

```text
same context + good action chunk -> progress future
same context + bad/no-op/time-shift action chunk -> stall or worse future
```

Build records like:

```json
{
  "context_id": "example",
  "positive_action_chunk": "path_or_index",
  "negative_action_chunk": "path_or_index",
  "positive_future": "path_or_index",
  "negative_future": "path_or_index",
  "label": "positive_better",
  "source": "soar",
  "task": "put red object on cloth"
}
```

Prefer:

- SOAR high-action/high-visual-delta windows.
- SOAR success/failure or progress-labeled windows.
- LIBERO if added later, because it has clean tasks and success labels.
- Bridge/Fractal only if labels and action effects are auditable.
- DROID only after reward/action-effect labels are fixed.

Avoid SMOTE over raw latents as the main fix. It fabricates invalid trajectory geometry and does not create causal action evidence.

## If Resuming The Long Planner

Use:

```bash
cd /home/mkrzus/kairos-sensenova

RUN_ID=all_data_v1_resume_130k_save1k \
RESUME_CKPT=/workspace/sensenova_drone_agent/output/latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt \
SAVE_EVERY=1000 \
TRACE_EVERY=1000 \
EVAL_EVERY=1000 \
REQUIRE_VISUAL_DELTA=0 \
GPU_SELECTOR=device=0 \
bash sensenova_drone_agent/scripts/experiments/launch_latent_imagination_planner.sh
```

Monitor:

```bash
docker logs -f sda-latent-imagination-planner-all_data_v1_resume_130k_save1k
tail -f sensenova_drone_agent/output/latent_imagination_planner_all_data_v1_resume_130k_save1k/logs/payload.log
```

If using both GPUs, launch separate run IDs and separate `GPU_SELECTOR=device=0` / `device=1`. Do not let two containers write to the same output directory.

## Recommended Paper Framing

The cleanest paper arc is:

```text
1. Dreamer4 teaches that action tokens must be native to dynamics for offline imagination.
2. Retrofitting frozen video latents often learns scene motion rather than action causality.
3. We build a battery of causal gates showing exactly where this fails.
4. Closed-loop action-token dynamics and residual adapters improve causal action usage.
5. Policy improvements inside learned simulators are possible but fragile.
6. The next robust abstraction is inspectable latent imagination planning:
   propose multiple futures, score them, select one, and audit whether selection improves decisions.
```

The headline should avoid "Kairos controls drones." A safer title direction:

```text
Auditing and Training Action-Grounded Imagination in Video World-Model Latent Spaces
```

or:

```text
From Scene Priors to Decision-Quality Imagination: Retrofitting Action Grounding into Video World Models
```

## Most Important Existing Docs

Read these in order:

```text
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ALGORITHM.md
sensenova_drone_agent/docs/LATENT_IMAGINATION_PLANNING_ADJUSTED_PLAN.md
sensenova_drone_agent/docs/SOAR_RESIDUAL_SIMULATOR_IMAGINATION.md
sensenova_drone_agent/docs/DREAM_VLX_ACTION_LESSONS.md
sensenova_drone_agent/docs/ACTION_WORLD_MODEL_DATA_COLLECTION.md
sensenova_drone_agent/docs/ACTION_CONDITIONED_LATENT_DYNAMICS.md
sensenova_drone_agent/docs/SOAR_RETROFIT_DYNAMICS_ABLATIONS.md
sensenova_drone_agent/docs/NATIVE_DREAMER4_IMAGINATION_RESULT.md
sensenova_drone_agent/docs/TRAINING_STATUS.md
```

## Final Recommendation

Do not spend the next block of work on another massive blind training run. The next paper-critical step is measurement:

```text
Build the decision-quality audit.
Show whether chosen imagined candidates beat controls.
Only then decide whether to resume/scalably train the planner.
```

If the audit passes, scale the planner and add visual trace decoding. If the audit fails, collect/construct stronger contrastive thinking data before training more.
