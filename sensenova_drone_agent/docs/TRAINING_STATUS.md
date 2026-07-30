# PyBullet Imagination Training

## Result
- Learned latent simulator scaffold implemented: true
- Real PyBullet transition collection: true
- Behavior-cloned prior actor: true
- Frozen-dynamics imagined actor updates: true
- PMPO-style imagined policy update: true
- Real PyBullet eval before and after imagination: true
- Regression-safe checkpoint selection: true
- Pixel temporal feature stack: true
- Kairos/Wan VAE-flat feature path: true
- Cached transition dataset path: true
- Fixed evaluation seeds: true
- Visual/Kairos suite runner: true
- GPU PyBullet/Kairos image: true
- Kairos CUDA VAE encoding: true
- Dreamer4-style ad hoc action-token dynamics: true
- Action-shuffle sensitivity probe: true
- Dreamer4-style sequence-window dynamics training: true
- Transition cache episode/step metadata: true
- Robotics dataset inspection bridge: true
- Resumable SOAR numpy downloader: true
- SOAR numpy zip downloaded: true
- Dreamer4 Hugging Face dataset downloaded: true
- Dreamer4 Hugging Face dataset path: `sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4`
- Dreamer4 Hugging Face dataset size: `28.79 GiB`
- Dreamer4 Hugging Face dataset splits: `expert,mixed-small,mixed-large`
- SOAR zip-native sequence cache export: true
- SOAR Kairos/Wan VAE-flat sequence cache export: true
- Phase-2 BC midtraining scaffold: true
- Phase-2 BC midtraining smoke train: true
- Phase-2 SOAR BC midtraining smoke train: true
- Phase-2 SOAR Kairos/Wan VAE-flat smoke train: true
- Action-conditioned latent dynamics scaffold: true
- Action-conditioning gate runner: true
- SOAR temporal action aggregation export: true
- SOAR stride-8 RGB action signal: weak_positive
- SOAR stride-8 summed-action Kairos/Wan cache rebuilt: true
- SOAR stride-8 summed-action Kairos/Wan cache steps: `8281`
- SOAR stride-8 summed-action Kairos/Wan cache episodes: `489`
- SOAR stride-8 summed-action Kairos/Wan cache tasks: `64`
- SOAR stride-8 summed-action Kairos/Wan cache action_mean_abs: `0.7167`
- SOAR large MLP-head dynamics scaffold: true
- SOAR large MLP-head context-16 anchors: `580`
- SOAR large MLP-head context-16 strict gate: `failed`
- SOAR large MLP-head context-16 normal_over_persistence: `1.2361`
- SOAR large MLP-head context-16 shuffle_over_normal: `1.0012`
- SOAR large MLP-head context-16 zero_over_normal: `0.9695`
- SOAR large MLP-head context-8 diagnostic anchors: `1747`
- SOAR large MLP-head context-8 strict gate: `failed`
- SOAR large MLP-head context-8 normal_over_persistence: `1.1504`
- SOAR large MLP-head context-8 shuffle_over_normal: `1.0362`
- SOAR large MLP-head context-8 zero_over_normal: `0.9187`
- SOAR action-grounding controls added: true
- SOAR action-grounding future-action offset support: true
- SOAR action-grounding high-motion anchor filter: true
- SOAR action-grounding delta-z loss: true
- SOAR action-grounding contrastive true-vs-shuffled/zero loss: true
- SOAR action-grounding smoke strict gate: `failed`
- SOAR action-grounding smoke action_conditioning_strength: `weak`
- SOAR action-grounding smoke normal_over_persistence: `0.9964`
- SOAR action-grounding smoke shuffle_over_normal: `1.0635`
- SOAR action-grounding smoke zero_over_normal: `1.0522`
- SOAR action-grounded Kairos dynamics strict gate: `passed`
- SOAR action-grounded Kairos dynamics output: `output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3`
- SOAR action-grounded Kairos dynamics normal_over_persistence: `0.9398`
- SOAR action-grounded Kairos dynamics shuffle_over_normal: `1.0650`
- SOAR action-grounded Kairos dynamics zero_over_normal: `1.4083`
- SOAR action-grounded full BC/imagination output: `output/soar_dreamer_lite_action_grounding_full_ctx8_v1`
- SOAR action-grounded full BC/imagination ran: true
- SOAR action-grounded full BC/imagination final val imagined return0: `7.0429`
- SOAR action-grounded full BC/imagination dynamics gate after joint training: `failed`
- SOAR action-grounded full BC/imagination normal_over_persistence: `1.0041`
- SOAR joint dynamics+agent training can lose action grounding: true
- SOAR frozen-dynamics agent BC/imagination output: `output/soar_dreamer_lite_frozen_agent_bc_imagination_ctx8_v1`
- SOAR frozen-dynamics source checkpoint: `output/soar_dreamer_lite_action_grounding_contrastive_ctx8_v3/best_dynamics_bc.pt`
- SOAR frozen-dynamics gate before agent strict pass: `true`
- SOAR frozen-dynamics gate before agent normal_over_persistence: `0.9398`
- SOAR frozen-dynamics gate before agent shuffle_over_normal: `1.0630`
- SOAR frozen-dynamics gate before agent zero_over_normal: `1.4083`
- SOAR frozen-dynamics gate after agent strict pass: `true`
- SOAR frozen-dynamics gate after agent normal_over_persistence: `0.9398`
- SOAR frozen-dynamics gate after agent shuffle_over_normal: `1.0611`
- SOAR frozen-dynamics gate after agent zero_over_normal: `1.4083`
- SOAR frozen-dynamics agent BC best val action MSE: `0.8751`
- SOAR frozen-dynamics agent BC final val action MSE: `1.1006`
- SOAR frozen-dynamics agent BC final val reward MSE: `0.6019`
- SOAR frozen-dynamics agent BC final val value MSE: `48.8521`
- SOAR frozen-dynamics imagination final val return0: `3.8839`
- SOAR frozen-dynamics imagination final val prior MSE: `0.2533`
- SOAR split dynamics/agent protocol fixed dynamics regression: true
- SOAR frozen agent BC overfit observed: true
- SOAR regularized frozen agent run output: `output/soar_dreamer_lite_frozen_agent_bc_regularized_ctx8_v1`
- SOAR regularized frozen agent val ratio: `0.20`
- SOAR regularized frozen agent dropout: `0.25`
- SOAR regularized frozen agent learning rate: `0.00005`
- SOAR regularized frozen agent weight decay: `0.01`
- SOAR regularized frozen agent early stop triggered: `true`
- SOAR regularized frozen agent epochs run: `29`
- SOAR regularized frozen agent best epoch: `17`
- SOAR regularized frozen agent best val loss: `1.6022`
- SOAR regularized frozen agent best val action MSE: `0.9237`
- SOAR regularized frozen agent best val reward MSE: `0.4267`
- SOAR regularized frozen agent best val value MSE: `29.6599`
- SOAR regularized frozen agent gate before strict pass: `true`
- SOAR regularized frozen agent gate before normal_over_persistence: `0.8997`
- SOAR regularized frozen agent gate before shuffle_over_normal: `1.1001`
- SOAR regularized frozen agent gate before zero_over_normal: `1.4734`
- SOAR regularized frozen agent gate after strict pass: `true`
- SOAR regularized frozen agent gate after normal_over_persistence: `0.8997`
- SOAR regularized frozen agent gate after shuffle_over_normal: `1.1078`
- SOAR regularized frozen agent gate after zero_over_normal: `1.4734`
- SOAR regularized frozen imagination final val return0: `10.6079`
- SOAR regularized frozen imagination final val prior MSE: `0.8993`
- SOAR regularized frozen agent overfit reduced: true
- SOAR regularized reward/value calibration remains weak: true
- SOAR reward calibration controls added: true
- SOAR raw reward target mode added: true
- SOAR BCE reward head mode added: true
- SOAR raw discounted value target mode added: true
- SOAR Huber value loss mode added: true
- SOAR reward/value calibration metrics added: true
- SOAR task+outcome split mode added: true
- SOAR calibrated long run output: `output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2`
- SOAR calibrated long run motion filter quantile: `0.25`
- SOAR calibrated long run valid anchors: `1310`
- SOAR calibrated long run train anchors: `1046`
- SOAR calibrated long run val anchors: `264`
- SOAR calibrated long run agent BC epochs run: `63`
- SOAR calibrated long run agent BC early stop triggered: `true`
- SOAR calibrated long run best agent epoch: `3`
- SOAR calibrated long run best val loss: `1.3236`
- SOAR calibrated long run best val action MSE: `0.8372`
- SOAR calibrated long run reward Brier: `0.0472`
- SOAR calibrated long run reward accuracy: `0.9360`
- SOAR calibrated long run reward ECE@10: `0.0713`
- SOAR calibrated long run reward target mean: `0.0909`
- SOAR calibrated long run reward pred mean: `0.1316`
- SOAR calibrated long run value MSE: `3.2279`
- SOAR calibrated long run value MAE: `0.8658`
- SOAR calibrated long run value corr: `0.7208`
- SOAR calibrated long run dynamics strict gate: `true`
- SOAR calibrated long run normal_over_persistence: `0.8515`
- SOAR calibrated long run shuffle_over_normal: `1.0996`
- SOAR calibrated long run zero_over_normal: `1.6279`
- SOAR calibrated long run imagination epochs: `100`
- SOAR calibrated long run final val return0: `19.5141`
- SOAR calibrated long run final val reward: `0.1187`
- SOAR calibrated long run final val value MSE: `1.1887`
- SOAR calibrated long run final val prior MSE: `0.0116`
- SOAR calibrated reward/value scale problem improved: true
- SOAR imagination training mechanically ready: true
- SOAR imagination transfer proof pending: true
- SOAR learned-model transfer eval output: `output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2/soar_transfer_eval_h16`
- SOAR learned-model transfer eval rollout horizon: `16`
- SOAR learned-model transfer eval val anchors: `264`
- SOAR learned-model transfer eval strict dynamics gate: `true`
- SOAR learned-model transfer eval zero-action return: `0.8106`
- SOAR learned-model transfer eval BC prior return: `1.2432`
- SOAR learned-model transfer eval after-imagination return: `1.7367`
- SOAR learned-model transfer eval return delta after-minus-BC: `0.4935`
- SOAR learned-model transfer eval return ratio after-over-BC: `1.3970`
- SOAR learned-model transfer eval BC prior mean reward: `0.0791`
- SOAR learned-model transfer eval after-imagination mean reward: `0.1109`
- SOAR learned-model transfer eval after prior-plan MSE: `0.0091`
- SOAR learned-model transfer eval model-transfer improved: `true`
- SOAR learned-model transfer eval prior constrained: `true`
- SOAR learned-model transfer eval BC open-loop action MSE: `0.8372`
- SOAR learned-model transfer eval after open-loop action MSE: `0.8438`
- SOAR learned-model transfer eval BC value MSE: `3.2279`
- SOAR learned-model transfer eval after value MSE: `328.5844`
- SOAR learned-model transfer policy improved under learned dynamics: true
- SOAR learned-model transfer value head drift observed: true
- SOAR Kairos/Wan VAE-flat action-offset gate: failed
- SOAR RGB-flat action-conditioning sanity gate: failed
- Policy/reward/value promotion from current SOAR cache: blocked pending value-head drift fix and external SOAR eval
- Latest visual/Kairos selected actor: `bc_prior`

## Current Result
- Output: `output/pybullet_drones_imagination_pmpo_deep_kinematic_v1`
- Feature: `kinematic`
- Transitions: `12288`
- BC prior success: `1.0`
- After-imagination success: `1.0`
- BC prior mean final distance: `0.0840m`
- After-imagination mean final distance: `0.0674m`

## Visual/Kairos Result
- RGB output: `output/pybullet_drones_imagination_rgb_stack_small_v2_stable`
- RGB feature: `rgb_downsample`
- RGB transitions: `3994`
- RGB BC prior success: `0.0`
- RGB after-imagination success: `0.1667`
- RGB BC prior mean final distance: `0.3547m`
- RGB after-imagination mean final distance: `0.3076m`
- Kairos output: `output/pybullet_drones_imagination_kairos_flat_small_v1`
- Kairos feature: `kairos_vae_flat`
- Kairos transitions: `512`
- Kairos BC prior success: `0.0`
- Kairos after-imagination success: `0.0`
- Kairos BC prior mean final distance: `0.3849m`
- Kairos after-imagination mean final distance: `0.3757m`

## Fixed-Seed Suite Smoke
- Suite runner: `scripts/run_pybullet_drones_visual_imagination_suite.py`
- RGB smoke output: `output/visual_suite_rgb_smoke_v1`
- RGB cache reuse verified: true
- RGB cached transitions: `32`
- RGB smoke selected actor: `bc_prior`
- RGB small fixed-seed output: `output/visual_suite_rgb_small_fixed_v1`
- RGB small fixed-seed transitions: `4096`
- RGB small eval seeds: `171030,171031,171032`
- RGB small BC prior success: `0.0`
- RGB small after-imagination success: `0.3333`
- RGB small BC prior mean final distance: `0.4409m`
- RGB small after-imagination mean final distance: `0.3232m`
- RGB small selected actor: `after_imagination`
- Kairos smoke output: `output/visual_suite_kairos_smoke_v1`
- Kairos smoke selected actor: `after_imagination`
- Kairos smoke BC prior mean final distance: `0.8433m`
- Kairos smoke after-imagination mean final distance: `0.8428m`
- Kairos CUDA smoke output: `output/visual_suite_kairos_cuda_smoke_v1`
- Kairos CUDA smoke cache reuse verified: true
- Kairos CUDA medium output: `output/visual_suite_kairos_cuda_medium_v1`
- Kairos CUDA medium transitions: `512`
- Kairos CUDA medium elapsed: `20.5s`
- Kairos CUDA medium eval seeds: `171050,171051,171052`
- Kairos CUDA medium BC prior success: `0.0`
- Kairos CUDA medium after-imagination success: `0.0`
- Kairos CUDA medium BC prior mean final distance: `0.4670m`
- Kairos CUDA medium after-imagination mean final distance: `0.4389m`
- Kairos CUDA medium v2 output: `output/visual_suite_kairos_cuda_medium_v2`
- Kairos CUDA medium v2 transitions: `512`
- Kairos CUDA medium v2 elapsed: `21.7s`
- Kairos CUDA medium v2 eval seeds: `171060,171061,171062`
- Kairos CUDA medium v2 BC prior success: `0.0`
- Kairos CUDA medium v2 after-imagination success: `0.0`
- Kairos CUDA medium v2 BC prior mean final distance: `0.5838m`
- Kairos CUDA medium v2 after-imagination mean final distance: `0.6447m`
- Kairos CUDA medium v2 selected actor: `bc_prior`
- Kairos CUDA medium v2 result: imagined actor rejected by real-sim regression gate
- Kairos concat probe output: `output/visual_suite_kairos_concat_probe_medium_v1`
- Kairos concat probe action_shuffle_loss_ratio: `1.002`
- Kairos concat probe action_effect_rms: `0.0178`
- Kairos action-token output: `output/visual_suite_kairos_action_token_medium_v1`
- Kairos action-token action_shuffle_loss_ratio: `1.146`
- Kairos action-token action_effect_rms: `0.1888`
- Kairos action-token selected actor: `bc_prior`
- Current action-token conclusion: improves dynamics action-sensitivity, but does not yet stabilize imagined policy improvement
- Kairos sequence output: `output/visual_suite_kairos_sequence_medium_v1`
- Kairos sequence transitions: `482`
- Kairos sequence windows: `426`
- Kairos sequence length: `8`
- Kairos sequence z_mse: `0.0948`
- Kairos sequence action_shuffle_loss_ratio: `1.0065`
- Kairos sequence sequence_action_shuffle_loss_ratio: `1.0042`
- Kairos sequence selected actor: `bc_prior`
- Current sequence conclusion: corrected training shape lowers latent MSE, but action sensitivity remains weak on current data
- Kairos sequence-context PMPO output: `output/visual_suite_kairos_sequence_context_medium_v1`
- Kairos sequence-context imagination_context_length: `8`
- Kairos sequence-context z_mse: `0.0898`
- Kairos sequence-context sequence_action_shuffle_loss_ratio: `1.0027`
- Kairos sequence-context BC prior mean final distance: `0.6387m`
- Kairos sequence-context after-imagination mean final distance: `0.6678m`
- Kairos sequence-context selected actor: `bc_prior`
- Current sequence-context conclusion: PMPO now rolls through the sequence dynamics interface, but the learned Kairos latent simulator still has weak action grounding
- Dreamer3 reference available: `dreamerv3/`
- Dreamer4 reference available: `dreamer4/`
- Robotics data bridge doc: `docs/ROBOTICS_DATA_BRIDGE.md`
- Robotics dataset inspector: `scripts/experiments/inspect_robotics_dataset.py`
- SOAR numpy downloader: `scripts/download_soar_numpy_dataset.py`
- Dreamer4 Hugging Face downloader: `scripts/download_soar_numpy_dataset.py --source dreamer4-hf`
- BC midtraining module: `src/sensenova_drone/midtraining.py`
- BC midtraining trainer: `scripts/train_behavior_cloning_midtraining.py`
- BC midtraining doc: `docs/BEHAVIOR_CLONING_MIDTRAINING.md`
- BC midtraining smoke output: `output/bc_midtraining_smoke/train`
- BC midtraining smoke valid anchors: `336`
- BC midtraining smoke val loss epoch 1: `2.4782`
- BC midtraining smoke val loss epoch 2: `1.5940`
- SOAR zip: `data/robotics/soar/soar-dataset-numpy.zip`
- SOAR zip entries: `347703`
- SOAR trajectory count: `31812`
- SOAR RGB-flat cache: `output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz`
- SOAR RGB-flat cache steps: `192`
- SOAR RGB-flat cache episodes: `6`
- SOAR RGB-flat cache tasks: `5`
- SOAR RGB-flat cache feature dim: `3072`
- SOAR RGB-flat cache action dim: `7`
- SOAR RGB-flat valid anchors, context=8/horizon=8: `96`
- SOAR BC smoke output: `output/soar_bc_midtraining_smoke/train`
- SOAR BC smoke valid anchors, context=8/horizon=4: `120`
- SOAR BC smoke val loss epoch 1: `0.8057`
- SOAR BC smoke val loss epoch 2: `0.7646`
- SOAR BC smoke val loss epoch 3: `0.7188`
- SOAR RGB-flat medium cache: `output/soar_sequence_cache_medium/soar_rgb32_32traj.npz`
- SOAR RGB-flat medium cache steps: `1024`
- SOAR RGB-flat medium cache episodes: `32`
- SOAR RGB-flat medium cache tasks: `23`
- SOAR RGB-flat medium valid anchors, context=8/horizon=8: `512`
- SOAR BC medium output: `output/soar_bc_midtraining_medium_rgb32/train`
- SOAR BC medium valid anchors, context=8/horizon=4: `640`
- SOAR BC medium best epoch: `3`
- SOAR BC medium best val loss: `1.3315`
- SOAR BC medium best val action MSE: `0.6572`
- SOAR BC medium last val action MSE: `0.6198`
- SOAR Kairos/Wan VAE-flat smoke cache: `output/soar_sequence_cache_kairos_smoke/soar_kairos_flat128_2traj16.npz`
- SOAR Kairos/Wan VAE-flat smoke steps: `32`
- SOAR Kairos/Wan VAE-flat smoke latent shape per frame: `[1, 16, 1, 16, 16]`
- SOAR Kairos/Wan VAE-flat smoke feature dim: `4096`
- SOAR Kairos/Wan VAE-flat smoke action dim: `7`
- SOAR Kairos/Wan VAE-flat smoke valid anchors, context=8/horizon=4: `8`
- SOAR Kairos/Wan VAE-flat smoke train output: `output/soar_bc_midtraining_kairos_smoke/train`
- SOAR Kairos/Wan VAE-flat smoke val loss epoch 1: `0.4646`
- SOAR Kairos/Wan VAE-flat smoke val loss epoch 2: `0.3994`
- SOAR Kairos/Wan VAE-flat medium cache: `output/soar_sequence_cache_kairos_medium/soar_kairos_flat128_32traj32.npz`
- SOAR Kairos/Wan VAE-flat medium steps: `1002`
- SOAR Kairos/Wan VAE-flat medium episodes: `32`
- SOAR Kairos/Wan VAE-flat medium tasks: `20`
- SOAR Kairos/Wan VAE-flat medium feature dim: `4096`
- SOAR Kairos/Wan VAE-flat medium action dim: `7`
- SOAR Kairos/Wan VAE-flat medium valid anchors, context=8/horizon=4: `620`
- SOAR Kairos/Wan VAE-flat medium train output: `output/soar_bc_midtraining_kairos_medium/train`
- SOAR Kairos/Wan VAE-flat medium best checkpoint: `output/soar_bc_midtraining_kairos_medium/train/best.pt`
- SOAR Kairos/Wan VAE-flat medium train anchors: `558`
- SOAR Kairos/Wan VAE-flat medium val anchors: `62`
- SOAR Kairos/Wan VAE-flat medium best epoch: `25`
- SOAR Kairos/Wan VAE-flat medium epoch 1 val loss: `1.1302`
- SOAR Kairos/Wan VAE-flat medium epoch 1 val action MSE: `0.7620`
- SOAR Kairos/Wan VAE-flat medium epoch 25 val loss: `0.8486`
- SOAR Kairos/Wan VAE-flat medium epoch 25 val action MSE: `0.4698`
- SOAR Kairos/Wan VAE-flat baseline report: `output/soar_bc_midtraining_kairos_medium_baselines/report.md`
- SOAR Kairos/Wan VAE-flat normal best val action MSE: `0.4698`
- SOAR Kairos/Wan VAE-flat shuffle_targets best val action MSE: `0.8158`
- SOAR Kairos/Wan VAE-flat shuffle_z_context best val action MSE: `0.6073`
- SOAR Kairos/Wan VAE-flat zero_z_context best val action MSE: `0.5440`
- SOAR Kairos/Wan VAE-flat zero_prev_actions best val action MSE: `0.5525`
- SOAR Kairos/Wan VAE-flat mean-action control val action MSE: `0.8265`
- SOAR Kairos/Wan VAE-flat repeat-last-action control val action MSE: `0.9443`
- SOAR reward labels source: `success.txt + language_task.txt`
- SOAR final_success best val reward MSE: `0.1951`
- SOAR final_success best val value MSE: `0.7770`
- SOAR trajectory_success cache: `output/soar_sequence_cache_kairos_medium_reward_modes/soar_kairos_flat128_32traj32_trajectory_success.npz`
- SOAR trajectory_success best val action MSE: `0.4281`
- SOAR trajectory_success best val reward MSE: `0.0015`
- SOAR trajectory_success best val value MSE: `0.0041`
- SOAR linear_success_progress best val action MSE: `0.4831`
- SOAR linear_success_progress best val reward MSE: `0.0048`
- SOAR linear_success_progress best val value MSE: `0.0661`
- SOAR reward-mode report: `output/soar_bc_midtraining_kairos_medium_reward_modes/report.md`
- SOAR Kairos/Wan VAE-flat large trajectory_success cache: `output/soar_sequence_cache_kairos_large/soar_kairos_flat128_128traj32_trajectory_success.npz`
- SOAR Kairos/Wan VAE-flat large trajectory_success steps: `4096`
- SOAR Kairos/Wan VAE-flat large trajectory_success episodes: `128`
- SOAR Kairos/Wan VAE-flat large trajectory_success tasks: `60`
- SOAR Kairos/Wan VAE-flat large trajectory_success valid anchors: `2560`
- SOAR Kairos/Wan VAE-flat large trajectory_success train anchors: `2304`
- SOAR Kairos/Wan VAE-flat large trajectory_success val anchors: `256`
- SOAR Kairos/Wan VAE-flat large trajectory_success checkpoint: `output/soar_bc_midtraining_kairos_large_trajectory_success/train/best.pt`
- SOAR Kairos/Wan VAE-flat large trajectory_success best epoch: `29`
- SOAR Kairos/Wan VAE-flat large trajectory_success best val action MSE: `0.4460`
- SOAR Kairos/Wan VAE-flat large trajectory_success best val reward MSE: `0.0012`
- SOAR Kairos/Wan VAE-flat large trajectory_success best val value MSE: `0.0072`
- SOAR Kairos/Wan VAE-flat large trajectory_success report: `output/soar_bc_midtraining_kairos_large_trajectory_success/report.md`
- SOAR midtraining theoretical validation suite: `scripts/run_soar_midtraining_validation_suite.py`
- SOAR trainer agent-token isolation: true
- SOAR trainer episode-heldout split: true
- SOAR trainer task-stratified episode split: true
- SOAR trainer positive-reward-only BC action loss: true
- SOAR trainer metric-specific checkpoints: true
- SOAR trainer early stopping: true
- SOAR task-balanced Kairos/Wan VAE-flat cache: `output/soar_sequence_cache_kairos_task_balanced_512/soar_kairos_flat128_512traj32_trajectory_success.npz`
- SOAR task-balanced cache steps: `16384`
- SOAR task-balanced cache episodes: `512`
- SOAR task-balanced cache tasks: `64`
- SOAR task-balanced cache valid anchors, context=8/horizon=8: `8192`
- SOAR strict MTP-8 validation, 128 trajectory task split: `output/soar_midtraining_validation_v2_task_split/report.md`
- SOAR strict MTP-8 validation, 128 trajectory result: controls not beaten reliably
- SOAR task-balanced 512 validation, all trajectories BC: `output/soar_midtraining_validation_v3_task_balanced_512/report.md`
- SOAR task-balanced 512 validation, all trajectories result: controls not beaten reliably
- SOAR task-balanced 512 regularized validation: `output/soar_midtraining_validation_v4_task_balanced_512_regularized/report.md`
- SOAR task-balanced 512 positive-BC validation: `output/soar_midtraining_validation_v5_task_balanced_512_positive_bc/report.md`
- SOAR task-balanced 512 positive-BC normal seed mean best BC action MSE: `0.7945`
- SOAR task-balanced 512 positive-BC normal seed std best BC action MSE: `0.0294`
- SOAR task-balanced 512 positive-BC positive-reward mean-action baseline: `0.8875`
- SOAR task-balanced 512 positive-BC positive-reward repeat-previous-action baseline: `1.2130`
- SOAR task-balanced 512 positive-BC shuffle_targets ratio vs normal: `1.022`
- SOAR task-balanced 512 positive-BC zero_prev_actions ratio vs normal: `1.069`
- SOAR task-balanced 512 positive-BC zero_z_context ratio vs normal: `0.996`

## Action-Conditioning Gate

Current decision:

```text
Do not promote the current SOAR Kairos/Wan or RGB-flat caches to policy/reward/value
midtraining or imagination RL yet.
```

Implemented gate:

```text
scripts/run_action_conditioning_gate.py
```

Gate criterion:

```text
normal future actions must beat persistence, shuffled future actions, and zero future actions.
```

SOAR Kairos/Wan VAE-flat offset gate:

```text
cache: output/soar_sequence_cache_kairos_task_balanced_512/soar_kairos_flat128_512traj32_trajectory_success.npz
output: output/soar_action_conditioning_gate_kairos_task_balanced_512_offsets
offsets: -2, -1, 0, 1, 2
ready_for_bc_or_imagination: false
passed_offsets: []
best_offset: -1
best_normal_vs_persistence_ratio: 1.000273
best_shuffle_vs_normal_ratio: 1.000047
best_zero_vs_normal_ratio: 1.000012
```

RGB-flat sanity gate:

```text
cache: output/soar_sequence_cache_medium/soar_rgb32_32traj.npz
output: output/soar_action_conditioning_gate_rgb32_medium
ready_for_bc_or_imagination: false
normal_vs_persistence_ratio: 1.019120
shuffle_vs_normal_ratio: 0.999323
zero_vs_normal_ratio: 0.997915
```

Temporal/action-aggregated RGB gates:

```text
stride4_sum_output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride4_sum
stride4_sum_ready: false
stride4_sum_normal_vs_persistence_ratio: 0.991179
stride4_sum_shuffle_vs_normal_ratio: 1.006619
stride4_sum_zero_vs_normal_ratio: 1.007921

stride8_sum_output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride8_sum_ctx4_e30
stride8_sum_ready: false
stride8_sum_normal_vs_persistence_ratio: 0.918304
stride8_sum_shuffle_vs_normal_ratio: 1.049823
stride8_sum_zero_vs_normal_ratio: 1.040453

stride8_motion_filter_output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride8_sum_ctx4_motion_filter_e30
stride8_motion_filter_ready: false
stride8_motion_filter_kept_anchors: 1475 / 3935
stride8_motion_filter_normal_vs_persistence_ratio: 0.919450
stride8_motion_filter_shuffle_vs_normal_ratio: 1.011234
stride8_motion_filter_zero_vs_normal_ratio: 1.014176
```

Interpretation:

```text
The dynamics head can use action tokens on synthetic action-driven data. Temporal aggregation and
stride 8 expose weak but real SOAR RGB action signal. However, zero future actions are still too
competitive, so the next blocker remains action grounding, not policy optimization.

The latest Kairos/Wan VAE-flat stride-8 summed-action cache and larger 512-hidden/4-layer/MLP-head
dynamics model did not fix this. The model fits train dynamics but fails validation against persistence,
and zero-action futures remain competitive. Do not resume policy/reward/value BC or imagination RL
from this cache.

Action-grounding controls are now implemented so we can test whether direct contrastive supervision,
delta-z prediction, motion-window filtering, and action/frame offset sweeps can make the true action
necessary for latent prediction.

Latest result: the dynamics-only action-grounding run passed the strict SOAR/Kairos dynamics gate.
However, the first joint policy/reward/value BC plus imagination run regressed the dynamics gate.
Next protocol should freeze the strict-gated dynamics checkpoint and train agent heads separately.
```

Next required work:

```text
Add explicit robot state/state deltas or use a dataset/environment where actions are more directly
coupled to first-person visual consequences. Re-run the action-conditioning gate before BC/RL
promotion.
```
- SOAR task-balanced 512 positive-BC training duration status: `overfit_after_best`
- SOAR task-balanced 512 positive-BC early stopping recommended: true
- SOAR task-balanced 512 positive-BC reward/value validated: false
- SOAR task-balanced 512 positive-BC checkpoint: `output/soar_bc_midtraining_kairos_task_balanced_512_positive_bc/train_seed2_earlystop/best_bc_action_mse.pt`
- Dreamer-style action-conditioned latent dynamics implemented: true
- Action-conditioned latent dynamics trainer: `scripts/train_action_conditioned_latent_dynamics.py`
- Action-conditioned latent dynamics doc: `docs/ACTION_CONDITIONED_LATENT_DYNAMICS.md`
- Synthetic action-driven dynamics smoke output: `output/latent_dynamics_smoke/action_driven_normal`
- Synthetic normal best val z MSE: `0.1564`
- Synthetic shuffled future-action best val z MSE: `0.3380`
- Synthetic zero future-action best val z MSE: `0.3371`
- Synthetic persistence MSE: `0.5135`
- Synthetic action-conditioning result: normal future actions clearly beat shuffled/zero future actions
- SOAR action-conditioned dynamics probe output: `output/soar_latent_dynamics_task_balanced_512_probe`
- SOAR action-conditioned dynamics normal best val z MSE: `0.141892`
- SOAR action-conditioned dynamics shuffled future-action best val z MSE: `0.141896`
- SOAR action-conditioned dynamics zero future-action best val z MSE: `0.141890`
- SOAR action-conditioned dynamics persistence MSE: `0.141856`
- SOAR action-conditioned dynamics result: current Kairos/Wan VAE-flat latents are action-insensitive in this probe
- Dreamer-style relevant/uniform sampler implemented: true
- SOAR relevant/uniform sampler smoke output: `output/soar_bc_mixture_sampler_smoke`
- SOAR relevant/uniform sampler requested relevant fraction: `0.5`
- SOAR relevant/uniform sampler enabled: true
- SOAR relevant/uniform sampler train relevant windows: `4380`
- SOAR relevant/uniform sampler train non-relevant windows: `4580`

## Decision
- Imagination training path works technically: true
- Imagination policy improvement proven on kinematic state: true
- Pixel-feature imagination improvement observed: true
- Kairos latent path works end to end: true
- Kairos latent control success proven: false
- Action tokens increase measured action use: true
- Sequence training implemented: true
- Promote imagined actor on corrected Kairos sequence run: false
- Phase-2 BC midtraining implemented: true
- SOAR real-data phase-2 smoke verified: true
- SOAR Kairos/Wan VAE-flat phase-2 smoke verified: true
- SOAR Kairos/Wan VAE-flat medium BC midtraining completed: true
- SOAR Kairos/Wan VAE-flat control baselines passed: true
- SOAR trajectory-level success/failure reward labels usable: true
- Dreamer-style phase-2 corrections added: true
- Action-conditioned dynamics scaffold validated synthetically: true
- Action-conditioned dynamics validated on SOAR Kairos/Wan latents: false
- Action-conditioned dynamics strict-gated on SOAR Kairos/Wan latents: true
- Dreamer-style relevant/uniform BC mixture implemented: true
- Full imagination RL from current SOAR Kairos/Wan latent cache recommended now: false
- Full imagination RL with jointly trained dynamics recommended now: false
- Full imagination RL with frozen strict-gated dynamics recommended next: true
- Recommended SOAR reward mode: `trajectory_success`
- SOAR Kairos/Wan VAE-flat large trajectory_success midtraining completed: true
- SOAR strict phase-2 target matched structurally: true
- SOAR strict phase-2 controls passed: false
- SOAR strict phase-2 reward/value ready: false
- SOAR BC prior artifact available: true
- Current SOAR phase-2 conclusion: architecture is close to target, but data/reward signal is not strong enough for imagination RL
- Next requirement: improve reward/progress labels or collect a more coherent action-conditioned dataset before imagination RL

# Pre-training Action-Conditioning Gate

## Result
- Action-conditioned Kairos rollouts tested: true
- camera_control_direction used: true
- camera_control_speed used: true
- yaw_left/yaw_right distinct: False
- hover distinct from yaw: False
- real SITL baseline compared: true
- Kairos-MPC teacher ready: False

## Decision
- Start BC/SFT: true
- Start MPC distillation: False
- Defer MPC distillation: True
- Need Kairos action-conditioning fine-tune: True

# Reactive Teacher Upgrade

## Result
- Observation-driven reactive teacher collector implemented: true
- Privileged teacher inputs: `depth + pose + sampled local waypoint`
- Student-visible inputs recorded: `RGB frame + pose/intrinsics + goal features in metadata`
- Reactive teacher SITL smoke verified: true
- Goal-feature-aware BC training path verified: true

## Current Dataset
- Current real episode count: 47
- Current real example count: 290
- Validation split present: true
- Validation episode count: 8
- Current action counts:
  - `hover`: 83
  - `yaw_left`: 26
  - `yaw_right`: 42
  - `ascend`: 20
  - `descend`: 14
  - `forward`: 48
  - `backward`: 23
  - `strafe_left`: 17
  - `strafe_right`: 17

## Verified Artifacts
- Reactive teacher smoke episode: `data/bc_sft/episodes/reactive_teacher_smoke_20260503T000002Z`
- Second-direction smoke episode: `data/bc_sft/episodes/reactive_teacher_smoke_20260503T000003Z`
- Reactive teacher routine pilot: `logs/overnight_bc/overnight_bc_20260503T135447Z/summary.json`
- Current manifest summary: `data/bc_sft/manifests/bc_manifest_summary.json`
- Goal-feature BC smoke checkpoint: `output/bc_policy_teacher_smoke/best.pt`

## Commands
- Reactive teacher collect:
  `docker compose run --rm tools python3 scripts/collect_sitl_bc_episode.py --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image --depth-topic /depth_camera --policy reactive_teacher --num-steps 8 --i-understand-this-is-sitl`
- Reactive teacher overnight:
  `python3 scripts/run_overnight_bc_routine.py --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image --depth-topic /depth_camera --collector-policy reactive_teacher --cycles 6 --episodes-per-cycle 6 --epochs-per-cycle 30 --batch-size 8 --image-size 128 --device auto --val-ratio 0.15 --i-understand-this-is-sitl`

## Notes
- The reactive teacher breaks the old scripted-label failure mode by choosing actions from current depth, pose, and a sampled local waypoint.
- The current model is still narrow because it uses one world and mostly single-frame inputs. More world/spawn diversity and temporal context are still needed.

# BC/SFT Bootstrap

## Status
- SITL episode collector scaffolded: true
- Manifest export script scaffolded: true
- Baseline BC trainer scaffolded: true
- Live SITL episode collected: true
- Manifest export verified: true
- Baseline BC training smoke verified: true
- Expanded SITL dataset collected: true
- Current real episode count: 5
- Current real example count: 31
- All discrete actions represented: true
- Validation split present: true
- Validation episode count: 1
- Expanded BC retraining completed: true
- Recommended collection environment: Docker tools container
- Recommended training environment: host `python3` with torch

## Verified Artifacts
- Latest SITL episode: `data/bc_sft/episodes/episode_20260503T044201Z`
- Exported manifest: `data/bc_sft/manifests/bc_manifest.jsonl`
- BC checkpoint smoke output: `output/bc_policy_baseline/best.pt`

## Current Dataset
- Episodes:
  - `episode_20260503T035455Z`
  - `episode_20260503T043958Z`
  - `episode_20260503T044033Z`
  - `episode_20260503T044109Z`
  - `episode_20260503T044201Z`
- Action counts:
  - `hover`: 11
  - `yaw_left`: 3
  - `yaw_right`: 3
  - `ascend`: 2
  - `descend`: 2
  - `forward`: 3
  - `backward`: 3
  - `strafe_left`: 2
  - `strafe_right`: 2

## Current Baseline
- Training manifest examples: 31
- Training split examples: 22
- Validation split examples: 9
- Device used: `cuda`
- Best validation loss: `2.3616`
- Final training accuracy: `0.4091`
- Final validation accuracy: `0.2222`
- Class-weighted action loss enabled: true
- Note:
  The BC path is working technically, but the current dataset is still too small and visually homogeneous to expect a strong policy yet.

## Commands
- Collect:
  `docker compose run --rm tools python3 scripts/collect_sitl_bc_episode.py --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image --actions hover,forward,yaw_left,forward,yaw_right,hover --i-understand-this-is-sitl`
- Export:
  `python3 scripts/export_bc_dataset.py --episodes-root data/bc_sft/episodes --out-jsonl data/bc_sft/manifests/bc_manifest.jsonl --summary-json data/bc_sft/manifests/bc_manifest_summary.json`
- Train:
  `python3 scripts/train_bc_policy.py --manifest data/bc_sft/manifests/bc_manifest.jsonl --out-dir output/bc_policy_baseline --epochs 10 --batch-size 16 --image-size 224 --device auto`
- Overnight:
  `python3 scripts/run_overnight_bc_routine.py --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image --cycles 6 --episodes-per-cycle 6 --epochs-per-cycle 30 --batch-size 8 --image-size 128 --device auto --val-ratio 0.15 --i-understand-this-is-sitl`

## Notes
- The collector now avoids `cv_bridge` when the existing tools image has NumPy 2.x. This keeps live collection working immediately.
- Rebuild the `tools` image when convenient so the pinned `numpy<2` in `docker/Dockerfile.tools` becomes the default environment:
  `docker compose build tools`
- The `tools` image has now been rebuilt successfully and currently resolves `numpy==1.26.4`.

# SOAR Dreamer-Lite Gate

## Status
- Drone runtime in scope: false
- SOAR cache schema has terminal `done` flags: true
- Action-conditioned latent dynamics implemented: true
- Isolated agent-token BC/reward/value heads implemented: true
- Frozen-dynamics imagination scaffold implemented: true
- Synthetic action-causal smoke test passed strict action-conditioning gate: true
- SOAR RGB stride-8 smoke test ran end-to-end: true
- SOAR RGB stride-8 strict action-conditioning gate passed: false
- SOAR Kairos/Wan VAE-flat tuned long run completed: true
- SOAR Kairos/Wan VAE-flat dynamics-focused run completed: true
- SOAR Kairos/Wan VAE-flat strict action-conditioning gate passed: false
- Current SOAR action-conditioning strength: none

## Current Evidence
- Synthetic normal/persistence: `0.856`
- Synthetic shuffled/normal: `1.162`
- Synthetic zero/normal: `1.166`
- SOAR normal/persistence: `0.995`
- SOAR shuffled/normal: `1.002`
- SOAR zero/normal: `1.002`
- SOAR Kairos tuned normal/persistence: `0.984`
- SOAR Kairos tuned shuffled/normal: `1.027`
- SOAR Kairos tuned zero/normal: `0.994`
- SOAR Kairos dynamics-focused normal/persistence: `0.970`
- SOAR Kairos dynamics-focused shuffled/normal: `1.034`
- SOAR Kairos dynamics-focused zero/normal: `0.997`

## Decision
- The SOAR-only Dreamer-lite scaffold is runnable through dynamics, BC, and imagination.
- Parameter tuning reduces unstable imagination policy drift, but does not fix the missing action grounding.
- The current Kairos/Wan SOAR cache should not yet be used as evidence of a strong controllable world simulator.
- Next target: rebuild the Kairos/Wan SOAR cache with temporal stride and summed action intervals, then rerun the same strict action-conditioning gate.

# SOAR Freeze-Value Imagination Update

## Status
- Drone runtime in scope: false
- Strict SOAR action-conditioning gate passed on current summed-action cache: true
- Calibrated BC reward/value heads trained: true
- Frozen-value imagination training implemented: true
- Frozen-value imagination run completed: true
- Learned-model transfer eval completed: true
- Previous value drift fixed by freezing value head: true

## Artifacts
- BC source checkpoint: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2/best_agent_bc.pt`
- Freeze-value imagination run: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1`
- Freeze-value transfer eval: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/soar_transfer_eval_h16`

## Evidence
- Dynamics normal/persistence: `0.852`
- Dynamics shuffled/normal: `1.092`
- Dynamics zero/normal: `1.628`
- Zero-action model return: `0.811`
- BC-prior model return: `1.243`
- After-imagination model return: `2.099`
- Return delta after minus BC: `0.856`
- Return ratio after over BC: `1.689`
- After-imagination prior-plan MSE: `0.031`
- BC-prior held-out value MSE: `3.228`
- After-imagination held-out value MSE: `3.228`
- BC-prior held-out value corr: `0.721`
- After-imagination held-out value corr: `0.721`

## Decision
- The SOAR-only action-conditioned dynamics plus frozen-value imagination loop now has a positive learned-model transfer signal.
- The previous failure was value-head drift during imagination, not necessarily policy optimization itself.
- The value head should stay frozen during near-term imagination policy optimization unless a replay-calibrated value update is separately validated.
- Next proof target: evaluate this policy outside its own learned dynamics, either by held-out SOAR counterfactual scoring or by a fresh-observation environment.

# SOAR Learned-Dynamics Eval

## Status
- Direct learned-dynamics eval script added: true
- Single-pass dynamics control eval completed: true
- Autoregressive horizon sweep completed: true
- H4 autoregressive strict gate passed: true
- H8 autoregressive strict gate passed: true
- H16 autoregressive strict gate passed: false

## Artifacts
- Locked result note: `sensenova_drone_agent/docs/SOAR_FREEZE_VALUE_IMAGINATION_RESULT.md`
- Eval script: `sensenova_drone_agent/scripts/eval_soar_learned_dynamics.py`
- H4 eval: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/learned_dynamics_eval_h4`
- H8 eval: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/learned_dynamics_eval_h8`
- H16 eval: `sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1/learned_dynamics_eval_h16`

## Evidence
- Single-pass normal/persistence: `0.852`
- Single-pass shuffled/normal: `1.092`
- Single-pass zero/normal: `1.628`
- H4 autoregressive normal/persistence: `0.922`
- H4 autoregressive shuffled/normal: `1.058`
- H4 autoregressive zero/normal: `2.668`
- H8 autoregressive normal/persistence: `0.947`
- H8 autoregressive shuffled/normal: `1.091`
- H8 autoregressive zero/normal: `6.197`
- H16 autoregressive normal/persistence: `1.061`
- H16 autoregressive shuffled/normal: `1.087`
- H16 autoregressive zero/normal: `19.445`

## Decision
- The learned SOAR dynamics are valid for short-horizon h4/h8 imagination and replanning.
- The learned SOAR dynamics are not yet valid for h16 open-loop imagination claims.
- Near-term policy/eval should use short-horizon MPC-style replanning rather than long autoregressive rollout.

# SOAR Retrofit Dynamics Ablations

## Status
- Retrofit ablation note written: true
- Rollout MSE h8/h16 tested: true
- Direct prediction horizon 8 tested: true
- Context length 16 tested: true
- Rollout contrast h8/h16 tested: true
- Conservative rollout-contrast run with dynamics-specific checkpoint metric tested: true
- Dynamics-specific checkpoint selection knobs added: true

## Artifacts
- Ablation note: `sensenova_drone_agent/docs/SOAR_RETROFIT_DYNAMICS_ABLATIONS.md`
- Trainer: `sensenova_drone_agent/scripts/train_soar_dreamer_lite.py`
- Aggressive h8 rollout contrast: `sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h8_w05_ctx8_v1`
- Aggressive h16 rollout contrast: `sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h16_w05_ctx8_v1`
- Conservative h8 rollout contrast: `sensenova_drone_agent/output/soar_retrofit_rollout_contrast_h8_w01_conservative_ctx8_v1`

## Evidence
- Current best h8 autoregressive strict gate: pass
- Current best h16 autoregressive strict gate: fail
- Rollout MSE h8 strict gates: fail at h4/h8/h16
- Rollout MSE h16 strict gates: fail at h4/h8/h16
- Rollout contrast h8 w0.5 strict gates: fail at h4/h8/h16
- Rollout contrast h16 w0.5 strict gates: fail at h4/h8/h16
- Rollout contrast h8 w0.1 conservative strict gates: fail at h4/h8/h16

## Decision
- The best current retrofit path remains the frozen-value h4/h8 short-horizon dynamics.
- Naive rollout losses do not extend the usable horizon and tend to collapse toward persistence.
- Longer direct prediction and longer context do not fix action grounding.
- The next useful retrofit knobs are gate-aware checkpointing, action-lag/window alignment, and action-gated residual dynamics.
- H16+ imagination should be treated as requiring native action-token dynamics or continued world-model training.

# SOAR Alignment/Residual/Native-WM Update

## Status
- Alignment offset -1 tested: true
- Alignment offset +1 tested: true
- Future action window 2 mean tested: true
- Action-gated residual dynamics tested: true
- Action-query-token dynamics tested: true
- Native Dreamer4-style SOAR converter added: true
- Native tokenizer smoke train completed: true
- Native action-conditioned dynamics smoke train completed: true

## Artifacts
- Updated ablation note: `sensenova_drone_agent/docs/SOAR_RETROFIT_DYNAMICS_ABLATIONS.md`
- SOAR-to-Dreamer4 converter: `sensenova_drone_agent/scripts/export_soar_dreamer4_dataset.py`
- Dreamer4 SOAR smoke dataset: `sensenova_drone_agent/data/robotics/soar/dreamer4_soar_smoke`
- Dreamer4 native smoke: `sensenova_drone_agent/output/dreamer4_soar_native_smoke_v2`

## Evidence
- Offset -1 strict gate: fail
- Offset +1 strict gate: fail
- Window 2 mean strict gate: fail
- Action-query tokens strict gate: fail
- Action-gated residual single-pass normal/persistence: `0.997`
- Action-gated residual single-pass shuffled/normal: `1.003`
- Action-gated residual h8 autoregressive normal/persistence: `1.048`
- Action-gated residual h8 autoregressive shuffled/normal: `1.005`
- Native Dreamer4 tokenizer smoke: 2 steps completed
- Native Dreamer4 action-conditioned dynamics smoke: 2 steps completed

## Decision
- The new retrofit knobs did not improve over the existing h4/h8 baseline.
- Alignment/window tests do not support a simple action-frame lag explanation.
- Residual gating is the only new knob that produced any action sensitivity, but it is too weak and worse than persistence in autoregressive rollout.
- Native Dreamer4-style world-model training on SOAR is now plumbed at smoke-test level.
- The next serious path is either gate-aware selection/curriculum on the best existing adapter or a real native SOAR tokenizer+dynamics run, not more single-run prompt-level retrofit tweaks.

# Native SOAR Dreamer4 Run v1

## Status
- Larger SOAR Dreamer4-format dataset exported: true
- Native tokenizer training completed: true
- Native action-conditioned dynamics training completed: true
- Native dynamics action-grounding eval completed: true

## Artifacts
- Dataset: `sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v1`
- Run output: `sensenova_drone_agent/output/dreamer4_soar_native_v1`
- Tokenizer checkpoint: `sensenova_drone_agent/output/dreamer4_soar_native_v1/tokenizer_ckpts/latest.pt`
- Dynamics checkpoint: `sensenova_drone_agent/output/dreamer4_soar_native_v1/dynamics_ckpts/latest.pt`
- Eval: `sensenova_drone_agent/output/dreamer4_soar_native_v1/native_dynamics_eval_h8.json`
- Native dynamics evaluator: `sensenova_drone_agent/scripts/eval_dreamer4_soar_dynamics.py`

## Evidence
- Selected SOAR trajectories: `127`
- Tasks: `16`
- Exported steps: `6066`
- Tokenizer steps: `3000`
- Dynamics steps: `5000`
- H8 autoregressive normal/persistence: `0.206`
- H8 autoregressive shuffled/normal: `1.001`
- H8 autoregressive zero/normal: `0.973`
- Direct shuffled/normal: `1.000`
- Direct zero/normal: `1.000`

## Decision
- Native Dreamer4-style training now runs end-to-end on SOAR pixels/actions.
- The native dynamics model learned a useful visual latent dynamics prior, beating persistence strongly at h8.
- The native dynamics model did not yet learn measurable action grounding.
- This checkpoint is not ready for imagination RL.
- Next native run should increase action signal and data scale before adding policy/reward/value heads.

# Native SOAR Dreamer4 Run v2 Action-Contrast

## Status
- Larger SOAR export completed: true
- Tokenizer continuation completed: true
- Dynamics continuation with action contrast completed: true
- Action-frame offset sweep completed: true

## Artifacts
- Launcher: `sensenova_drone_agent/scripts/experiments/launch_soar_native_action_contrast_v2.sh`
- Container payload: `sensenova_drone_agent/scripts/experiments/soar_native_action_contrast_v2_payload.sh`
- Dataset target: `sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast`
- Run output: `sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast`
- Log: `sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast/native_run.log`

## Configuration
- SOAR trajectories requested: `1024`
- SOAR tasks requested: `64`
- Tokenizer: continue v1 tokenizer to step `7000`, reset optimizer, LR `7e-5`
- Dynamics: continue v1 dynamics to step `18000`, reset optimizer, LR `5e-5`
- Dynamics seq len: `12`
- Dynamics batch size / grad accum: `2 / 2`
- Action frame offset during training: `0`
- Action contrast: normal actions must beat shuffled and zero actions
- Action contrast weight: `0.25`
- Action contrast margin: `0.01`
- Action contrast signal level: `0.1`
- Eval offsets: `-2, -1, 0, 1, 2`

## Evidence
- Dataset size on disk: about `2.3G`
- Tasks: `64`
- Eval batches per offset: `64`
- Best h8 autoregressive normal/persistence: `0.160` at offset `2`
- Offset `-2` h8 shuffled/normal: `1.002`
- Offset `-1` h8 shuffled/normal: `0.996`
- Offset `0` h8 shuffled/normal: `0.995`
- Offset `1` h8 shuffled/normal: `0.995`
- Offset `2` h8 shuffled/normal: `0.999`
- Offset `-2` h8 zero/normal: `0.963`
- Offset `-1` h8 zero/normal: `0.969`
- Offset `0` h8 zero/normal: `0.967`
- Offset `1` h8 zero/normal: `0.963`
- Offset `2` h8 zero/normal: `0.972`
- Summary: `sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast/summary.json`
- Report: `sensenova_drone_agent/output/dreamer4_soar_native_v2_action_contrast/report.md`

## Decision Gate
- If shuffled/normal and zero/normal remain near `1.0`, action grounding is still absent.
- If normal actions clearly beat shuffled/zero and autoregressive rollout still beats persistence, proceed to policy/reward/value heads.
- If action grounding improves only at one offset, fix the native action-frame alignment before imagination training.

## Decision
- Latent dynamics beats persistence: true
- Action grounding detected: false
- Native dynamics ready for imagination RL: false
- Do not start policy/reward/value imagination RL from this checkpoint yet.
- The action-contrast training signal affected training logs, especially zero-action contrast, but the held-out offset sweep did not preserve controllability.

# Dreamer4 HF Long Run v1

## Status
- Launched: `2026-05-14`
- Container: `sda-dreamer4-hf-longrun-20260514_114607`
- Current phase: completed
- Training queued after preprocessing: true
- Tokenizer target steps: `50000`
- Dynamics target steps: `100000`
- Full shard preprocessing completed: true
- Tokenizer training completed: true
- Dynamics training completed: true
- Final action-grounding eval completed: true

## Artifacts
- Launcher: `sensenova_drone_agent/scripts/experiments/launch_dreamer4_hf_long_run.sh`
- Container payload: `sensenova_drone_agent/scripts/experiments/dreamer4_hf_long_run_payload.sh`
- Raw dataset: `sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4`
- Full shard cache: `sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full`
- Run output: `sensenova_drone_agent/output/dreamer4_hf_long_run_v1`
- Main log: `sensenova_drone_agent/output/dreamer4_hf_long_run_v1/logs/payload.log`

## Configuration
- Dataset splits: `expert`, `mixed-small`, `mixed-large`
- Tokenizer: `128x128`, patch `8`, d_model `128`, depth `4`, latents `16`, batch `2`, sequence length `8`, grad accumulation `4`
- Dynamics: d_model `128`, depth `4`, batch `3`, sequence length `12`, grad accumulation `8`
- Dynamics actions: enabled
- Dynamics action contrast: enabled
- Action contrast weight: `0.25`
- Action contrast margin: `0.01`
- Action contrast signal level: `0.1`
- Final eval: normal actions vs shuffled actions vs zero actions, horizon `8`

## Runtime Notes
- `sglang` Kubernetes workloads were scaled to zero on `2026-05-14` to free GPU memory:
  `deployment/sglang-asgi`, `deployment/sglang-proxy`, `statefulset/sglang--asgi-event-listener`, `statefulset/sglang-handle-session`.
- Full shard cache size after preprocessing: about `337G`.
- Conservative run continues on GPU `1`.

## Final Metrics
- Dynamics final step: `100000`
- Eval batches: `128`
- Direct normal MSE: `0.0251156813`
- Direct shuffled/normal: `1.0011`
- Direct zero/normal: `1.0013`
- H8 autoregressive normal MSE: `0.0229171828`
- H8 persistence MSE: `0.0280576021`
- H8 normal/persistence: `0.8168`
- H8 shuffled/normal: `1.0043`
- H8 zero/normal: `0.9925`

## Final Decision
- Latent dynamics beats persistence: true
- Direct action conditioning detected: false
- Autoregressive action conditioning detected: false
- Native dynamics ready for imagination RL: false

# Native Dreamer4 BC-Relative Advantage Smoke v1

## Status
- Date: `2026-05-15`
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_bc_relative_smoke_v1`
- Current phase: completed
- Held-out split enabled: true
- No-update control already passed: true
- Advantage baseline: `bc_return`
- Policy-loss confidence gates enabled: true

## Configuration
- Seed: `37`
- BC steps: `240`
- Imagination updates: `80`
- Eval batches: `16`
- Eval holdout fraction: `0.1`
- Split seed: `20260515`
- Eval seed: `12345`
- Policy-loss min advantage abs: `0.25`
- Policy-loss max prior MSE: `0.12`

## Result
- Before policy-minus-BC: `-0.0075`
- Before policy-minus-zero: `0.0118`
- After policy-minus-BC: `-0.0060`
- After policy-minus-zero: `0.0134`
- Policy return delta: `0.0016`
- Policy prior MSE delta: `-0.00085`
- Policy action abs delta: `-0.0024`

## Decision
- BC-relative advantage improved stability: true
- Held-out learned return improved: true
- Policy drift reduced: true
- Policy beats BC prior: false
- Imagination repeatability gate passed: false

## Next Step
- Run a full seed-37 BC-relative imagination test.
- If it beats BC, repeat across seeds `31`, `37`, and `43`.
- If it still fails, improve reward calibration and value/return targets before scaling further.

# Native Dreamer4 Balanced Held-Out Evaluation v1

## Status
- Date: `2026-05-15`
- Code updated: `sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py`
- No-update control: `sensenova_drone_agent/output/native_dreamer4_imagination_balanced_eval_no_update_smoke_v1`
- Full BC-relative run: `sensenova_drone_agent/output/native_dreamer4_imagination_balanced_eval_bc_relative_seed37_v1`
- Current phase: completed

## Why This Was Needed
- Previous held-out evaluation iterated through task-sorted windows.
- With `64` eval batches and batch size `4`, the full seed-37 result only evaluated `acrobot-swingup`.
- This made the prior pass/fail result too narrow to trust.

## Balanced Eval Fix
- Eval sampling mode: `balanced_task_round_robin_with_replacement`
- Eval samples: `256`
- Tasks sampled: `37`
- Min samples per task: `6`
- Max samples per task: `7`
- Deterministic eval seed: `12345`

## No-Update Control
- Tasks evaluated before: `37`
- Tasks evaluated after: `37`
- Policy return delta: `0.0000`
- Policy prior MSE delta: `0.0000`
- Policy action abs delta: `0.0000`
- Decision: balanced eval plumbing works

## Full BC-Relative Seed-37 Result
- Before policy-minus-BC: `0.0059`
- Before policy-minus-zero: `0.0405`
- After policy-minus-BC: `0.0029`
- After policy-minus-zero: `0.0375`
- Policy return delta: `-0.0030`
- Policy prior MSE delta: `-0.0064`
- Policy action abs delta: `0.0251`
- Per-task mean policy-minus-BC after: `0.0041`

## Decision
- Balanced evaluation required going forward: true
- Policy after imagination beats BC: true
- Policy after imagination beats zero-action: true
- Imagination improves over pre-imagination policy: false
- Native imagination training pass: false

## Interpretation
- The constrained BC-relative update is stable and reduces policy-prior MSE.
- It does not yet improve the policy beyond the post-BC baseline.
- The next objective must optimize against the pre-imagination policy or use a more reliable reward/value target before multi-seed repeatability is meaningful.

# Native Dreamer4 Imagination Repeatability v1

## Status
- Date: `2026-05-15`
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1`
- Report: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1/report.md`
- Summary: `sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1/summary.json`
- Existing seed included: `31` from `native_dreamer4_imagination_calibrated_v2`
- New seeds run: `37`, `43`
- Result: repeatability pass `false`

## Configuration
- Base recipe: calibrated v2 native Dreamer4 imagination
- Dynamics: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/dynamics_ckpts/latest.pt`
- Tokenizer: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/tokenizer_ckpts/latest.pt`
- Context length: `8`
- Imagination horizon: `8`
- BC steps: `1200`
- Imagination updates: `400`
- Eval batches: `64`
- Action features: `current,prev,delta,mean4,norm`
- Imagination learning rate: `3e-5`
- Target normalization: `per_task`
- Advantage mode: `centered_sign`
- Value head frozen during imagination: true

## Results
- Seed `31`: pass, after policy-minus-BC `0.0587`, after policy-minus-zero `0.2369`
- Seed `37`: fail, after policy-minus-BC `-0.0190`, after policy-minus-zero `0.1043`
- Seed `43`: fail, after policy-minus-BC `-0.0100`, after policy-minus-zero `0.1111`

## Aggregate
- Completed runs: `3`
- Passing runs: `1`
- Pass fraction: `0.333`
- Mean after policy-minus-BC: `0.0099`
- Mean after policy-minus-zero: `0.1508`
- Mean policy return delta: `0.0716`
- Mean policy prior MSE after: `0.00094`

## Decision
- Native imagination produces a repeatable improvement over zero-action controls: true
- Native imagination reliably beats BC prior: false
- Prior-constrained update remains stable: true
- Advance to transfer/drone claims: false
- Next: tune reward/value/imagination objective and add held-out/no-policy-update ablations before trying transfer.

# Native Dreamer4 Held-Out Gated Imagination v1

## Status
- Date: `2026-05-15`
- No-update control output: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_no_update_smoke_v2`
- Gated smoke output: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_gated_smoke_v1`
- Full seed-37 gated output: `sensenova_drone_agent/output/native_dreamer4_imagination_heldout_gated_seed37_v1`
- Result: failed

## Implementation
- Episode-level held-out eval split added: true
- Deterministic eval seed added: true
- No-policy-update control mode added: true
- Policy-loss advantage deadzone added: true
- Policy-loss prior-distance filter added: true
- Repeatability runner updated for held-out/gating args: true

## Held-Out Split
- Split mode: episode holdout
- Holdout fraction: `0.1`
- Train windows: `3265020`
- Eval windows: `362780`

## No-Update Control
- Policy return delta: `0.0000`
- Policy prior MSE delta: `0.0000`
- Policy action abs delta: `0.0000`
- Interpretation: held-out eval plumbing is deterministic and valid.

## Full Seed-37 Gated Result
- Before policy-minus-BC: `0.0099`
- Before policy-minus-zero: `-0.0283`
- After policy-minus-BC: `-0.0059`
- After policy-minus-zero: `-0.0441`
- Policy return delta: `-0.0158`
- Policy prior MSE after: `0.0021`
- Policy prior MSE delta: `0.0017`
- Policy action abs delta: `0.0886`

## Decision
- Held-out/gated policy update improves over BC: false
- Held-out/gated policy update improves over zero: false
- Gating active: true
- Gating sufficient: false
- Next: change the objective to a BC-relative advantage baseline before running more seeds.

# Native Dreamer4 Imagination Test v1

## Status
- Launched: `2026-05-15`
- Container: `sda-native-imagination-strong-v1-cuda_20260515_160155`
- Runtime image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
- GPU: RTX 5090 via CUDA 12.8 / Torch 2.8
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_strong_contrast_v1`
- Current phase: completed

## Inputs
- Tokenizer checkpoint: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/tokenizer_ckpts/latest.pt`
- Dynamics checkpoint: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/dynamics_ckpts/latest.pt`
- Dataset splits: `expert`, `mixed-small`, `mixed-large`
- Action features: `current,prev,delta,mean4,norm`
- Action dimension: `64`

## Configuration
- Sequence length: `16`
- Context length: `8`
- Imagination horizon: `8`
- BC head steps: `1200`
- Imagination updates: `400`
- Eval batches: `64`
- Policy/value updated in imagination: true
- Frozen during imagination: tokenizer, dynamics, BC prior, reward head

## Claim Boundary
- This is a learned-dynamics imagination test only.
- It tests whether policy/value heads can improve learned reward under the frozen action-conditioned dynamics.
- It does not yet prove real-environment control, drone control, or SOAR robot transfer.
- The current dynamics run used `wm_agent_isolated`, so this first imagination test uses latent context features for policy/reward/value heads rather than Dreamer4 agent-token readouts.

## Monitoring
- Container logs: `docker logs -f sda-native-imagination-strong-v1-cuda_20260515_160155`
- File log: `tail -f sensenova_drone_agent/output/native_dreamer4_imagination_strong_contrast_v1/logs/train.log`
- Summary target: `sensenova_drone_agent/output/native_dreamer4_imagination_strong_contrast_v1/summary.json`
- Report target: `sensenova_drone_agent/output/native_dreamer4_imagination_strong_contrast_v1/report.md`

## Gate
- Pass if final policy learned return beats BC-prior learned return and zero-action learned return after imagination.
- Treat as weak/passive evidence only unless reward/value calibration and action-prior drift remain sane.

## Final Metrics
- Runtime: `769.16s`
- Before zero-action learned return: `6.3366`
- Before BC-prior learned return: `6.3914`
- Before policy learned return: `6.3782`
- After zero-action learned return: `5.8872`
- After BC-prior learned return: `5.9277`
- After policy learned return: `5.8732`
- After policy minus BC-prior: `-0.0545`
- After policy minus zero-action: `-0.0140`
- Policy prior MSE before: `0.0065`
- Policy prior MSE after: `0.0127`
- Policy action abs before: `0.1466`
- Policy action abs after: `0.2295`

## Final Decision
- First imagination training gate passed: false
- Policy improved over BC prior: false
- Policy improved over zero action: false
- Result interpretation: the policy moved farther from the BC prior and increased action magnitude, but the learned return decreased. The first objective is not stable enough to claim useful imagination improvement.
- Next correction: improve reward/value calibration and constrain policy drift before running a longer imagination phase. Candidate fixes: normalize reward/value targets per task, evaluate per-task returns instead of global averages, use advantage normalization or clipped advantages, strengthen KL/prior constraint, and run a BC-prior-only reward calibration gate before policy updates.

# Native Dreamer4 Calibrated Imagination Test v1

## Status
- Launched: `2026-05-15`
- Container: `sda-native-imagination-calibrated-v1_20260515_162847`
- Runtime image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v1`
- Current phase: running

## Purpose
- Tests the correction to the failed first imagination run.
- Adds per-task reward/value target normalization.
- Clips normalized reward/value targets to `[-5, 5]`.
- Uses centered-sign PMPO-style advantages instead of raw-sign advantages.
- Adds per-task eval summaries.
- Strengthens policy drift constraint with prior weight `1.0` plus hinge penalty.

## Configuration
- Tokenizer checkpoint: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/tokenizer_ckpts/latest.pt`
- Dynamics checkpoint: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/dynamics_ckpts/latest.pt`
- Dataset splits: `expert`, `mixed-small`, `mixed-large`
- Action features: `current,prev,delta,mean4,norm`
- Sequence/context/horizon: `16/8/8`
- BC head steps: `1200`
- Imagination updates: `400`
- Eval batches: `64`
- Target normalization: `per_task`
- Advantage mode: `centered_sign`
- Prior hinge target: `0.008`
- Value trained during imagination: true

## Monitoring
- Container logs: `docker logs -f sda-native-imagination-calibrated-v1_20260515_162847`
- File log: `tail -f sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v1/logs/train.log`
- Summary target: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v1/summary.json`

## Gate
- Pass if after-imagination policy return beats BC-prior and zero-action returns.
- Also require bounded drift: deterministic policy prior MSE should not materially increase.

## Final Metrics
- Before zero-action learned return: `-0.1804`
- Before BC-prior learned return: `-0.1445`
- Before policy learned return: `-0.1371`
- After zero-action learned return: `-0.2761`
- After BC-prior learned return: `-0.2514`
- After policy learned return: `-0.3515`
- After policy minus BC-prior: `-0.1002`
- After policy minus zero-action: `-0.0755`
- Policy prior MSE before: `0.0026`
- Policy prior MSE after: `0.0394`
- Policy action abs before: `0.1713`
- Policy action abs after: `0.4271`

## Final Decision
- Calibrated imagination v1 gate passed: false
- Reward/value target normalization helped BC scale: true
- Centered-sign advantages prevented all-positive PMPO collapse: true
- Policy drift still too large: true
- Result interpretation: reward/value calibration fixed the obvious scale problem, but the policy update still moved the deterministic policy mean too far from the BC prior. The next run adds a direct policy-mean prior penalty, lowers imagination LR, and freezes the value head during imagination.

# Native Dreamer4 Calibrated Imagination Test v2

## Status
- Launched: `2026-05-15`
- Container: `sda-native-imagination-calibrated-v2_20260515_164540`
- Runtime image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
- Output: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2`
- Current phase: completed

## Purpose
- Tests the next correction after calibrated v1 failed through deterministic policy drift.
- Adds direct policy-mean-to-BC-prior loss.
- Freezes the value head during imagination.
- Lowers imagination learning rate from `1e-4` to `3e-5`.
- Keeps per-task reward/value normalization and centered-sign advantages.

## Configuration
- Target normalization: `per_task`
- Advantage mode: `centered_sign`
- Prior weight: `1.0`
- Sample prior hinge: `25 @ 0.008`
- Mean-prior weight: `10`
- Mean-prior hinge: `100 @ 0.004`
- Value trained during imagination: false
- Imagination LR: `3e-5`

## Monitoring
- Container logs: `docker logs -f sda-native-imagination-calibrated-v2_20260515_164540`
- File log: `tail -f sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2/logs/train.log`
- Summary target: `sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2/summary.json`

## Final Metrics
- Runtime: `793.55s`
- Before zero-action learned return: `-0.4905`
- Before BC-prior learned return: `-0.3040`
- Before policy learned return: `-0.3152`
- After zero-action learned return: `-0.4672`
- After BC-prior learned return: `-0.2890`
- After policy learned return: `-0.2303`
- After policy minus BC-prior: `0.0587`
- After policy minus zero-action: `0.2369`
- Policy prior MSE before: `0.0022`
- Policy prior MSE after: `0.0012`
- Policy prior MSE delta: `-0.0010`
- Policy action abs before: `0.1687`
- Policy action abs after: `0.2141`
- Per-task mean policy-minus-BC: `0.0479`
- Tasks evaluated: `36`

## Final Decision
- Calibrated imagination v2 gate passed: true
- Policy improved over BC prior: true
- Policy improved over zero action: true
- Deterministic policy drift bounded: true
- Result interpretation: the combination of per-task target normalization, centered-sign advantages, frozen value head, lower imagination LR, and direct policy-mean prior penalty produced the first positive learned-dynamics imagination result.
- Claim boundary: this is still an internal learned-dynamics result, not external environment transfer or real drone control.
- Dedicated note: `sensenova_drone_agent/docs/NATIVE_DREAMER4_IMAGINATION_RESULT.md`

# Dreamer4 Dynamics Action Dimension Plumbing

## Status
- Updated: `2026-05-14`
- Previous behavior: the native Dreamer4 dynamics path always padded actions to `16` dimensions and constructed `ActionEncoder(... action_dim=16)`.
- New behavior: `action_dim` is an explicit trainer/eval/launcher parameter with default `16` for compatibility.
- Derived action features are now an explicit trainer/eval/launcher parameter via `action_features`, defaulting to `current`.

## Why This Matters
- The current datasets have smaller raw action vectors, but the next action-grounding ablations need room for derived action features such as previous action, action delta, rolling action summaries, and action norms.
- Supported feature list: `current`, `prev`, `delta`, `mean4`, `norm`.
- Example next ablation: `ACTION_DIM=64 ACTION_FEATURES=current,prev,delta,mean4,norm`.
- Checkpoints now record `action_dim` in `args`, and eval reads it from the checkpoint unless `--action-dim` overrides it.
- Checkpoints now record `action_features` in `args`, and eval reads it from the checkpoint unless `--action-features` overrides it.
- Shuffled-action eval/logging now shuffles action masks with the actions, so mixed-task batches remain valid.
- Long-run payload supports `SKIP_TOKENIZER=1` so action-feature dynamics ablations can reuse a validated tokenizer checkpoint without re-entering tokenizer training.
- This change does not by itself prove stronger action grounding; it removes the fixed-width/raw-current bottleneck so richer action encodings can be tested cleanly.

# Dreamer4 HF Rich Action Features Long Run v1

## Status
- Launched: `2026-05-14 21:52:38 America/Chicago`
- Container: `sda-dreamer4-hf-rich-actions-v1b_20260514_215238`
- Output: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_long_run_v1`
- Current phase: completed
- Tokenizer reused from: `sensenova_drone_agent/output/dreamer4_hf_long_run_big_v1/tokenizer_ckpts/latest.pt`
- Tokenizer skipped: true
- Dynamics fresh start: true
- Dynamics training completed: true
- Final action-grounding eval completed: true

## Configuration
- Dataset: Dreamer4 HF `expert`, `mixed-small`, `mixed-large`
- Tokenizer: big run checkpoint, patch `4`, d_model `256`, depth `8`
- Dynamics: d_model `256`, depth `8`, sequence length `16`, batch `2`, grad accumulation `8`
- Dynamics target steps: `100000`
- Action dim: `64`
- Action features: `current,prev,delta,mean4,norm`
- Max observed expanded action width at startup: `49`
- Tasks at startup: `37`
- Valid sequences at startup: `3627800`

## Early Log
- `ActionEncoder.fc1`: `in_features=64`, `out_features=512`
- Step `0`: loss `0.502479`, flow MSE `0.980097`, action contrast `0.020000`, shuffle ratio `1.000`, zero ratio `1.000`
- Step `100`: loss `0.154921`, flow MSE `0.286724`, action contrast `0.019999`, shuffle ratio `1.000`, zero ratio `1.000`

## Final Metrics
- Dynamics final step: `100000`
- Eval batches: `128`
- Direct normal MSE: `0.0107028242`
- Direct shuffled/normal: `1.0081`
- Direct zero/normal: `1.0073`
- H8 autoregressive normal MSE: `0.0180489038`
- H8 persistence MSE: `0.0436552756`
- H8 normal/persistence: `0.4134`
- H8 shuffled/normal: `1.0105`
- H8 zero/normal: `0.9724`

## Comparison To Big Baseline
- Big baseline H8 normal/persistence: `0.6089`
- Rich features H8 normal/persistence: `0.4134`
- Big baseline H8 shuffled/normal: `1.0176`
- Rich features H8 shuffled/normal: `1.0105`
- Big baseline H8 zero/normal: `0.9960`
- Rich features H8 zero/normal: `0.9724`

## Decision
- Latent dynamics beats persistence: true
- Rich action features improved latent rollout quality: true
- Direct action conditioning detected: false
- Autoregressive action conditioning detected: false
- Native dynamics ready for imagination RL: false
- Interpretation: richer action features made the simulator better at predicting future latents, but did not make predictions reliably depend on the supplied actions.

## Monitoring
- Container logs: `docker logs -f sda-dreamer4-hf-rich-actions-v1b_20260514_215238`
- Payload log: `tail -f sensenova_drone_agent/output/dreamer4_hf_rich_actions_long_run_v1/logs/payload.log`
- Dynamics log: `tail -f sensenova_drone_agent/output/dreamer4_hf_rich_actions_long_run_v1/logs/dynamics_train.log`
- Final eval target: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_long_run_v1/native_dynamics_eval_h8.json`

# Dreamer4 HF Strong Contrast Test v1

## Status
- Launched: `2026-05-15 09:08:46 America/Chicago`
- Container: `sda-dreamer4-hf-strong-contrast-v1_20260515_090846`
- Output: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1`
- Current phase: completed
- Tokenizer reused from: `sensenova_drone_agent/output/dreamer4_hf_long_run_big_v1/tokenizer_ckpts/latest.pt`
- Tokenizer skipped: true
- Dynamics fresh start: true
- Dynamics training completed: true
- Final action-grounding eval completed: true

## Configuration
- Dataset: Dreamer4 HF `expert`, `mixed-small`, `mixed-large`
- Dynamics: d_model `256`, depth `8`, sequence length `16`, batch `2`, grad accumulation `8`
- Dynamics target steps: `50000`
- Action dim: `64`
- Action features: `current,prev,delta,mean4,norm`
- Action contrast weight: `1.0`
- Action contrast margin: `0.03`
- Action contrast signal: `0.5`
- Action contrast start: `0`

## Early Log
- `ActionEncoder.fc1`: `in_features=64`, `out_features=512`
- Step `0`: loss `0.557479`, flow MSE `0.980097`, action contrast `0.060000`, shuffle ratio `1.000`, zero ratio `1.000`
- Step `100`: loss `0.209922`, flow MSE `0.286729`, action contrast `0.059998`, shuffle ratio `1.000`, zero ratio `1.000`

## Final Metrics
- Dynamics final step: `50000`
- Eval batches: `128`
- Direct normal MSE: `0.0156594056`
- Direct shuffled/normal: `1.0112`
- Direct zero/normal: `1.0091`
- H8 autoregressive normal MSE: `0.0170747675`
- H8 persistence MSE: `0.0342613167`
- H8 normal/persistence: `0.4984`
- H8 shuffled/normal: `1.0554`
- H8 zero/normal: `1.0448`

## Comparison
- Rich-features weak-contrast H8 normal/persistence: `0.4134`
- Strong-contrast H8 normal/persistence: `0.4984`
- Rich-features weak-contrast H8 shuffled/normal: `1.0105`
- Strong-contrast H8 shuffled/normal: `1.0554`
- Rich-features weak-contrast H8 zero/normal: `0.9724`
- Strong-contrast H8 zero/normal: `1.0448`

## Decision Gate
- This run tests whether stronger action-preference loss can make normal actions beat shuffled/zero actions without destroying rollout quality.
- Passing requires both action sensitivity and latent rollout quality: `shuffled/normal > 1.02`, `zero/normal > 1.02`, and `normal/persistence < 1.0`.

## Decision
- Direct action conditioning detected: false
- Autoregressive action conditioning detected: true
- Autoregressive latent dynamics beats persistence: true
- Native dynamics ready for first imagination RL test: true
- Caveat: this is an autoregressive gate pass, not a direct one-step action-conditioning pass. The next imagination test must include a regression check against persistence and shuffled/zero-action controls.

## Monitoring
- Container logs: `docker logs -f sda-dreamer4-hf-strong-contrast-v1_20260515_090846`
- Payload log: `tail -f sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/logs/payload.log`
- Dynamics log: `tail -f sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/logs/dynamics_train.log`
- Final eval target: `sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/native_dynamics_eval_h8.json`

## Monitoring
- Container logs: `docker logs -f sda-dreamer4-hf-longrun-20260514_114607`
- File log: `tail -f sensenova_drone_agent/output/dreamer4_hf_long_run_v1/logs/payload.log`

## Decision Gate
- If normal actions beat shuffled/zero actions and autoregressive dynamics beats persistence, proceed to policy/reward/value heads.
- If normal/shuffled/zero remain indistinguishable, keep this as evidence that native Dreamer4-style plumbing learns scene dynamics but not action grounding at this scale.

# Dreamer4 HF Big Long Run v1

## Status
- Launched: `2026-05-14`
- Container: `sda-dreamer4-hf-longrun-big_20260514_142228`
- Current phase: completed
- Tokenizer target steps: `50000`
- Dynamics target steps: `100000`
- Tokenizer training completed: true
- Dynamics training completed: true
- Final action-grounding eval completed: true

## Artifacts
- Run output: `sensenova_drone_agent/output/dreamer4_hf_long_run_big_v1`
- Main log: `sensenova_drone_agent/output/dreamer4_hf_long_run_big_v1/logs/payload.log`

## Configuration
- Dataset splits: `expert`, `mixed-small`, `mixed-large`
- Tokenizer: `128x128`, patch `4`, d_model `256`, depth `8`, latents `16`, batch `2`, sequence length `8`, grad accumulation `4`
- Dynamics: d_model `256`, depth `8`, batch `2`, sequence length `16`, grad accumulation `8`
- Dynamics actions: enabled
- Dynamics action contrast: enabled
- Action contrast weight: `0.25`
- Final eval: normal actions vs shuffled actions vs zero actions, horizon `8`

## Monitoring
- Container logs: `docker logs -f sda-dreamer4-hf-longrun-big_20260514_142228`
- File log: `tail -f sensenova_drone_agent/output/dreamer4_hf_long_run_big_v1/logs/payload.log`

## Purpose
- Uses freed GPU memory to test a higher-capacity native Dreamer4-style tokenizer and dynamics model in parallel with the conservative long run.

## Final Metrics
- Dynamics final step: `100000`
- Eval batches: `128`
- Direct normal MSE: `0.0129581477`
- Direct shuffled/normal: `1.0053`
- Direct zero/normal: `1.0137`
- H8 autoregressive normal MSE: `0.0222541435`
- H8 persistence MSE: `0.0365462657`
- H8 normal/persistence: `0.6089`
- H8 shuffled/normal: `1.0176`
- H8 zero/normal: `0.9960`

## Final Decision
- Latent dynamics beats persistence: true
- Direct action conditioning detected: false
- Autoregressive action conditioning detected: false
- Native dynamics ready for imagination RL: false
