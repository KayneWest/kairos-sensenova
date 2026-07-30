# Action World-Model Data Collection

This is the input corpus for continued action-conditioned world-model training. It is not the policy/reward/value midtraining corpus by itself; its job is to make future latent prediction explicitly depend on actions.

## Current Corpus

Manifest:

```text
sensenova_drone_agent/data/action_world_model_continue_v1/manifest.json
```

Report:

```text
sensenova_drone_agent/data/action_world_model_continue_v1/report.md
```

Audited sources:

```text
dreamer4_hf_expert        raw=37  frames=37  usable=37  shards=185
dreamer4_hf_mixed_small   raw=37  frames=37  usable=37  shards=185
dreamer4_hf_mixed_large   raw=30  frames=30  usable=30  shards=1470
soar_robotics_task_balanced raw=64 frames=64 usable=64 shards=64
robonet_sample_64         raw=9   frames=9   usable=9   shards=24
hf_robot_droid_lerobot_dreamer4 raw=1 frames=1 usable=1 shards=436
hf_robot_fractal20220817_data_lerobot_dreamer4 raw=1 frames=1 usable=1 shards=268
hf_robot_bridge_orig_lerobot_dreamer4 raw=1 frames=1 usable=1 shards=212
```

Total usable task streams: `180`.

Recommended initial sampling weights:

```text
dreamer4_hf_expert              0.20
dreamer4_hf_mixed_small         0.20
dreamer4_hf_mixed_large         0.20
soar_robotics_task_balanced     0.25
robonet_sample_64               0.15
hf_robot_*                      included as large fixed-task streams in all-data launcher
```

## Why These Sources

Dreamer4-HF is the strongest action-causality anchor because wrong simulator actions should visibly produce wrong futures.

SOAR is the closest local source to the Dreamer 4 paper's robotics setting. It provides real manipulation video, 7D end-effector actions, and success/failure labels that can support reward/event heads later.

RoboNet adds extra robot action-video replay. It has zero reward placeholders in our export, so it should be used for action-conditioned dynamics and anti-forgetting, not reward claims.

## Continued Training Target

Freeze or mostly freeze the tokenizer/VAE, then continue-train dynamics with:

```text
z_t + action tokens + optional proprio/action features -> future z
```

Losses to preserve in this phase:

```text
future latent prediction
true-action vs zero/shuffle/time-shift contrast
inverse dynamics
action-effect prediction
reward/event prediction where available
original reconstruction/no-action replay to reduce forgetting
```

Strict gates:

```text
normal actions beat shuffled, zero, and time-shifted actions
normal autoregressive latent prediction beats persistence
true-action reward/event return beats counterfactual action return on positive windows
no-action generation quality does not collapse
```

## Reproduce Collection

Audit existing local data:

```bash
python3 sensenova_drone_agent/scripts/collect_action_world_model_data.py
```

Download/export missing pieces if needed:

```bash
python3 sensenova_drone_agent/scripts/collect_action_world_model_data.py --download --preprocess --export
```

The collector intentionally writes a manifest even in audit mode so training launchers can consume one stable source of truth.

## Smoke Validation

Single-GPU continuation smoke:

```text
run: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_train_smoke_v1
base dynamics: dreamer4_all_data_native_causal_ident_gate20k_continue_25k_v1/dynamics_ckpts/latest.pt
global steps: 25000 -> 25002
usable tasks loaded by training: 109
valid train windows: 3669335
action features: current,prev,delta,mean4,norm
action contrast negatives: shuffle,zero,time_shift
```

One-batch causal eval:

```text
direct normal MSE:        0.027155
direct shuffle/normal:   1.073924
direct zero/normal:      1.051194
autoregressive normal:   0.019210
autoregressive shuffle/normal: 1.304962
autoregressive zero/normal:    1.222612
normal/persistence:      0.705862
strict gate passed:      true
```

This validates data loading, resumed continued dynamics training, richer action features, and strict action-counterfactual evaluation plumbing. It is not a publication-strength result because it uses only one eval batch.

## Additional Real-Action Sources

We added a Hugging Face robotics-action downloader for larger LeRobot/OXE-style corpora:

```text
sensenova_drone_agent/scripts/download_robot_action_hf_datasets.py
```

The first active profile is `oxe-compact`, using video-paired parquet filtering so we only fetch episodes that have both low-level actions and matching MP4 observations:

```text
IPEC-COMMUNITY/droid_lerobot                 10.09 GiB, 3000 parquets, 7293 videos
IPEC-COMMUNITY/fractal20220817_data_lerobot   2.92 GiB, 12515 parquets, 12515 videos
IPEC-COMMUNITY/bridge_orig_lerobot            4.50 GiB, 12000 parquets, 46475 videos
IPEC-COMMUNITY/language_table_lerobot         skipped for pixel dynamics in paired-video mode
```

Download status:

```text
status: complete
manifest: sensenova_drone_agent/data/robotics/hf_action_sources/download_manifest.json
target: sensenova_drone_agent/data/robotics/hf_action_sources
downloaded local size: about 21 GiB
```

The downloader writes one `.download.json` state file per repo and retries each repo after transient HF/network failures. It reads `HF_TOKEN` from the environment if available.

Implementation note: in paired-video mode, the downloader audits matching MP4 episodes and downloads those exact files via parallel `hf_hub_download` calls. This avoids `snapshot_download` pattern matching over thousands of exact paths and should not fetch the full action-only parquet tree.

We also added the export bridge for these sources:

```text
sensenova_drone_agent/scripts/export_lerobot_hf_dreamer4_dataset.py
```

Example export after a snapshot completes:

```bash
docker run --rm --ipc=host \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-dreamer:local \
  python sensenova_drone_agent/scripts/export_lerobot_hf_dreamer4_dataset.py \
    --input sensenova_drone_agent/data/robotics/hf_action_sources/IPEC_COMMUNITY_bridge_orig_lerobot \
    --out sensenova_drone_agent/data/robotics/hf_action_exports/bridge_orig_lerobot_dreamer4 \
    --dataset-name bridge_orig_lerobot \
    --max-trajectories 0 \
    --paired-video-parquets-only \
    --frame-stride 2 \
    --frame-size 128 \
    --task-mode fixed \
    --reward-mode zero
```

Export status:

```text
container: sda-hf-action-export exited 0
root: sensenova_drone_agent/data/robotics/hf_action_exports
datasets: droid_lerobot_dreamer4, fractal20220817_data_lerobot_dreamer4, bridge_orig_lerobot_dreamer4
droid_lerobot_dreamer4: 2999 trajectories, 436 frame shards
fractal20220817_data_lerobot_dreamer4: 12457 trajectories, 268 frame shards
bridge_orig_lerobot_dreamer4: 11997 trajectories, 212 frame shards
```

The collector and all-data launcher now support these exports via the `hf-robot` source family:

```bash
python3 sensenova_drone_agent/scripts/collect_action_world_model_data.py \
  --sources dreamer4-hf,soar,robonet,hf-robot
```

## Active Long Continued-Dynamics Run

Started on 2026-05-21:

```text
run: continued_action_wm_hf_robot_v1
container: sda-dreamer4-all-data-continued_action_wm_hf_robot_v1
output: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_v1
base tokenizer: dreamer4_all_data_native_causal_ident_gate20k_continue_25k_v1/tokenizer_ckpts/latest.pt
base dynamics: dreamer4_all_data_native_causal_ident_gate20k_continue_25k_v1/dynamics_ckpts/latest.pt
tokenizer: frozen/skipped
dynamics target: 150000 steps, resuming from step 25000
gpus: 2
merged tasks: 113
active tasks after valid-window filtering: 112
valid train windows: 4196094
action dim: 49
action features: current,prev,delta,mean4,norm
action contrast negatives: shuffle,zero,time_shift
contrast min action norm: 0.001
eval max batches: 256
```

Early resumed metrics:

```text
step 0025000 | loss=0.026949 | flow_mse=0.059488 | boot_mse=0.020767 | act_contrast=0.015353 | act_shuffle=1.450 | act_zero=1.400
step 0025100 | loss=0.013743 | flow_mse=0.054188 | boot_mse=0.010439 | act_contrast=0.008282 | act_shuffle=1.000 | act_zero=1.201
step 0025200 | loss=0.009719 | flow_mse=0.022002 | boot_mse=0.005155 | act_contrast=0.005685 | act_shuffle=1.000 | act_zero=1.140
step 0025300 | loss=0.011361 | flow_mse=0.022352 | boot_mse=0.003705 | act_contrast=0.007011 | act_shuffle=1.375 | act_zero=1.122
```

Interpretation: launch and data plumbing are verified, but this is not yet a final action-causal result. Wait for held-out eval at the checkpoint intervals and require normal actions to beat shuffled, zero, time-shifted actions, and persistence under the strict gates.

Final result:

```text
container: exited 0
final checkpoint: dynamics_ckpts/final_step_0150000.pt
eval: native_dynamics_eval_h8_all_data.json
batches: 256
direct shuffle/normal: 1.0622
direct zero/normal: 1.0388
autoregressive shuffle/normal: 1.0911
autoregressive zero/normal: 1.0471
autoregressive normal/persistence: 0.5200
strict gate passed: true
```

This validates shuffle/zero action grounding and persistence on the held-out eval. It did not validate time-shift.

## Time-Shift Counterfactual Eval

Dedicated eval:

```text
container: sda-dreamer4-time-shift-full-eval-v1 exited 0
eval: native_dynamics_eval_h8_all_data_time_shift_full.json
negative modes: time_shift
valid train/eval windows loaded: 4196094 across 112 tasks
batches: 256
direct normal: 0.0160486
direct time_shift: 0.0159221
direct time_shift/normal: 0.9921
direct pair pass fraction: 0.0742
autoregressive normal: 0.0151755
autoregressive time_shift: 0.0151852
autoregressive persistence: 0.0272141
autoregressive normal/persistence: 0.5576
autoregressive time_shift/normal: 1.0006
autoregressive pair pass fraction: 0.0859
strict gate passed: false
```

Interpretation: the trained checkpoint is sensitive to shuffled and zeroed actions but not to one-step temporally shifted actions. This means the current checkpoint is not yet robustly time-index action-causal.

Important bug found: the 150k launch used `ACTION_CONTRAST_NEGATIVE_MODES=shuffle,zero,time_shift`, but `dreamer4/dreamer4/train_dynamics.py` only recognized `time` or `temporal`, so `time_shift` was silently ignored during training. The trainer has been patched to accept `time_shift`, `timeshift`, and `shift`, and to log `act_time`. Future time-shift contrast runs must be launched from `final_step_0150000.pt` or a later checkpoint with this parser fix.

Operational note: the first time-shift eval container lacked `tensordict`, causing Dreamer4-HF streams to be skipped and only `568294` windows across `75` tasks to load. The corrected full eval installed `tensordict` in the transient container and matched the full all-data count.

## Time-Shift Contrast Continuation

Started on 2026-05-21:

```text
run: continued_action_wm_hf_robot_timeshift_25k_v1
container: sda-dreamer4-all-data-continued_action_wm_hf_robot_timeshift_25k_v1
output: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_timeshift_25k_v1
base tokenizer: dreamer4_all_data_native_continued_action_wm_hf_robot_v1/tokenizer_ckpts/latest.pt
base dynamics: dreamer4_all_data_native_continued_action_wm_hf_robot_v1/dynamics_ckpts/final_step_0150000.pt
tokenizer: frozen/skipped
dynamics target: 175000 steps, resuming from step 150000
incremental training budget: 25000 steps
gpus: 2
valid train windows: 4196094 across 112 tasks
action dim: 49
action features: current,prev,delta,mean4,norm
action contrast negatives: shuffle,zero,time_shift
contrast min action norm: 0.001
temporal contrast start: 1
eval negatives: shuffle,zero,time_shift
eval max batches: 256
```

Early resumed metrics after fixing the `time_shift` parser:

```text
step 0150000 | loss=0.007323 | flow_mse=0.022147 | boot_mse=0.010873 | act_contrast=0.004440 | act_shuffle=1.142 | act_zero=1.110 | act_time=1.001
step 0150100 | loss=0.007261 | flow_mse=0.026676 | boot_mse=0.005579 | act_contrast=0.006381 | act_shuffle=1.000 | act_zero=1.087 | act_time=1.003
step 0150200 | loss=0.004205 | flow_mse=0.010085 | boot_mse=0.000598 | act_contrast=0.004057 | act_shuffle=1.000 | act_zero=1.126 | act_time=1.029
step 0150300 | loss=0.003619 | flow_mse=0.007974 | boot_mse=0.002587 | act_contrast=0.001583 | act_shuffle=1.210 | act_zero=1.202 | act_time=1.029
step 0150400 | loss=0.002298 | flow_mse=0.001411 | boot_mse=0.001636 | act_contrast=0.003584 | act_shuffle=1.154 | act_zero=1.362 | act_time=1.097
step 0150500 | loss=0.002627 | flow_mse=0.008418 | boot_mse=0.000364 | act_contrast=0.000257 | act_shuffle=1.276 | act_zero=1.328 | act_time=1.112
```

Interpretation: the continuation is now applying the intended temporal counterfactual loss. Early minibatches show `act_time` moving above 1.0, but this is not sufficient; require the final held-out eval to show normal actions beat `shuffle`, `zero`, and `time_shift` negatives.

Training completion status:

```text
container: exited 2 after training, due to a post-training shell eval quoting issue
final checkpoint saved: dynamics_ckpts/final_step_0175000.pt
latest checkpoint saved: dynamics_ckpts/latest.pt
last logged train step: 0174900
last-50 logged train mean shuffle ratio: 1.0916
last-50 logged train mean zero ratio: 1.1449
last-50 logged train mean time_shift ratio: 1.1230
```

The training pass itself completed and checkpointed. The final held-out eval is being rerun from `final_step_0175000.pt` in a separate container:

```text
container: sda-dreamer4-timeshift-final-eval-v1
eval output: native_dynamics_eval_h8_all_data_shuffle_zero_time_shift_final.json
negative modes: shuffle,zero,time_shift
```

Final held-out eval result:

```text
container: sda-dreamer4-timeshift-final-eval-v1 exited 0
eval: native_dynamics_eval_h8_all_data_shuffle_zero_time_shift_final.json
batches: 256
direct normal: 0.0153539
direct shuffle/normal: 1.0782
direct zero/normal: 1.0451
direct time_shift/normal: 1.0068
direct time_shift pair pass fraction: 0.2695
autoregressive normal: 0.0137676
autoregressive persistence: 0.0289567
autoregressive normal/persistence: 0.4755
autoregressive shuffle/normal: 1.2066
autoregressive zero/normal: 1.1597
autoregressive time_shift/normal: 1.0060
autoregressive time_shift pair pass fraction: 0.1563
strict gate passed: false
native dynamics ready for imagination: false
```

Interpretation: the time-shift contrast continuation improved temporal sensitivity compared with the previous full eval (`time_shift/normal` direct `0.9921 -> 1.0068`, autoregressive `1.0006 -> 1.0060`), but it still does not clear the strict `1.02` held-out causal gate and has weak pairwise pass fractions. The model is robustly action-conditioned for shuffled and zeroed actions, but still not reliably time-index action-causal.

## Hard Temporal Negative Continuation

Started on 2026-05-22 to attack the remaining time-index causality failure:

```text
run: continued_action_wm_hf_robot_temporal_hard_50k_v1
container: sda-dreamer4-all-data-continued_action_wm_hf_robot_temporal_hard_50k_v1
output: sensenova_drone_agent/output/dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1
base tokenizer: dreamer4_all_data_native_continued_action_wm_hf_robot_timeshift_25k_v1/tokenizer_ckpts/latest.pt
base dynamics: dreamer4_all_data_native_continued_action_wm_hf_robot_timeshift_25k_v1/dynamics_ckpts/final_step_0175000.pt
tokenizer: frozen/skipped
dynamics target: 225000 steps, resuming from step 175000
incremental training budget: 50000 steps
batch size per process: 8
grad accumulation: 2
gpus: 2
valid train windows: 4196094 across 112 tasks
action dim: 49
action features: current,prev,delta,mean4,norm
action contrast weight: 2.0
action contrast margin: 0.02
action contrast signal: 0.1
action contrast negatives: shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_reverse,time_perm
contrast min action norm: 0.002
temporal contrast start: 2
contrast action norm focus: 2.0
contrast latent delta focus: 3.0
contrast weight clip: 20.0
self fraction: 0.10
eval negatives: shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_reverse,time_perm
eval max batches: 512
```

Code changes for this run:

```text
train_dynamics.py: supports time_shiftN, time_reverse, and time_perm same-window temporal negatives.
train_dynamics.py: contrast loss can focus on high-action-norm and high-latent-delta timesteps.
eval_dreamer4_soar_dynamics.py: supports the same hard temporal negative modes.
all_data_native_dreamer4_payload.sh: exposes contrast focus weights and dynamics LR via environment variables.
```

Early metrics:

```text
step 0175000 | loss=0.013633 | flow_mse=0.024054 | boot_mse=0.001577 | act_contrast=0.004841 | act_shuffle=1.123 | act_zero=1.167 | act_time=1.063
step 0175100 | loss=0.005729 | flow_mse=0.019170 | boot_mse=0.000810 | act_contrast=0.000789 | act_shuffle=1.170 | act_zero=1.169 | act_time=1.280
step 0175200 | loss=0.011639 | flow_mse=0.063198 | boot_mse=0.000280 | act_contrast=0.001868 | act_shuffle=1.110 | act_zero=1.187 | act_time=1.289
```

Interpretation: early minibatch temporal ratios are much higher than the prior one-step-only run, but this is deliberately a harder training objective and not yet a held-out result. The final decision must come from the `512` batch eval with all hard negatives.

Final held-out eval result:

```text
container: sda-dreamer4-all-data-continued_action_wm_hf_robot_temporal_hard_50k_v1 exited 0
final checkpoint: dynamics_ckpts/final_step_0225000.pt
eval: native_dynamics_eval_h8_all_data.json
batches: 512
direct normal: 0.0130191
direct shuffle/normal: 1.1176
direct zero/normal: 1.0597
direct time_shift/normal: 0.9642
direct time_shift2/normal: 0.9503
direct time_shift4/normal: 0.9409
direct time_shift8/normal: 0.9424
direct time_reverse/normal: 0.9449
direct time_perm/normal: 0.9435
autoregressive normal: 0.0126032
autoregressive persistence: 0.0281849
autoregressive normal/persistence: 0.4472
autoregressive shuffle/normal: 1.1331
autoregressive zero/normal: 1.0903
autoregressive time_shift/normal: 0.9989
autoregressive time_shift2/normal: 1.0037
autoregressive time_shift4/normal: 1.0167
autoregressive time_shift8/normal: 1.0203
autoregressive time_reverse/normal: 1.0093
autoregressive time_perm/normal: 1.0120
strict gate passed: false
native dynamics ready for imagination: false
```

Interpretation: the aggressive hard-negative continuation preserves strong shuffle/zero action grounding and improves some autoregressive temporal negatives, with `time_shift8` barely clearing `1.02`. However, one-step and small-shift temporal causality still fail, and direct denoising temporal negatives are below normal. The in-training `act_time` ratio did not transfer to the held-out temporal gate, so the current architecture/objective can still use action identity without reliably assigning actions to the exact timestep.

## Offset And Per-Source Causal Sweep

Started on 2026-05-25 to distinguish timestamp alignment issues from source-specific data contamination:

```text
script: sensenova_drone_agent/scripts/run_dreamer4_action_alignment_sweep.py
base checkpoint: dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1/dynamics_ckpts/final_step_0225000.pt
tokenizer: dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1/tokenizer_ckpts/latest.pt
output dir: dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1/alignment_sweep
```

Running containers:

```text
sda-dreamer4-align-full-offset-v1: full corpus, offsets -3..+3, 96 batches/offset, GPU 0
sda-dreamer4-align-source-v1: each source, offsets -2..+2, 32 batches/source/offset, GPU 1
```

Outputs:

```text
full_offset_sweep.json
full_offset_sweep.csv
source_offset_sweep.json
source_offset_sweep.csv
full_offset.log
source_offset.log
```

Final full-corpus offset observations:

```text
offset -3: ar=0.012423, ar_t1=1.0016, ar_t4=0.9923, ar_t8=0.9928, ar_zero=1.0719, ar_shuffle=1.1404
offset -2: ar=0.012407, ar_t1=0.9970, ar_t4=0.9988, ar_t8=0.9928, ar_zero=1.0705, ar_shuffle=1.1433
offset -1: ar=0.012398, ar_t1=1.0015, ar_t4=1.0041, ar_t8=0.9990, ar_zero=1.0734, ar_shuffle=1.1475
offset  0: ar=0.012386, ar_t1=0.9981, ar_t4=1.0039, ar_t8=1.0012, ar_zero=1.0714, ar_shuffle=1.1488
offset +1: ar=0.012406, ar_t1=1.0016, ar_t4=1.0047, ar_t8=1.0089, ar_zero=1.0734, ar_shuffle=1.1472
offset +2: ar=0.012366, ar_t1=1.0032, ar_t4=1.0063, ar_t8=1.0014, ar_zero=1.0769, ar_shuffle=1.1441
offset +3: ar=0.012358, ar_t1=1.0004, ar_t4=1.0016, ar_t8=0.9987, ar_zero=1.0776, ar_shuffle=1.1338
```

Final per-source best temporal observations:

```text
dreamer4_hf_expert: best_temporal=1.0146 via time_reverse at offset -2, zero=1.1372, shuffle=1.2095
dreamer4_hf_mixed_large: best_temporal=1.0352 via time_shift4 at offset -2, zero=1.1041, shuffle=1.0055
dreamer4_hf_mixed_small: best_temporal=1.0104 via time_shift8 at offset +1, zero=0.9502, shuffle=0.9652
hf_robot_bridge_orig_lerobot_dreamer4: best_temporal=1.0082 via time_perm at offset -1, zero=1.0516, shuffle=1.0042
hf_robot_droid_lerobot_dreamer4: best_temporal=1.0396 via time_shift8 at offset -2, zero=1.2122, shuffle=0.9744
hf_robot_fractal20220817_data_lerobot_dreamer4: best_temporal=1.0171 via time_shift8 at offset -2, zero=1.1048, shuffle=1.0191
robonet_sample_64: best_temporal=1.0165 via time_perm at offset -1, zero=1.0141, shuffle=0.9673
soar_native_v2: best_temporal=1.0240 via time_shift4 at offset -1, zero=1.1278, shuffle=1.0146
```

Interpretation: the sweep does not support a clean global action timestamp correction. Full-corpus offset differences are small, one-step temporal causality remains near 1.0, and the strongest temporal signal is source-specific. SOAR, DROID, and Dreamer4 mixed-large carry the most useful action-time signal. Mixed-small, Bridge, and RoboNet should be downweighted or excluded from the next causality-focused continuation.

Next training implication: do not run another uniformly mixed all-data continuation. Run a source-weighted continuation that focuses on `soar_native_v2`, `hf_robot_droid_lerobot_dreamer4`, and `dreamer4_hf_mixed_large`, with offset candidates `-1` and `-2`, stronger losses on `time_shift4`, `time_shift8`, and `time_perm`, and separate per-mode training logs instead of the aggregate `act_time`.

## Source-Weighted Continuation

Added launcher support for source duplication weights and action-frame offsets:

```text
SOURCE_DEFAULT_WEIGHT=0
SOURCE_WEIGHTS=soar_native_v2=5,hf_robot_droid_lerobot_dreamer4=4,dreamer4_hf_mixed_large=4,dreamer4_hf_expert=1,hf_robot_fractal20220817_data_lerobot_dreamer4=1
ACTION_FRAME_OFFSET=-1 or -2
```

Training code now logs per-temporal-negative contrast ratios such as `time_shift4`, `time_shift8`, `time_perm`, and `time_reverse` instead of only the aggregate `act_time`. This matters because the previous hard-negative continuation improved larger temporal shifts while one-step shift stayed weak.

Planned run:

```text
base dynamics: dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1/dynamics_ckpts/final_step_0225000.pt
base tokenizer: dreamer4_all_data_native_continued_action_wm_hf_robot_temporal_hard_50k_v1/tokenizer_ckpts/latest.pt
target step: 275000
negative modes: shuffle,zero,time_shift2,time_shift4,time_shift4,time_shift8,time_shift8,time_perm,time_reverse
contrast weight: 2.0
LR: 2e-5
```

Launched on 2026-05-26:

```text
sda-dreamer4-source-weight-m1-v1: offset -1, GPU 0, output dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1
sda-dreamer4-source-weight-m2-v1: offset -2, GPU 1, output dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m2_50k_v1
```

Both runs resumed successfully at step `225000`. Initial source-weighted minibatch logs show strong temporal contrast ratios around `1.5+` for `time_shift2`, `time_shift4`, and `time_shift8`; this only confirms the objective is active, not held-out causality.

Final held-out evals completed cleanly:

```text
offset -1 output: dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/native_dynamics_eval_h8_all_data.json
offset -2 output: dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m2_50k_v1/native_dynamics_eval_h8_all_data.json
batches: 512 each
```

Offset `-1`:

```text
direct normal: 0.0122817
direct shuffle/normal: 1.0549
direct zero/normal: 1.0319
direct time_shift/normal: 0.9850
direct time_shift2/normal: 0.9814
direct time_shift4/normal: 0.9824
direct time_shift8/normal: 0.9806
AR normal: 0.0105726
AR persistence: 0.0265604
AR normal/persistence: 0.3981
AR shuffle/normal: 1.1389
AR zero/normal: 1.0775
AR time_shift/normal: 1.0040
AR time_shift2/normal: 1.0137
AR time_shift4/normal: 1.0227
AR time_shift8/normal: 1.0000
strict gate: false
```

Offset `-2`:

```text
direct normal: 0.0122429
direct shuffle/normal: 1.0505
direct zero/normal: 1.0234
direct time_shift/normal: 0.9891
direct time_shift2/normal: 0.9809
direct time_shift4/normal: 0.9759
direct time_shift8/normal: 0.9790
AR normal: 0.0100178
AR persistence: 0.0261871
AR normal/persistence: 0.3825
AR shuffle/normal: 1.1431
AR zero/normal: 1.0545
AR time_shift/normal: 1.0043
AR time_shift2/normal: 1.0067
AR time_shift4/normal: 1.0049
AR time_shift8/normal: 0.9973
strict gate: false
```

Interpretation: source weighting improved AR prediction quality and preserved coarse action identity, but it still does not produce reliable exact action-time causality. Offset `-1` is the better causal checkpoint because `AR time_shift4/normal=1.0227` clears the `1.02` threshold, but one-step/two-step/eight-step/perm/reverse remain below threshold. Offset `-2` has lower AR normal MSE but weaker temporal causality.

## Immediate Causal Diagnostics

Started after source-weighted continuation:

```text
per-source eval container: sda-dreamer4-per-source-m1-v1
checkpoint: source_weighted_m1_50k_v1/dynamics_ckpts/final_step_0275000.pt
offset: -1
negative modes: shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse
output: source_weighted_m1_50k_v1/per_source_eval_offset_m1/source_eval_offset_m1.json
```

Per-source eval completed:

```text
dreamer4_hf_expert: AR=0.011049, persist=0.3636, shuffle=1.0728, zero=1.0571, t1=1.0018, t2=1.0047, t4=1.0041, t8=0.9905, perm=1.0016, reverse=1.0063
dreamer4_hf_mixed_small: AR=0.011168, persist=0.3902, shuffle=1.0084, zero=1.0070, t1=1.0023, t2=1.0056, t4=0.9994, t8=0.9727, perm=0.9879, reverse=1.0143
dreamer4_hf_mixed_large: AR=0.007054, persist=0.2958, shuffle=1.1344, zero=1.0868, t1=1.0122, t2=1.0199, t4=1.0287, t8=1.0009, perm=0.9942, reverse=1.0217
soar_native_v2: AR=0.010355, persist=0.5941, shuffle=0.9958, zero=1.4091, t1=1.0036, t2=1.0067, t4=0.9996, t8=1.0070, perm=1.0090, reverse=1.0135
robonet_sample_64: AR=0.028791, persist=0.6220, shuffle=0.9949, zero=1.0552, t1=1.0021, t2=0.9976, t4=0.9891, t8=0.9898, perm=0.9959, reverse=0.9886
droid_lerobot: AR=0.011519, persist=0.6399, shuffle=0.9966, zero=1.5224, t1=1.0084, t2=1.0211, t4=1.0192, t8=1.0382, perm=1.0291, reverse=1.0090
fractal: AR=0.080807, persist=0.9567, shuffle=0.9969, zero=1.1773, t1=1.0014, t2=1.0052, t4=1.0071, t8=1.0130, perm=1.0098, reverse=1.0040
bridge: AR=0.015667, persist=0.5851, shuffle=0.9930, zero=1.1697, t1=1.0135, t2=1.0188, t4=1.0136, t8=0.9981, perm=0.9943, reverse=1.0250
```

Interpretation: no source passes one-step temporal causality. `dreamer4_hf_mixed_large` is the best general source with strong AR quality and `t4/reverse` causality. `DROID` has the strongest zero-action dependence and passes `t2`, `t8`, and `perm`, but not shuffle or one-step. SOAR has strong zero-action sensitivity and strong inverse-dynamics signal, but weak forward temporal shift.

Added inverse-dynamics probe:

```text
script: sensenova_drone_agent/scripts/run_inverse_dynamics_probe.py
probe: (z_t, z_{t+1}, z_{t+1}-z_t) -> raw_action_t
sources: soar_native_v2, hf_robot_droid_lerobot_dreamer4, dreamer4_hf_mixed_large
offsets: 0,-1,-2
container: sda-dreamer4-inverse-probe-v1
output: source_weighted_m1_50k_v1/inverse_dynamics_probe/probe_high_signal_sources.json
```

Added high-action-window filter plumbing:

```text
train_dynamics.py:
  --require_non_noop
  --no_op_threshold
  --min_non_noop_steps
  --reward_filter_mode
  --reward_signal_threshold
  --min_reward_signal_steps

eval_dreamer4_soar_dynamics.py / run_dreamer4_action_alignment_sweep.py:
  --require-non-noop
  --no-op-threshold
  --min-non-noop-steps
```

Raw action norm stats show action magnitude alone is a blunt filter because many sources have high action norms on most timesteps:

```text
dreamer4_hf_mixed_large: q50=1.28, q90=2.37, frac norm>0.1=0.804
soar_native_v2: q50=2.00, q90=2.00, frac norm>0.1=0.716
droid_lerobot: q50=1.00, q90=1.30, frac norm>0.1=0.991
fractal: q50=1.00, q90=1.06, frac norm>0.1=0.866
```

Practical high-action filter for the next eval/training attempt:

```text
REQUIRE_NON_NOOP=1
NO_OP_THRESHOLD=0.1
MIN_NON_NOOP_STEPS=12
```

This is not yet a true latent-effect filter. It is the first cheap high-action filter. A stronger version should filter by encoded latent delta or pixel/latent motion once the inverse-dynamics probe establishes which sources preserve action-identifiable transitions.

Started high-action filtered eval:

```text
container: sda-dreamer4-high-action-eval-v1
sources: dreamer4_hf_mixed_large, soar_native_v2, droid_lerobot, bridge
filter: require_non_noop, no_op_threshold=0.1, min_non_noop_steps=12
offset: -1
output: source_weighted_m1_50k_v1/high_action_eval_offset_m1/high_action_eval_offset_m1.json
```

High-action filtered eval completed:

```text
filter: require_non_noop, no_op_threshold=0.1, min_non_noop_steps=12
dreamer4_hf_mixed_large: AR=0.010641, persist=0.3909, shuffle=0.9490, zero=0.9039, t1=0.9973, t2=1.0002, t4=0.9899, t8=0.9533, perm=0.9706, reverse=1.0083
soar_native_v2: AR=0.008952, persist=0.5833, shuffle=0.9997, zero=1.4525, t1=1.0043, t2=1.0137, t4=1.0101, t8=1.0080, perm=1.0137, reverse=1.0217
droid_lerobot: AR=0.006743, persist=0.5468, shuffle=1.0051, zero=1.6043, t1=1.0069, t2=1.0181, t4=1.0170, t8=1.0318, perm=1.0201, reverse=1.0168
bridge: AR=0.013223, persist=0.7777, shuffle=1.0334, zero=1.3297, t1=1.0025, t2=1.0122, t4=1.0071, t8=1.0264, perm=1.0165, reverse=1.0129
```

Interpretation: action-magnitude filtering improves AR normal quality and delayed temporal/zero-action sensitivity for SOAR, DROID, and Bridge, but it still does not fix one-step temporal causality. `t1` remains around `1.00-1.007`. Filtering alone is therefore not enough.

Inverse-dynamics probe completed:

```text
SOAR best R2: 0.5364 at offset 0
DROID best R2: 0.3081 at offset 0
dreamer4_hf_mixed_large best R2: 0.0113 at offset -1
```

Interpretation: SOAR and DROID tokenizer latents preserve action-identifiable transition information, especially SOAR. Mixed-large has useful forward temporal signals but its latent transition does not linearly identify actions, likely because the action semantics/tasks are more heterogeneous. The failure is therefore not just the tokenizer erasing action information; the forward dynamics needs a stronger action-specific path.

Decision: proceed to residual action-dynamics adapter. Use frozen tokenizer and frozen source-weighted dynamics, and train a small action-conditioned residual head:

```text
base_pred = frozen_dynamics(z_context, actions)
residual = residual_adapter(z_context, actions, action_mask)
pred = base_pred + residual
loss = latent MSE + hard action counterfactual contrast
```

Prioritize SOAR and DROID for the first adapter because inverse dynamics confirms the latents contain action information. Keep mixed-large as a secondary source because its forward dynamics has good delayed temporal signal.

Residual adapter run started:

```text
container: sda-residual-action-adapter-v1
output: sensenova_drone_agent/output/residual_action_adapter_soar_droid_high_action_m1_v1
base dynamics: dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1/dynamics_ckpts/final_step_0275000.pt
sources: soar_native_v2,hf_robot_droid_lerobot_dreamer4
offset: -1
steps: 12000
contrast modes: shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse
```

Early minibatch ratios show strong temporal and zero-action separation but weak cross-trajectory shuffle separation. Held-out eval remains the gate; training ratios alone are not sufficient evidence of causal dynamics.

Added follow-up controls for the next run if needed:

```text
train_residual_action_adapter.py: --contrast-action-norm-weight, --contrast-latent-delta-weight, --contrast-weight-clip
WMDataset: --require_visual_delta / --require-visual-delta with threshold, min steps, stride
launch_residual_action_adapter.sh: reproducible Docker launcher for adapter runs
```

Visual-delta smoke test on SOAR `put-the-red-object-on-the-cloth` kept `486/498` high-action windows at threshold `0.01`, min visual-delta steps `8`, stride `4`, so the filter works and is not overly aggressive at that setting.

Residual adapter v1 result:

```text
output: sensenova_drone_agent/output/residual_action_adapter_soar_droid_high_action_m1_v1/residual_adapter_eval.json
direct normal: 0.370435
direct zero/normal: 0.3468
direct time_shift/normal: 0.9730
direct time_reverse/normal: 1.0783
AR normal: 3.203531
AR persistence: 0.017050
AR normal/persistence: 187.8958
AR shuffle/normal: 0.9996
AR zero/normal: 0.7923
AR time_shift/normal: 0.9938
```

Interpretation: the first residual adapter is not usable. It learned to separate temporal negatives during training but destroyed held-out dynamics. Root cause: it trained only at fixed shortcut signal level `0.1`, while eval and autoregressive rollout use other signal/step levels. The residual adapter had untrained signal embeddings for those levels, producing arbitrary residuals.

Fix applied:

```text
train_residual_action_adapter.py:
- zero-init step and signal embeddings
- add --residual-scale
- train next run with --random-signal

launch_residual_action_adapter.sh:
- supports RANDOM_SIGNAL and RESIDUAL_SCALE
```

Corrected residual adapter run started:

```text
container: sda-residual-action-adapter-randsig-v2
output: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2
random signal: yes
residual scale: 0.25
lr: 2e-4
contrast weight: 0.5
contrast action norm weight: 1.0
contrast latent delta weight: 2.0
```

Early logs are intentionally less aggressive than v1: reconstruction remains near the base range and negative ratios are modest. This is preferred because the adapter must preserve the simulator before improving action causality.

Intermediate held-out eval at step 5000:

```text
adapter: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_step_0005000.pt
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_step_0005000_eval64.json
batches: 64
direct normal: 0.004885
direct zero/normal: 13.9445
direct time_shift/normal: 35.8390
direct time_reverse/normal: 30.4131
AR normal: 0.012257
AR persistence: 0.020277
AR normal/persistence: 0.6045
AR zero/normal: 12.0338
AR time_shift/normal: 18.3432
AR time_reverse/normal: 18.2888
AR shuffle/normal: 0.9781
```

Interpretation: this is the first residual adapter checkpoint that both preserves rollout quality and produces strong temporal/zero-action causality on held-out data. It still fails the cross-trajectory shuffle gate. Treat random batch shuffle as a weak negative for robot data because different windows can contain similar end-effector motion; next strict-control pass should add a far-shuffle or action-distance-gated shuffle negative.

Added `far_shuffle` / `action_far_shuffle` / `distance_shuffle` negative controls:

```text
train_residual_action_adapter.py: action-distance-gated farthest batch permutation
eval_dreamer4_soar_dynamics.py: same farthest batch permutation for held-out eval
eval_residual_action_adapter.py: standalone checkpoint evaluator for adapter_step_*.pt
```

Step-5000 far-shuffle eval:

```text
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_step_0005000_farshuffle_eval64.json
direct far_shuffle/normal: 0.9960
AR far_shuffle/normal: 1.0040
AR normal/persistence: 0.5266
AR zero/normal: 17.5021
AR time_shift/normal: 26.8011
AR time_perm/normal: 28.0998
AR time_reverse/normal: 26.8503
```

Interpretation: the current adapter learns within-trajectory action timing and zero-action dependence, not cross-trajectory action identity. The next training variant should include `far_shuffle` in the contrast modes and likely increase cross-trajectory weight only after preserving the step-5000 rollout quality.

Corrected residual adapter final 12k eval:

```text
adapter: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_latest.pt
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/residual_adapter_eval.json
batches: 256
direct normal: 0.006566
direct shuffle/normal: 0.9970
direct zero/normal: 10.0100
direct time_shift/normal: 39.3163
direct time_reverse/normal: 27.1951
AR normal: 0.009484
AR persistence: 0.017050
AR normal/persistence: 0.5563
AR shuffle/normal: 1.0047
AR zero/normal: 16.1375
AR time_shift/normal: 26.9111
AR time_reverse/normal: 23.9551
```

Additional final far-shuffle eval:

```text
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_latest_farshuffle_eval256.json
batches: 256
direct far_shuffle/normal: 1.0056
AR normal: 0.007736
AR persistence: 0.015428
AR normal/persistence: 0.5015
AR far_shuffle/normal: 0.9946
AR zero/normal: 19.5875
AR time_shift/normal: 32.4956
AR time_perm/normal: 33.3575
AR time_reverse/normal: 28.9631
```

Interpretation: the corrected residual adapter is now a usable action-timing-conditioned latent simulator. It beats persistence and strongly depends on the provided temporal action sequence. It is not yet a strict action-identity-conditioned simulator because cross-trajectory action replacements are not reliably worse. This distinction matters for claims: safe claim is "retrofitted Kairos/Sensenova dynamics can be made sensitive to robot action timing and no-op/temporal counterfactuals"; unsafe claim is "fully action-identified simulator" until far-shuffle/source/task controls pass.

Far-shuffle training variant started:

```text
container: sda-residual-action-adapter-farshuffle-v1
output: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1
contrast modes: far_shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse
```

Purpose: determine whether cross-trajectory action identity can be enforced directly without losing the rollout-quality gains from the corrected adapter.

Far-shuffle training variant step-5000 eval:

```text
adapter: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/adapter_step_0005000.pt
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/adapter_step_0005000_eval128.json
batches: 128
direct far_shuffle/normal: 1.0029
direct zero/normal: 13.8400
direct time_shift/normal: 35.4390
AR normal: 0.006272
AR persistence: 0.013113
AR normal/persistence: 0.4783
AR far_shuffle/normal: 1.0038
AR zero/normal: 28.2990
AR time_shift/normal: 34.8389
AR time_perm/normal: 35.9037
AR time_reverse/normal: 30.9702
```

Interpretation: explicitly replacing random shuffle with far-shuffle does not solve cross-trajectory identity at step 5000. It does preserve and slightly improve rollout quality and temporal/no-op causality. This suggests cross-trajectory identity may be underdetermined by current observation/action windows rather than just a missing negative sampler.

Far-shuffle training variant final 12k eval:

```text
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/residual_adapter_eval.json
batches: 256
direct far_shuffle/normal: 1.0033
direct zero/normal: 10.3353
direct time_shift/normal: 55.9319
AR normal: 0.009192
AR persistence: 0.017050
AR normal/persistence: 0.5391
AR far_shuffle/normal: 1.0264
AR zero/normal: 15.4467
AR time_shift/normal: 43.1452
AR time_perm/normal: 40.5018
AR time_reverse/normal: 34.1941
```

Interpretation: far-shuffle training makes the autoregressive rollout pass the far-shuffle ratio gate (`1.0264 > 1.02`) while preserving rollout quality, but direct one-step far-shuffle remains weak (`1.0033`). The next variant should choose cross-trajectory negatives that differ in both action and observed future effect, not action alone.

Added `effect_far_shuffle` / `action_effect_shuffle` / `far_effect_shuffle` negative controls:

```text
train_residual_action_adapter.py: farthest donor by normalized action distance + normalized latent-transition distance
eval_dreamer4_soar_dynamics.py: same action+effect donor selection for held-out eval
```

Effect-far-shuffle training variant started:

```text
container: sda-residual-action-adapter-effect-farshuffle-v1
output: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1
contrast modes: effect_far_shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse
```

Purpose: test whether cross-trajectory identity becomes detectable when the wrong action window is selected from a trajectory with both different controls and different observed latent transition.

Effect-far-shuffle training variant final 12k eval:

```text
eval: sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/residual_adapter_eval.json
batches: 256
direct effect_far_shuffle/normal: 1.0030
direct zero/normal: 10.7635
direct time_shift/normal: 45.4376
AR normal: 0.009190
AR persistence: 0.017050
AR normal/persistence: 0.5390
AR effect_far_shuffle/normal: 1.0252
AR zero/normal: 22.4219
AR time_shift/normal: 33.6631
AR time_perm/normal: 33.3086
AR time_reverse/normal: 29.7049
```

Interpretation: action+effect donor selection preserves the far-shuffle rollout gate but does not fix one-step cross-trajectory sensitivity. Across all completed variants, the stable capability is autoregressive action-sequence controllability, especially no-op and temporal corruption sensitivity. Direct one-step action identity remains the main open gap.
