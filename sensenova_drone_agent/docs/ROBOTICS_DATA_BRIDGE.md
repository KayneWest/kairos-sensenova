# Robotics Data Bridge

## Why

The Dreamer4 paper's robotics experiment uses real robot videos with actions to test whether a world model can learn action-conditioned physical dynamics. This is a better next gate than further tuning the current PyBullet waypoint toy setup.

The immediate goal is not to train a drone actor from robot-arm actions. A tabletop robot dataset has a different embodiment and action space. The useful first experiment is:

```text
real robot frames + robot actions -> action-conditioned latent dynamics
```

Then evaluate:

```text
normal action-conditioned prediction vs action-shuffled prediction
```

If shuffling actions does not hurt prediction, the world model has not learned meaningful action grounding.

## Candidate Datasets

```text
SOAR Robotics
  Source in Dreamer4 paper appendix.
  Real tabletop robot data.
  Approximate paper description: 256x256 video, 5 FPS, 7D relative end-effector actions.
  Public SOAR-Data is released by rail-berkeley/soar in RLDS and raw numpy formats.
  The upstream README reports the RLDS format requires about 136 GB of disk.
  Best for testing real-robot action conditioning.

nicklashansen/dreamer4
  Hugging Face dataset used by the local dreamer4 reproduction.
  DMControl/MMBench-style continuous-control tasks.
  Easier to load with dreamer4/dreamer4/wm_dataset.py.
  Not the SOAR tabletop dataset from the paper.

RoboNet TFDS sample
  Source: TensorFlow Datasets `robonet/robonet_sample_64`
  https://tensorflow.google.cn/datasets/catalog/robonet
  700 train trajectories.
  64x64 RGB video, variable-length 5D actions, 5D states.
  TFDS reports 119.80 MiB download and 183.04 MiB prepared dataset size.
  Useful for the action-identifiability gate because it contains robot-object interaction videos
  paired with end-effector delta/gripper actions.
```

## Required Schema

Any dataset we use needs to expose or let us derive:

```text
obs[t]       image or video frame
action[t]    continuous/discrete robot action
reward[t]    scalar reward, success label, or hindsight score
episode[t]   trajectory id
step[t]      timestep within trajectory
```

For SOAR-style data, reward can start as:

```text
success label at final step
hindsight goal-reaching score
zero except final success
```

For the first world-model gate, reward is optional. Actions and frame sequences are required.

## Inspection Tool

Use:

```bash
python3 sensenova_drone_agent/scripts/experiments/inspect_robotics_dataset.py \
  --dataset-id <huggingface/dataset-id> \
  --out sensenova_drone_agent/logs/robotics_data/inspection
```

Optional small sample download:

```bash
python3 sensenova_drone_agent/scripts/experiments/inspect_robotics_dataset.py \
  --dataset-id <huggingface/dataset-id> \
  --download-sample \
  --max-sample-files 32 \
  --max-sample-file-mb 64 \
  --out sensenova_drone_agent/logs/robotics_data/inspection
```

The tool writes:

```text
logs/robotics_data/inspection/<dataset>_manifest.json
logs/robotics_data/inspection/<dataset>_report.md
```

If using SOAR-Data from the upstream scripts instead of Hugging Face, download it first and then inspect the local directory:

```bash
python3 sensenova_drone_agent/scripts/experiments/inspect_robotics_dataset.py \
  --local-dir <path-to-soar-data> \
  --out sensenova_drone_agent/logs/robotics_data/inspection
```

## Resumable SOAR Numpy Download

For the 25 GB numpy zip:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py --dry-run
```

Start or resume the download:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py
```

Default output:

```text
sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip
```

The downloader writes:

```text
sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip.part
sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip.download.json
sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip.lock
```

Rerunning the same command resumes from the `.part` file. If the process died and left a stale lock:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py --force-lock
```

Recommended long-running session:

```bash
tmux new -s soar-download
cd /home/mkrzus/kairos-sensenova
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py
```

Optional extraction after download:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --extract \
  --extract-dir sensenova_drone_agent/data/robotics/soar/numpy
```

Verified local download:

```text
path: sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip
size: 25.31 GiB
zip entries: 347703
first entry: soar-dataset-local/
```

Extraction is not required for the current smoke path.

## Dreamer4 Hugging Face Dataset Download

The same downloader also supports the Hugging Face dataset used by the local Dreamer4 reproduction:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source dreamer4-hf \
  --dry-run
```

Default dataset:

```text
nicklashansen/dreamer4
```

Default output:

```text
sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4
```

Start or resume the download:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source dreamer4-hf
```

The downloader uses Hugging Face snapshot download with a size guard. To download a smaller subset:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source dreamer4-hf \
  --hf-splits expert,mixed-small
```

Verified local download:

```text
path: sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4
dataset: nicklashansen/dreamer4
splits: expert, mixed-small, mixed-large
remote size: 28.79 GiB
local size: 28.79 GiB
inspection: dreamer4_preprocess_ready
```

This is the dataset to use first for validating that our local Dreamer4 action-token dynamics can pass a normal-vs-shuffled action gate on data known to support action-conditioned world-model training.

## RoboNet TFDS Download

RoboNet is now wired into the same dataset downloader:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source robonet-tfds \
  --dry-run
```

Default TFDS dataset:

```text
robonet/robonet_sample_64
```

Default TFDS output:

```text
sensenova_drone_agent/data/robotics/robonet/tfds
```

Start or resume the TFDS preparation:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source robonet-tfds
```

If the local Python environment does not have TensorFlow Datasets, use the Dreamer data image after rebuilding it:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
docker build -f docker/Dockerfile.dreamer -t sensenova_drone_agent-dreamer:local ..

docker run --rm -it --gpus all \
  -v /home/mkrzus/kairos-sensenova:/workspace \
  -w /workspace \
  sensenova_drone_agent-dreamer:local \
  python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
    --source robonet-tfds
```

Observed caveat:

```text
Direct TFDS preparation can fail when Google Drive returns a small confirmation page instead
of the RoboNet tarball. In that case, use the gdown-backed archive source below.
```

Validated fallback download:

```bash
python3 sensenova_drone_agent/scripts/download_soar_numpy_dataset.py \
  --source robonet-gdrive
```

Default archive output:

```text
sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz
```

Verified local fallback archive:

```text
path: sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz
size: 119.80 MiB
sha256: 33367bb81c85a98630d0610c425d9cb33dc1652be57c98ce0ac239d12168d671
archive trajectories: 700
```

## RoboNet Dreamer4 Export

After TFDS preparation, export RoboNet into the local Dreamer4/WMDataset layout:

```bash
python3 sensenova_drone_agent/scripts/export_robonet_dreamer4_dataset.py \
  --source tfds \
  --tfds-data-dir sensenova_drone_agent/data/robotics/robonet/tfds \
  --out sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64 \
  --max-trajectories 700 \
  --frame-size 128 \
  --frame-stride 1
```

Validated archive export:

```bash
python3 sensenova_drone_agent/scripts/export_robonet_dreamer4_dataset.py \
  --source tar \
  --tar sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz \
  --out sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64 \
  --max-trajectories 700 \
  --frame-size 128 \
  --frame-stride 1 \
  --task-mode robot_name
```

The exporter writes a native Dreamer4/WMDataset-style tree:

```text
sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64/
  raw/
    berkeley_sawyer.pt
    ...
  frames/
    berkeley_sawyer/
      frames_shard_00000.pt
      ...
  tasks.json
  summary.json
```

Validated full export:

```text
exported trajectories: 700
task grouping: robot_name
action_dim: 5
tasks:
  berkeley_sawyer: 3100 steps, 100 episodes
  berkeley_widowx: 3100 steps, 100 episodes
  google: 2000 steps, 100 episodes
  penn_baxter_left: 1860 steps, 60 episodes
  penn_baxter_right: 1240 steps, 40 episodes
  penn_kuka: 3100 steps, 100 episodes
  stanford_fetch: 1500 steps, 100 episodes
  stanford_franka: 1922 steps, 62 episodes
  stanford_franka_corr_noise: 1178 steps, 38 episodes
```

Native WMDataset load check:

```text
seq_len: 16
img_size: 128
action_features: current,prev,delta,mean4,norm
expanded action_dim: 21
valid sequences: 7900
loaded tasks: 8
sample obs shape: [17, 3, 128, 128]
sample action shape: [16, 21]
```

`stanford_fetch` is present in the export but skipped by this specific `seq_len=16` check because
its episodes decode to 15 frames.

This can be used as an additional raw/frame source for the native Dreamer4-style training scripts.
The immediate gate is not reward learning; it is:

```text
normal action rollout MSE < shuffled action rollout MSE
normal action rollout MSE < zero action rollout MSE
```

If RoboNet passes that action-conditioning gate more cleanly than the current SOAR/Kairos cache,
it becomes the preferred dataset for enforcing action usage before imagination-RL tuning.

## Zip-Native SOAR Cache Export

The current exporter reads the SOAR ZIP directly:

```bash
python3 sensenova_drone_agent/scripts/export_soar_sequence_cache.py \
  --out sensenova_drone_agent/output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz \
  --summary-json sensenova_drone_agent/output/soar_sequence_cache_smoke/summary.json \
  --max-trajectories 6 \
  --max-steps-per-trajectory 32 \
  --frame-size 32 \
  --seed 7
```

Verified smoke output:

```text
trajectory count discovered: 31812
selected trajectories: 6
exported steps: 192
feature: rgb_flat
z_dim: 3072
action_dim: 7
episodes: 6
tasks: 5
valid anchors, context=8/horizon=8: 96
```

The exporter uses `ffmpeg` to extract frames from each trajectory MP4 into a temporary directory.
Supported feature paths:

```text
rgb_flat          cheap placeholder: resized RGB flattened into z
kairos_vae        Wan VAE latent pooled into channel mean/std features
kairos_vae_flat   Wan VAE latent flattened into z
```

## Experiment Plan

1. Inspect the exact dataset layout.
2. Implement the smallest schema adapter for that layout.
3. Export a small sequence cache with frames, actions, episode ids, and steps.
4. Encode frames through the same visual paths used by the drone scaffold:
   - RGB downsample baseline
   - Kairos/Wan VAE-flat features
5. Train sequence action-token dynamics only.
6. Measure:
   - next-latent MSE
   - multi-step rollout MSE
   - action_shuffle_loss_ratio
   - sequence_action_shuffle_loss_ratio
   - contact-sheet visual rollout quality
7. Train phase-2 behavior-cloning heads using `scripts/train_behavior_cloning_midtraining.py`.
8. Only after action grounding and BC midtraining are credible, run imagination RL.

## Current Smoke Result

Completed:

```text
SOAR download: true
SOAR zip-native inspection/export: true
SOAR RGB-flat sequence cache: true
SOAR BC midtraining smoke train: true
```

BC smoke training:

```text
cache: output/soar_sequence_cache_smoke/soar_rgb32_6traj.npz
train output: output/soar_bc_midtraining_smoke/train
valid anchors, context=8/horizon=4: 120
epoch 1 val loss: 0.8057
epoch 2 val loss: 0.7646
epoch 3 val loss: 0.7188
```

Medium RGB-flat scaling run:

```text
cache: output/soar_sequence_cache_medium/soar_rgb32_32traj.npz
train output: output/soar_bc_midtraining_medium_rgb32/train
trajectories: 32
steps: 1024
tasks: 23
valid anchors, context=8/horizon=4: 640
best epoch: 3
best val loss: 1.3315
best val action MSE: 0.6572
last epoch val action MSE: 0.6198
```

Kairos/Wan VAE-flat smoke run:

```text
cache: output/soar_sequence_cache_kairos_smoke/soar_kairos_flat128_2traj16.npz
train output: output/soar_bc_midtraining_kairos_smoke/train
trajectories: 2
steps: 32
feature: kairos_vae_flat
latent shape per frame: [1, 16, 1, 16, 16]
z_dim: 4096
action_dim: 7
valid anchors, context=8/horizon=4: 8
epoch 1 val loss: 0.4646
epoch 2 val loss: 0.3994
```

Kairos/Wan VAE-flat medium midtraining run:

```text
cache: output/soar_sequence_cache_kairos_medium/soar_kairos_flat128_32traj32.npz
train output: output/soar_bc_midtraining_kairos_medium/train
trajectories: 32
steps: 1002
tasks: 20
z_dim: 4096
action_dim: 7
valid anchors, context=8/horizon=4: 620
train anchors: 558
val anchors: 62
best epoch: 25
epoch 1 val loss: 1.1302
epoch 1 val action MSE: 0.7620
epoch 25 val loss: 0.8486
epoch 25 val action MSE: 0.4698
```

Control baselines:

```text
report: output/soar_bc_midtraining_kairos_medium_baselines/report.md
normal best val action MSE: 0.4698
shuffle_targets best val action MSE: 0.8158
shuffle_z_context best val action MSE: 0.6073
zero_z_context best val action MSE: 0.5440
zero_prev_actions best val action MSE: 0.5525
mean-action val action MSE: 0.8265
repeat-last-action val action MSE: 0.9443
```

Interpretation:

```text
Aligned Kairos/Wan VAE latents carry useful signal for SOAR action prediction, but the current
reward/value heads are not yet a strong basis for imagination RL.
```

SOAR success/failure reward labels:

```text
success.txt and language_task.txt are present per trajectory.
trajectory_success reward mode:
  best val action MSE: 0.4281
  best val reward MSE: 0.0015
  best val value MSE: 0.0041
linear_success_progress reward mode:
  best val action MSE: 0.4831
  best val reward MSE: 0.0048
  best val value MSE: 0.0661
final_success sparse reward mode:
  best val action MSE: 0.4698
  best val reward MSE: 0.1951
  best val value MSE: 0.7770
```

Recommended reward mode for the next phase-2 run:

```text
trajectory_success
```

Scaled trajectory-success run:

```text
cache: output/soar_sequence_cache_kairos_large/soar_kairos_flat128_128traj32_trajectory_success.npz
train output: output/soar_bc_midtraining_kairos_large_trajectory_success/train
checkpoint: output/soar_bc_midtraining_kairos_large_trajectory_success/train/best.pt
episodes: 128
steps: 4096
tasks: 60
valid anchors: 2560
train anchors: 2304
val anchors: 256
best epoch: 29
best val action MSE: 0.4460
best val reward MSE: 0.0012
best val value MSE: 0.0072
```

Remaining before the SOAR result can support a world-model-control claim:

```text
1. Run control baselines on the 128-trajectory trajectory_success cache.
2. Show the learned policy/reward/value heads improve closed-loop or imagined rollouts.
```

## Claim Boundary

A successful SOAR-style run would support:

```text
Kairos-derived visual latents can support action-conditioned world-model learning on real robot video.
```

It would not yet prove:

```text
Kairos controls drones.
```

The drone-control claim still requires closing the loop back into drone simulation or PX4/Gazebo with real drone observations overwriting imagined state every step.
