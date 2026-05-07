# BC/SFT Workflow

## Goal

Train a first supervised drone policy from real PX4 SITL trajectories instead of weak Kairos MPC teacher labels.

## Environments

- SITL collection:
  `cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent`
  Run inside the Docker `tools` service so ROS 2, Gazebo bridge tooling, and MAVSDK are available together.
- BC training:
  Run on the host in a Python environment with `torch` installed.
  The current host `python3` resolves to the active `blackwell-prod` conda environment and is suitable for training.

## Container Note

The tools image is now pinned to `numpy<2` because ROS 2 Jazzy `cv_bridge` is built against NumPy 1.x.
If the image was built before that pin, rebuild it once:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
docker compose build tools
```

## 1. Launch PX4 + Gazebo

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
./scripts/launch_px4_gazebo_x500_depth.sh --headless --world forest --pose 6,0,1.8,0,0,1.5708
```

Known camera topic:

```text
/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image
```

## 2. Collect a SITL episode

Recommended reactive-teacher collection:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
docker compose run --rm tools python3 scripts/collect_sitl_bc_episode.py \
  --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image \
  --depth-topic /depth_camera \
  --policy reactive_teacher \
  --num-steps 8 \
  --i-understand-this-is-sitl
```

Legacy scripted collection is still available with `--policy scripted --actions ...`, but it is no longer the recommended path because the labels are not observation-driven.

Episode output goes to:

```text
data/bc_sft/episodes/<episode_id>/
```

Each step contains:

- `frame_before.png`
- `frame_after.png`
- `step.json`

In reactive-teacher mode, `step.json` also contains:

- teacher decision reason
- goal-relative features
- depth clearance diagnostics

## 3. Export the supervised manifest

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
python3 scripts/export_bc_dataset.py \
  --episodes-root data/bc_sft/episodes \
  --out-jsonl data/bc_sft/manifests/bc_manifest.jsonl \
  --summary-json data/bc_sft/manifests/bc_manifest_summary.json
```

## 4. Train the baseline BC policy

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
python3 scripts/train_bc_policy.py \
  --manifest data/bc_sft/manifests/bc_manifest.jsonl \
  --out-dir output/bc_policy_baseline \
  --epochs 10 \
  --batch-size 16 \
  --image-size 224 \
  --device auto
```

Outputs:

- `output/bc_policy_baseline/best.pt`
- `output/bc_policy_baseline/last.pt`
- `output/bc_policy_baseline/train_summary.json`

## 5. Run the overnight collection + retraining loop

Keep PX4 SITL running in a separate terminal, then run:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
python3 scripts/run_overnight_bc_routine.py \
  --gazebo-topic /world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image \
  --depth-topic /depth_camera \
  --collector-policy reactive_teacher \
  --cycles 6 \
  --episodes-per-cycle 6 \
  --epochs-per-cycle 30 \
  --batch-size 8 \
  --image-size 128 \
  --device auto \
  --val-ratio 0.15 \
  --i-understand-this-is-sitl
```

This routine:

- checks MAVSDK connectivity,
- collects multiple observation-driven SITL episodes with the privileged reactive teacher,
- re-exports the manifest after each cycle,
- retrains the BC baseline after each cycle,
- writes logs under `logs/overnight_bc/`.

## Current Scope

- This is BC/SFT-first only.
- Kairos MPC distillation remains deferred until action-conditioned rollouts become meaningfully action-discriminative.
- The workflow has now been verified on a real collected dataset with both scripted and reactive-teacher episodes.
