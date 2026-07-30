# Overnight BC Routine

## Goal

Run a safe overnight loop that:

1. collects more real PX4 SITL episodes from Gazebo,
2. re-exports the supervised manifest after each collection block,
3. retrains the baseline BC policy on the expanded dataset.

This routine does **not** use Kairos MPC teacher labels.

## Preconditions

- Launch PX4 SITL separately:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
./scripts/launch_px4_gazebo_x500_depth.sh --headless --world forest --pose 6,0,1.8,0,0,1.5708
```

- Ensure the `tools` image is rebuilt with the pinned NumPy version:

```bash
cd /home/mkrzus/kairos-sensenova/sensenova_drone_agent
docker compose build tools
```

## Recommended Overnight Command

Run this in another terminal while SITL is already up:

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

## What It Does

- Verifies MAVSDK can still reach local PX4 SITL.
- Collects short observation-driven SITL episodes with a privileged teacher:
  - RGB camera frames are recorded for the student.
  - depth + pose + sampled local waypoint are used by the teacher to choose actions.
  - goal features are written into `step.json` for goal-conditioned BC.
- Re-exports `data/bc_sft/manifests/bc_manifest.jsonl` after each cycle.
- Retrains `output/bc_policy_baseline` after each cycle.
- Writes per-cycle stdout/stderr logs under `logs/overnight_bc/<run_id>/`.

## Why This Is Worth Running Overnight

- It scales the real SITL dataset without depending on weak Kairos teacher labels.
- It makes the labels observation-driven instead of replaying a fixed scripted action list.
- It guarantees at least one validation episode once the episode count is large enough.
- It uses class-weighted action loss to reduce the impact of hover-heavy data.

## Current Limitation

- The routine still collects from one simulated world and one broad camera setup, so the resulting policy will remain narrow until collection diversity increases further.
- The student is still mostly single-frame + goal features; temporal context is the next model-side upgrade.
