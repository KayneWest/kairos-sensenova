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
