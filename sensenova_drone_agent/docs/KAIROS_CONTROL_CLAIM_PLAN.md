# Kairos/Sensenova Drone-Control Claim Plan

## Mission Claim

Target claim:

```text
Kairos/Sensenova itself participates in closed-loop drone control.
```

This is stronger than the current demonstrated claim:

```text
A learned drone-game world-model encoder supports closed-loop visual control.
```

To make the stronger claim, the actual Kairos/Sensenova model or one of its native learned components must be in the action-selection path.

## Non-Negotiable Evidence

The loop must be:

```text
real Gazebo/PX4 camera frame
  -> Kairos/Sensenova feature, latent, hidden state, rollout, or policy state
  -> selected high-level drone action
  -> MAVSDK/PX4 command
  -> Gazebo/PX4 moves
  -> next real Gazebo/PX4 camera frame
```

Generated Kairos frames may be used as temporary hypotheses, but must never become the next real state.

## Claim Levels

### Level 0: Not A Kairos Control Claim

```text
Toy world-model encoder -> policy -> drone-game action
```

This is what the best current result mostly supports.

### Level 1: Kairos Representation Controls A Toy Drone Game

```text
drone-game RGB frame
  -> Kairos/Sensenova VAE latent or image/text encoder feature
  -> policy head or DQN
  -> drone-game action
```

Minimum required before saying Kairos features help control.

### Level 2: Kairos Representation Controls PX4/Gazebo SITL

```text
PX4/Gazebo camera frame
  -> Kairos/Sensenova feature
  -> policy head
  -> MAVSDK command
  -> next PX4/Gazebo camera frame
```

Minimum required before saying Kairos participates in simulated drone control.

### Level 3: Kairos Rollouts Choose Actions

```text
real frame
  -> Kairos state
  -> candidate action-conditioned Kairos rollouts
  -> reward/safety score
  -> argmax action
  -> PX4/Gazebo command
```

This would support the stronger "Kairos as planner/world model" claim, but only if different candidate actions produce meaningfully different futures.

### Level 4: Kairos-Trained Policy Improves Through Imagination

```text
Kairos state/dynamics
  -> reward/value heads
  -> imagined trajectories
  -> policy improvement
  -> closed-loop PX4/Gazebo evaluation
```

This is closest to the Dreamer-style claim. It requires reliable action grounding and reward calibration.

## Required Experiments

### 1. Kairos Feature Extraction

Goal:

```text
real frame -> actual Kairos/Sensenova latent/feature
```

Acceptance:

- A real Gazebo/drone-game frame can be encoded by a native Kairos component.
- Feature shape, dtype, model path, and preprocessing are logged.
- Feature extraction is repeatable for the same image.
- Random-feature and generic-feature alternatives can be swapped in with the same policy code.

Initial route:

```text
Wan/Kairos VAE first-frame latent
```

Stronger later routes:

```text
Qwen-VL prompt/image embeddings
DiT hidden states
pooled decision token h_t
```

### 2. Kairos-Feature Policy In Toy Drone Game

Goal:

```text
drone-game frame -> Kairos feature -> policy -> action
```

Acceptance:

- Train DQN or BC policy on frozen Kairos features.
- Compare on matched seeds against:
  - random policy
  - heuristic policy
  - CNN from scratch
  - frozen random Kairos-shaped features
  - generic pretrained visual encoder
  - current lightweight world-model encoder
- Show Kairos features are better than random-shaped features.

### 3. Kairos-Feature Closed Loop In PX4/Gazebo

Goal:

```text
PX4/Gazebo camera -> Kairos feature -> policy -> MAVSDK command
```

Acceptance:

- Runs only in PX4 SITL.
- Uses high-level velocity/yaw commands only.
- Safety shield is final authority.
- Logs frame path, Kairos feature metadata, chosen action, shield decision, and executed command.
- Evaluates success/collision/timeout/clearance over repeated episodes or scripted tasks.

### 4. Kairos Action-Conditioning Gate

Goal:

```text
same real frame + different candidate actions -> different Kairos predicted futures
```

Acceptance:

- `hover`, `yaw_left`, `yaw_right`, and `forward` produce distinguishable rollouts.
- Motion metrics and contact sheets show action-conditioned differences.
- Rollouts are scoreable against goals and safety.
- If rollouts are near-static or indistinguishable, Kairos-MPC is not teacher-ready.

### 5. Kairos-MPC Or Policy-Head Runtime

Two possible routes:

MPC route:

```text
a_t = argmax_A R(Kairos rollout under candidate action sequence A)
```

Policy route:

```text
h_t = Kairos.encode_observation_and_memory(...)
a_t = policy_head(h_t)
```

Acceptance:

- The command comes from Kairos-conditioned state or Kairos-conditioned rollout scoring.
- Ablating Kairos features reduces performance.
- Safety shield remains final authority.

## Required Ablations

Representation ablations:

- Kairos/Sensenova feature.
- Frozen random feature with same shape.
- Current lightweight drone-game world-model feature.
- CNN trained from scratch.
- Generic pretrained image feature, such as ResNet, CLIP, or DINO.

Control ablations:

- No safety shield.
- Runtime safety shield.
- Shield-in-loop training.

Rollout ablations:

- Prompt-only Kairos rollout.
- Explicit camera-control fields.
- Synthetic action seed video fallback.
- Real action-conditioned rollout if available.

## Minimum Paper-Grade Evaluation

Toy game:

```text
256+ matched eval episodes
3+ training seeds for trainable methods
confidence intervals for success/collision/timeout
```

PX4/Gazebo:

```text
repeatable task suite
fixed worlds and seeds where possible
saved videos/contact sheets
MAVSDK command logs
camera frame logs
safety intervention logs
```

## What We Need To Build Next

Immediate implementation queue:

1. Add a Kairos feature extraction audit for real frames.
2. Add a frozen Kairos-feature policy path in the drone game.
3. Benchmark Kairos features against random-shaped features and current world-model features.
4. If toy-game results are positive, connect the same policy interface to PX4/Gazebo.
5. Only then revisit Kairos-MPC rollouts as a planner/teacher.

## Current Status

```text
KAIROS_CONTROL_CLAIM_READY=false
```

Reason:

```text
The actual Kairos/Sensenova feature or hidden state is not yet proven to drive closed-loop actions.
```

First milestone:

```text
KAIROS_FEATURE_EXTRACTION_READY=partial
```

Current feature audit:

```text
output/kairos_feature_audit_v1/feature_access_audit.json
```

Reproduce metadata audit from host Python:

```bash
python3 sensenova_drone_agent/scripts/extract_kairos_observation_features.py \
  --metadata-only \
  --out-dir sensenova_drone_agent/output/kairos_feature_audit_v1
```

Reproduce the first CPU VAE latent extraction in Docker:

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v "$PWD":/workspace \
  -w /workspace \
  sensenova_drone_agent-dreamer:local \
  python sensenova_drone_agent/scripts/extract_kairos_observation_features.py \
    --input-frame sensenova_drone_agent/sim_assets/sample_frames/gazebo_rgb_000001.png \
    --out-dir sensenova_drone_agent/output/kairos_feature_audit_v1_cpu128 \
    --device cpu \
    --dtype float32 \
    --height 128 \
    --width 128 \
    --no-tiled
```

Confirmed on this machine:

```text
Kairos/Wan VAE checkpoint exists.
Qwen-VL text/image encoder path exists.
Kairos DiT checkpoint exists on the host.
The configured Kairos DMD pipeline uses fuse_vae_embedding_in_latents=true.
```

First actual extracted feature:

```text
output/kairos_feature_audit_v1_cpu128/kairos_vae_feature_summary.json

input:
  sim_assets/sample_frames/gazebo_rgb_000001.png

feature:
  backend: kairos_vae
  latent_shape: [1, 16, 1, 16, 16]
  pooled_feature_dim: 32
  resolution: 128x128
  device: cpu
```

This proves we can extract a native Kairos/Wan VAE observation latent. It does
not yet prove the latent improves control.

Second milestone:

```text
KAIROS_FEATURE_TOY_CONTROL_READY=true
```

Third milestone:

```text
KAIROS_FEATURE_PX4_CONTROL_READY=true
```
