# gym-pybullet-drones Benchmark

## Why This Benchmark

`gym-pybullet-drones` is a recognized quadrotor RL benchmark built on PyBullet and Gymnasium:

```text
https://github.com/learnsyslab/gym-pybullet-drones
```

It is less realistic than PX4/Gazebo, but much cheaper to run. It gives us a reviewer-recognizable bridge between the toy drone game and full SITL.

## Current Scope

Implemented benchmark:

```text
HoverAviary
RGB or kinematic observations
velocity action mode
headless PyBullet
custom success threshold: final distance <= 0.15m from hover target
```

Implemented policies:

- `target_velocity`: privileged sanity controller that flies toward the hover target.
- `zero_velocity`: no-op velocity command.
- `random`: random action baseline.
- `kairos_vae_probe`: extracts native Kairos/Wan VAE features from RGB observations while `target_velocity` controls.

`kairos_vae_probe` is not a control result yet. It is a feature-access proof on external benchmark frames.

## Build

```bash
cd /home/mkrzus/kairos-sensenova
./sensenova_drone_agent/scripts/build_pybullet_drones_benchmark_image.sh
```

This creates:

```text
sensenova_drone_agent-pybullet-drones:local
```

The image extends the existing local Dreamer image and installs `gym-pybullet-drones` without disturbing the working Kairos/PX4 images.

## Run Smoke Benchmark

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_benchmark.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_hover_v1 \
  --episodes 8 \
  --obs rgb \
  --policies target_velocity,zero_velocity,random
```

Outputs:

```text
summary.json
report.md
<policy>/episodes.jsonl
<policy>/contact_sheet.png
```

## Run Kairos Feature Probe

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_benchmark.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_hover_kairos_probe_v1 \
  --episodes 2 \
  --obs rgb \
  --policies kairos_vae_probe \
  --kairos-vae-probe \
  --kairos-device cpu \
  --kairos-dtype float32 \
  --kairos-height 128 \
  --kairos-width 128
```

Expected output:

```text
kairos_vae_probe/kairos_feature_records.json
```

## Paper Relevance

Useful claims if this benchmark works:

```text
We evaluate not only in a custom toy drone game, but also in a recognized PyBullet quadrotor benchmark.
```

Future claim path:

```text
RGB observation
  -> Kairos/Sensenova feature
  -> learned policy
  -> gym-pybullet-drones velocity action
```

Required before paper use:

- Train a learned policy using Kairos features.
- Compare against random-shaped Kairos features.
- Compare against CNN-from-scratch and generic pretrained visual encoders.
- Report matched seeds and confidence intervals.

## Learned Feature Policy Path

The first feature-policy benchmark is tracked in:

```text
sensenova_drone_agent/docs/PYBULLET_DRONES_FEATURE_POLICY.md
```

It adds:

```text
RGB observation -> frozen feature extractor -> learned action head -> PyBullet velocity action
```

Current result:

```text
Kairos/Wan VAE features can be used end-to-end by a learned action head.
Flattened spatial VAE latents improve over pooled channel stats, but they are
not yet stronger than the best simple/pretrained visual baselines on randomized-
start hover. In the longer two-seed suite, kairos_vae_flat beats cnn_pixels and
random_projection by final distance, but remains behind ResNet18 and the only
baseline with non-zero success, rgb_downsample.
```

## First Results

Smoke output:

```text
output/pybullet_drones_hover_v1
```

Run:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_benchmark.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_hover_v1 \
  --episodes 4 \
  --max-steps 240 \
  --obs rgb \
  --policies target_velocity,zero_velocity,random \
  --trace-frames 8
```

Result:

```text
target_velocity:
  success_rate: 1.0
  mean_final_distance_m: 0.1057
  mean_return: 378.1749

zero_velocity:
  success_rate: 0.0
  mean_final_distance_m: 0.8000
  mean_return: 308.5376

random:
  success_rate: 0.0
  mean_final_distance_m: 0.8270
  mean_return: 302.0491
```

Kairos feature probe output:

```text
output/pybullet_drones_hover_kairos_probe_v1
```

Run:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_benchmark.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_hover_kairos_probe_v1 \
  --episodes 1 \
  --max-steps 48 \
  --obs rgb \
  --policies kairos_vae_probe \
  --kairos-vae-probe \
  --kairos-probe-every-n 24 \
  --kairos-device cpu \
  --kairos-dtype float32 \
  --kairos-height 128 \
  --kairos-width 128 \
  --trace-frames 6
```

Feature result:

```text
backend: kairos_vae
latent_shape: [1, 16, 1, 16, 16]
feature_dim: 32
records: 2 frames
```

Interpretation:

```text
The external drone benchmark can provide RGB observations, and native Kairos/Wan
VAE features can be extracted from those observations. This is not a learned
Kairos control result yet.
```
