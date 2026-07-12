# PyBullet Drone Feature Policy Benchmark

## Goal

Test the first external-benchmark version of:

```text
RGB observation
  -> frozen feature extractor
  -> learned action head
  -> gym-pybullet-drones velocity action
```

This is behavior cloning from a privileged teacher controller. It is not RL yet.

## Implemented

Script:

```text
sensenova_drone_agent/scripts/train_pybullet_drones_feature_policy.py
```

Docker wrapper:

```text
sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy.sh
```

Feature families:

```text
kinematic
rgb_downsample
random_projection
cnn_pixels
resnet18_imagenet
kairos_vae
kairos_vae_flat
```

Teacher:

```text
target_velocity controller toward [0, 0, 1]
```

## Important Benchmark Fix

Fixed-start hover is too easy. A nearly constant upward action can look good.

Use randomized starts for meaningful comparisons:

```text
--initial-xy-range 0.4
--initial-z-min 0.15
--initial-z-max 0.6
```

## Randomized-Start Validation

Command:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_feature_policy_randomized_non_kairos_v1 \
  --features kinematic,rgb_downsample,random_projection \
  --train-episodes 8 \
  --eval-episodes 2 \
  --max-steps 192 \
  --epochs 80 \
  --batch-size 64 \
  --eval-trace-frames 6 \
  --initial-xy-range 0.4 \
  --initial-z-min 0.15 \
  --initial-z-max 0.6
```

Result:

```text
kinematic:
  success_rate: 1.0
  mean_final_distance_m: 0.1171

rgb_downsample:
  success_rate: 0.0
  mean_final_distance_m: 0.2915

random_projection:
  success_rate: 0.0
  mean_final_distance_m: 0.3245
```

Interpretation:

```text
The randomized-start benchmark distinguishes real state signal from weak visual
or random features. It is a better paper/debug benchmark than fixed-start hover.
```

## Kairos VAE Short-Horizon Smoke

Command:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_feature_policy_randomized_kairos_v1 \
  --features kairos_vae \
  --train-episodes 2 \
  --eval-episodes 1 \
  --max-steps 72 \
  --epochs 60 \
  --batch-size 16 \
  --eval-trace-frames 6 \
  --initial-xy-range 0.4 \
  --initial-z-min 0.15 \
  --initial-z-max 0.6 \
  --kairos-device cpu \
  --kairos-dtype float32 \
  --kairos-height 128 \
  --kairos-width 128
```

Result:

```text
kairos_vae:
  success_rate: 0.0
  mean_final_distance_m: 0.3772
  best_val_mse: 0.0020
```

Matched 72-step non-Kairos comparison:

```text
kinematic:
  success_rate: 0.0
  mean_final_distance_m: 0.3603

random_projection:
  success_rate: 0.0
  mean_final_distance_m: 0.3711

rgb_downsample:
  success_rate: 0.0
  mean_final_distance_m: 0.3889
```

Interpretation:

```text
The Kairos/Wan VAE feature-policy path works end-to-end, but this pooled
32-dimensional first-frame VAE feature is not yet strong evidence of superior
control signal. It is roughly in the same range as weak visual/random baselines
on the short randomized smoke test.
```

## Stronger Baseline Comparison

Command:

```bash
./sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy.sh \
  --out-dir sensenova_drone_agent/output/pybullet_drones_feature_policy_strong_compare_v2 \
  --features kinematic,rgb_downsample,random_projection,cnn_pixels,resnet18_imagenet,kairos_vae,kairos_vae_flat \
  --train-episodes 2 \
  --eval-episodes 1 \
  --max-steps 72 \
  --epochs 60 \
  --batch-size 16 \
  --eval-trace-frames 6 \
  --initial-xy-range 0.4 \
  --initial-z-min 0.15 \
  --initial-z-max 0.6 \
  --kairos-device cpu \
  --kairos-dtype float32 \
  --kairos-height 128 \
  --kairos-width 128
```

Result:

```text
rgb_downsample:
  success_rate: 0.0
  mean_final_distance_m: 0.3410

random_projection:
  success_rate: 0.0
  mean_final_distance_m: 0.3721

kairos_vae_flat:
  success_rate: 0.0
  mean_final_distance_m: 0.3760

cnn_pixels:
  success_rate: 0.0
  mean_final_distance_m: 0.4061

kinematic:
  success_rate: 0.0
  mean_final_distance_m: 0.4206

resnet18_imagenet:
  success_rate: 0.0
  mean_final_distance_m: 0.4218

kairos_vae:
  success_rate: 0.0
  mean_final_distance_m: 0.4266
```

Interpretation:

```text
The flattened spatial Kairos/Wan VAE latent improves over pooled VAE channel
statistics, but it does not beat the best simple visual baseline in this short
smoke run. No 72-step method reaches the 0.15m success threshold, so this is a
ranking/debug run rather than a solved benchmark.
```

Output:

```text
sensenova_drone_agent/output/pybullet_drones_feature_policy_strong_compare_v2
```

## Longer Repeated-Seed Suite

Suite runner:

```text
sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy_suite.py
```

Command:

```bash
python3 sensenova_drone_agent/scripts/run_pybullet_drones_feature_policy_suite.py \
  --out-dir sensenova_drone_agent/output/pybullet_drones_feature_policy_power_suite_v1 \
  --features rgb_downsample,random_projection,cnn_pixels,resnet18_imagenet,kairos_vae_flat \
  --seeds 150000,151000 \
  --train-episodes 4 \
  --eval-episodes 2 \
  --max-steps 192 \
  --epochs 120 \
  --batch-size 32 \
  --initial-xy-range 0.4 \
  --initial-z-min 0.15 \
  --initial-z-max 0.6 \
  --kairos-device cpu \
  --kairos-dtype float32 \
  --kairos-height 128 \
  --kairos-width 128
```

Aggregate result:

```text
rgb_downsample:
  success_rate_mean: 0.25
  mean_final_distance_m_mean: 0.2982

resnet18_imagenet:
  success_rate_mean: 0.0
  mean_final_distance_m_mean: 0.2802

kairos_vae_flat:
  success_rate_mean: 0.0
  mean_final_distance_m_mean: 0.2868

cnn_pixels:
  success_rate_mean: 0.0
  mean_final_distance_m_mean: 0.3198

random_projection:
  success_rate_mean: 0.0
  mean_final_distance_m_mean: 0.3209
```

Interpretation:

```text
The longer suite produced the first non-zero success rate, from rgb_downsample.
By final-distance mean, ResNet18 and kairos_vae_flat are close, with ResNet18
slightly ahead. Kairos_vae_flat beats cnn_pixels and random_projection on final
distance, but does not yet beat the strongest simple/pretrained visual baselines.
```

Output:

```text
sensenova_drone_agent/output/pybullet_drones_feature_policy_power_suite_v1
```

## Current Claim Boundary

We can claim:

```text
Native Kairos/Wan VAE features can be extracted from external PyBullet drone RGB
observations and used by a learned policy head to produce closed-loop actions.
```

We cannot yet claim:

```text
Kairos/Sensenova controls drones better than baselines.
Kairos/Sensenova is a reliable drone policy.
Kairos rollouts are ready as MPC teachers.
```

## Next Work

Needed before this becomes paper-grade:

```text
1. Add DINO/CLIP frozen features.
2. Use more randomized-start episodes and at least 3 seeds.
3. Extract richer Kairos features than first-frame VAE latents.
4. Test temporal/stacked observations.
5. Compare closed-loop success, not just behavior-cloning MSE.
```
