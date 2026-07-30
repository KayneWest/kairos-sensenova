# DreamerV3 With Sensenova World-Model Features

## Goal

Test this hybrid:

```text
Gym drone RGB frame
  -> frozen Sensenova action-conditioned world-model encoder
  -> wm_latent vector
  -> DreamerV3 RSSM
  -> DreamerV3 actor/critic policy
  -> Gym drone action
```

This is not a direct policy-head swap. DreamerV3's policy consumes its RSSM
state, so the clean integration is to swap the observation encoder input:

```text
Dreamer sees wm_latent + goal features instead of raw pixels.
```

## Files

```text
dreamerv3/embodied/envs/sensenova_drone.py
dreamerv3/dreamerv3/main.py
dreamerv3/dreamerv3/configs.yaml
sensenova_drone_agent/docker/Dockerfile.dreamer
sensenova_drone_agent/scripts/run_dreamer_world_model_latent.sh
```

## Run

```bash
cd /home/mkrzus/kairos-sensenova

chmod +x sensenova_drone_agent/scripts/run_dreamer_world_model_latent.sh

DREAMER_STEPS=20000 \
./sensenova_drone_agent/scripts/run_dreamer_world_model_latent.sh \
  sensenova_drone_agent/output/dreamer_world_model_latent_debug
```

The script builds a separate Docker image:

```text
sensenova_drone_agent-dreamer:local
```

This keeps Dreamer/JAX dependencies separate from the PyTorch drone-sim tooling.

## What This Tests

This compares against:

```text
world_model_dqn:
  frozen world-model encoder -> DQN

dreamer_world_model_latent:
  frozen world-model encoder -> Dreamer RSSM -> Dreamer policy
```

If Dreamer improves over DQN with the same frozen latent, that suggests:

```text
the world-model representation supports stronger temporal decision learning
when paired with a proper latent dynamics RL algorithm.
```

If Dreamer fails, likely causes are:

```text
- too few environment steps
- Dreamer CPU config too small
- latent observation lacks enough safety information
- action masking/shielding needs better integration
```

## Current Status

Implemented and smoke-tested.

Verified:

```text
- Dreamer Docker image builds with CPU-only JAX + CPU-only PyTorch.
- SensenovaDreamer env reset/step returns wm_latent shape (128,) and goal shape (4,).
- DreamerV3 initializes RSSM/encoder/decoder/actor/critic from wm_latent observations.
- A 200-step CPU smoke run wrote metrics and a checkpoint.
```

Smoke output:

```text
sensenova_drone_agent/output/dreamer_world_model_latent_smoke/
```

Latest smoke metrics:

```text
step 81:  episode/score=-3.8485, episode/length=81
step 125: episode/score=-8.6874, episode/length=44
```

Longer CPU probes:

```text
2k run:
  output: sensenova_drone_agent/output/dreamer_world_model_latent_2k/
  episodes: 33
  best score: -0.4577 at step 426
  last-5 mean score: -6.6773
  wm_latent loss: 49.4563 -> 8.1407
  train/rand/action: ~1.0 -> ~1.0

10k run:
  output: sensenova_drone_agent/output/dreamer_world_model_latent_10k/
  episodes: 655
  best score: 2.0094 at step 3220
  last-5 mean score: -7.6202
  wm_latent loss: 49.3546 -> 0.2771
  goal loss: 1.3858 -> 0.0901
  reward loss: 5.5365 -> 2.0030
  train/rand/action: ~1.0 -> 0.2078
```

Interpretation:

```text
- The Dreamer integration is real: RSSM/model learning is happening on top of
  frozen Sensenova world-model features.
- The 10k policy is not good yet. As random exploration decays, it collapses
  toward short negative episodes instead of reliable goal-reaching.
- This points to a policy/reward/exploration issue, not an integration failure.
```

Use the dedicated Dreamer Dockerfile/script above for longer runs because the
existing drone tools Docker image does not include Dreamer dependencies.
