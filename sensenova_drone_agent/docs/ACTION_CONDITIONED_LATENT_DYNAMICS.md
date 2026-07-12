# Action-Conditioned Latent Dynamics

## Purpose

This implements the missing Dreamer-style controllable simulator layer:

```text
z[t-C+1:t], a[t-C+1:t], candidate a[t:t+H-1], task -> z[t+1:t+H]
```

The frozen Kairos/Sensenova/Wan features are not updated. This head learns dynamics over those
features so a later imagination-RL stage can roll out candidate actions in latent space.

## Implementation

Core files:

```text
src/sensenova_drone/midtraining.py
scripts/train_action_conditioned_latent_dynamics.py
```

The model includes:

```text
latent context tokens
past action context tokens
future/candidate action tokens
task token
rollout token
small transformer encoder
future latent heads
```

This is phase-2/phase-2.5 infrastructure, not full imagination RL.

## Controls

The trainer supports direct action-grounding controls:

```text
normal
shuffle_future_actions
zero_future_actions
shuffle_z_context
zero_z_context
shuffle_targets
zero_action_context
```

A usable controllable simulator must beat persistence and must beat shuffled/zero future-action
controls on held-out episodes.

## Synthetic Action-Control Smoke

Cache:

```text
output/latent_dynamics_smoke/action_driven_cache.npz
```

Results:

```text
normal:
  best val z MSE: 0.1564
  persistence MSE: 0.5135
  ratio: 0.3046

shuffle_future_actions:
  best val z MSE: 0.3380
  persistence MSE: 0.5135
  ratio: 0.6581

zero_future_actions:
  best val z MSE: 0.3371
  persistence MSE: 0.5135
  ratio: 0.6564
```

Interpretation:

```text
The dynamics head can use future action tokens when action signal exists in the latent transition data.
```

## SOAR Kairos/Wan VAE-Flat Probe

Cache:

```text
output/soar_sequence_cache_kairos_task_balanced_512/soar_kairos_flat128_512traj32_trajectory_success.npz
```

Short probe:

```text
episodes: 512
tasks: 64
steps: 16384
z_dim: 4096
action_dim: 7
train anchors: 8960
val anchors: 1280
context: 8
horizon: 4
epochs: 6
```

Results:

```text
normal:
  best val z MSE: 0.141892
  persistence MSE: 0.141856
  ratio: 1.000256

shuffle_future_actions:
  best val z MSE: 0.141896
  persistence MSE: 0.141856
  ratio: 1.000281

zero_future_actions:
  best val z MSE: 0.141890
  persistence MSE: 0.141856
  ratio: 1.000242
```

Interpretation:

```text
The current SOAR Kairos/Wan VAE-flat one-frame latent cache is effectively persistence-dominated.
Normal, shuffled, and zero future actions are indistinguishable in this probe.
```

## SOAR Action-Offset Gate

Runner:

```text
scripts/run_action_conditioning_gate.py
```

Cache:

```text
output/soar_sequence_cache_kairos_task_balanced_512/soar_kairos_flat128_512traj32_trajectory_success.npz
```

Output:

```text
output/soar_action_conditioning_gate_kairos_task_balanced_512_offsets
```

Offsets tested:

```text
-2, -1, 0, 1, 2
```

Gate result:

```text
ready_for_bc_or_imagination: false
passed_offsets: []
best offset: -1
best normal/persistence ratio: 1.000273
best shuffle/normal ratio: 1.000047
best zero/normal ratio: 1.000012
```

Interpretation:

```text
No tested action/frame offset made future actions predictive. The model remains at persistence,
and shuffled or zero future actions perform the same as real future actions.
```

## RGB-Flat Sanity Check

Cache:

```text
output/soar_sequence_cache_medium/soar_rgb32_32traj.npz
```

Output:

```text
output/soar_action_conditioning_gate_rgb32_medium
```

Gate result:

```text
ready_for_bc_or_imagination: false
normal/persistence ratio: 1.019120
shuffle/normal ratio: 0.999323
zero/normal ratio: 0.997915
```

Interpretation:

```text
The quick RGB-flat control also fails. This suggests the current SOAR cache/window/action setup is
not action-predictive enough, not merely that Kairos/Wan VAE-flat latents are hiding the signal.
```

## Temporal Action Aggregation

Exporter support:

```text
scripts/export_soar_sequence_cache.py
--frame-stride N
--action-aggregation sample|mean|sum|last
```

For `mean`, `sum`, and `last`, the exported action is computed over the interval from the current
exported frame to the next exported frame. This fixes the earlier weak alignment where a widened
frame stride still used only `actions[frame_idx]`.

Stride-4 summed-action RGB gate:

```text
cache: output/soar_sequence_cache_rgb32_task_balanced_512_stride4_sum/soar_rgb32_512traj128_stride4_sum_trajectory_success.npz
output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride4_sum
frame_stride: 4
action_aggregation: sum
context: 8
horizon: 4
ready_for_bc_or_imagination: false
best_offset: 0
normal_vs_persistence_ratio: 0.991179
shuffle_vs_normal_ratio: 1.006619
zero_vs_normal_ratio: 1.007921
```

Stride-8 summed-action RGB gate:

```text
cache: output/soar_sequence_cache_rgb32_task_balanced_512_stride8_sum/soar_rgb32_512traj160_stride8_sum_trajectory_success.npz
output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride8_sum_ctx4_e30
frame_stride: 8
action_aggregation: sum
context: 4
horizon: 4
ready_for_bc_or_imagination: false
normal_vs_persistence_ratio: 0.918304
shuffle_vs_normal_ratio: 1.049823
zero_vs_normal_ratio: 1.040453
```

High-action/high-motion filtered stride-8 gate:

```text
output: output/soar_action_conditioning_gate_rgb32_task_balanced_512_stride8_sum_ctx4_motion_filter_e30
min_future_action_rms: 1.0
min_target_delta_rms: 0.10
kept_anchors: 1475 / 3935
ready_for_bc_or_imagination: false
normal_vs_persistence_ratio: 0.919450
shuffle_vs_normal_ratio: 1.011234
zero_vs_normal_ratio: 1.014176
```

Interpretation:

```text
Temporal aggregation and wider frame stride reveal real action signal. The best unfiltered stride-8
run beats persistence clearly and nearly beats the shuffled-action threshold, but zero future actions
remain too competitive. The cache is still not rigorous enough for imagination RL promotion.
```

## Decision

Implemented:

```text
action-conditioned latent dynamics scaffold
task-conditioned dynamics/agent heads
action-shuffle and zero-action controls
metric-specific checkpoints
```

Not yet proven:

```text
SOAR Kairos/Wan latents contain enough action-conditioned dynamics for imagination RL.
```

Recommended next step:

```text
Do not start full imagination RL from this SOAR latent probe yet. Next, add explicit robot state or
state deltas to the dynamics target/input, or use an environment/dataset where action labels are
more directly coupled to first-person visual consequences. Re-run the gate before treating the world
model as a controllable simulator.
```
