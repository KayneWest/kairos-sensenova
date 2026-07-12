# Gym Drone Game Results

## What Was Tested

We added a lightweight first-person drone navigation game to test the core loop without PX4 startup cost:

```text
real simulated observation -> movement action -> environment update -> next real observation
```

This environment emits RGB frames and compact state features. The DQN policy currently trains from state features. The image BC policies train from RGB frames exported from DQN rollouts.

## State-Based RL Result

Run:

```text
output/gym_drone_game_dqn_overnight_20260509T032655Z
```

Training:

```text
100000 DQN steps
2555 completed train episodes
best training-eval success rate: 0.90625
```

Held-out eval:

```text
output/gym_drone_game_eval_overnight_best_256
```

Result over 256 held-out seeds:

```text
success_rate: 0.828125
collision_rate: 0.1171875
timeout_rate: 0.0625
mean_return: 26.9198
```

Same-seed baselines:

```text
heuristic success_rate: 0.4961
heuristic collision_rate: 0.0195
random success_rate: 0.0
random collision_rate: 0.1133
```

Interpretation:

```text
The RL scaffold works in the lightweight game.
The learned state policy beats the heuristic on success, but collides more.
```

## Pixel BC Distillation Result

### V1: Success-Only DQN Episodes

Dataset:

```text
data/gym_drone_game_dqn_teacher_v1
146 successful episodes
5620 image/action examples
```

Image BC:

```text
output/bc_policy_gym_drone_game_dqn_teacher_v1
best val accuracy: 0.6812
```

Closed-loop pixel eval:

```text
output/gym_drone_game_bc_eval_dqn_teacher_v1_128
success_rate: 0.3047
collision_rate: 0.2734
timeout_rate: 0.4062
```

### V2: All DQN Episodes, Balanced Sampler

Dataset:

```text
data/gym_drone_game_dqn_teacher_v2_all
256 episodes
9840 image/action examples
211 success, 33 collision, 12 timeout
```

Image BC:

```text
output/bc_policy_gym_drone_game_dqn_teacher_v2_all
best val accuracy: 0.7018
```

Closed-loop pixel eval:

```text
output/gym_drone_game_bc_eval_dqn_teacher_v2_all_128
success_rate: 0.4219
collision_rate: 0.0938
timeout_rate: 0.4453
```

### V3: All DQN Episodes, Unbalanced Sampler

Image BC:

```text
output/bc_policy_gym_drone_game_dqn_teacher_v3_unbalanced
best val accuracy: 0.8119
```

Closed-loop pixel eval:

```text
output/gym_drone_game_bc_eval_dqn_teacher_v3_unbalanced_128
success_rate: 0.4922
collision_rate: 0.2656
timeout_rate: 0.2422
```

Interpretation:

```text
Pixel imitation works partially, but is not yet close to the state DQN.
Balanced sampling over-weighted rare hover/yaw actions and increased stalling.
Unbalanced training improved success but still overuses hover and collides too often.
```

## Current Conclusion

We are closer to the original goal:

```text
observation -> learned movement decision -> environment responds -> next observation
```

The state-based RL policy proves the drone-game task is learnable. The gap is now visual grounding:

```text
state DQN: strong in the game
RGB BC policy: partial, unstable in closed loop
```

## Pixel World-Model Signal

We then trained an explicit action-conditioned pixel world model:

```text
(frame_t, action_t) -> predicted frame_t+1
```

Run:

```text
output/gym_drone_game_world_model_v1
```

Dataset:

```text
data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl
8968 train transitions
872 validation transitions
```

Result:

```text
epoch 1 val loss: 0.016018
best val loss: 0.012364
epoch 1 val image_mse: 0.015676
best/late val image_mse: about 0.011960
```

Action sensitivity check:

```text
mean predicted-frame difference across actions: 0.0386 on [0, 1] pixels
```

Interpretation:

```text
The pixel world model is learning a real action-conditioned transition signal.
It is not merely producing identical next-frame predictions for all actions.
```

Artifacts:

```text
output/gym_drone_game_world_model_v1/index.html
output/gym_drone_game_world_model_v1/prediction_contact_sheet.png
output/gym_drone_game_world_model_v1/action_sensitivity.json
```

## Frozen World-Model Policy Probe

To test whether the learned world-model representation can support movement decisions, we froze the world-model encoder and trained only a small policy head:

```text
frame_t -> frozen world-model encoder -> latent z_t -> policy head -> action
```

Run:

```text
output/gym_drone_game_world_model_policy_v1
```

Offline action prediction:

```text
best val loss: 0.661153
best val accuracy: about 0.766
```

Closed-loop eval over 128 game episodes:

```text
success_rate: 0.375
collision_rate: 0.515625
timeout_rate: 0.046875
mean_return: 12.6117
```

Interpretation:

```text
The frozen world-model encoder contains useful action-relevant information,
but the resulting controller is not safe enough yet.
The failure mode is collision-heavy forward/strafe behavior.
```

## Matched-Seed Model Benchmark

Run:

```text
output/gym_drone_game_model_benchmark_v1
```

Command:

```text
PYTHONPATH=src:. python3 scripts/benchmark_gym_drone_game_models.py \
  --out-dir output/gym_drone_game_model_benchmark_v1 \
  --episodes 128 \
  --seed 900000 \
  --device cpu \
  --image-width 64 \
  --image-height 48 \
  --models random,heuristic,state_dqn,image_bc,world_model_policy
```

All models were evaluated on the same 128 environment seeds.

| Model | Success | Collision | Timeout | Out of Bounds | Mean Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_dqn | 0.8047 | 0.1406 | 0.0703 | 0.0000 | 25.8457 |
| image_bc | 0.4297 | 0.3203 | 0.2578 | 0.0000 | 16.8673 |
| heuristic | 0.3984 | 0.0078 | 0.5703 | 0.0234 | 13.2396 |
| world_model_policy | 0.2734 | 0.5859 | 0.0781 | 0.0859 | 8.4897 |
| random | 0.0000 | 0.7656 | 0.2188 | 0.0156 | -4.5721 |

Artifacts:

```text
output/gym_drone_game_model_benchmark_v1/index.html
output/gym_drone_game_model_benchmark_v1/summary.json
output/gym_drone_game_model_benchmark_v1/*_trace.png
output/gym_drone_game_model_benchmark_v1/*_episodes.jsonl
```

Interpretation:

```text
The state DQN is the strongest controller in this game.
The image BC policy transfers some of the DQN behavior into pixels, but loses a lot of closed-loop performance.
The heuristic is very safe, but stalls and times out often.
The frozen world-model policy shows learned decision signal, but is currently too collision-heavy.
Random is a useful lower bound and fails almost entirely.
```

The comparison supports the current technical claim:

```text
The system can learn decision behavior in the drone game.
Pixel policies and world-model representations contain usable signal.
The unresolved gap is not "can anything learn"; it is safe visual control from pixels.
```

## Risk-Aware Visual Policy

We added a risk-aware visual policy that trains from the same pixel manifest, but also learns auxiliary privileged labels:

```text
image + goal_features -> action
image + goal_features -> command
image + goal_features -> collision risk
image + goal_features -> stall risk
image + goal_features -> front clearance
image + goal_features -> progress
```

The policy still receives pixels at runtime. The privileged simulator state is used only as supervised labels.

Training run:

```text
output/gym_drone_game_risk_policy_v1
```

Training result:

```text
best val loss: 0.859110
val action accuracy: 0.780963
val collision-label accuracy: 0.994266
val stall-label accuracy: 0.818807
val clearance MAE: 1.7209 m
```

Important implementation fix:

```text
The clearance head is trained through sigmoid. Runtime inference must also use sigmoid.
Using raw clearance logits made the shield over-block forward motion.
```

Standalone closed-loop eval, no learned shield:

```text
output/gym_drone_game_risk_policy_eval_v1_128_no_shield
success_rate: 0.3984
collision_rate: 0.3828
mean_return: 12.9392
```

Standalone closed-loop eval with learned-clearance shield:

```text
output/gym_drone_game_risk_policy_eval_v1_128_fixed_shield_35
shield_front_clearance_m: 3.5
success_rate: 0.3594
collision_rate: 0.2422
mean_return: 11.0473
```

Matched-seed benchmark including risk policy:

```text
output/gym_drone_game_model_benchmark_v2_risk
```

| Model | Success | Collision | Timeout | Out of Bounds | Mean Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_dqn | 0.8047 | 0.1406 | 0.0703 | 0.0000 | 25.8457 |
| image_bc | 0.4297 | 0.3203 | 0.2578 | 0.0000 | 16.8673 |
| heuristic | 0.3984 | 0.0078 | 0.5703 | 0.0234 | 13.2396 |
| risk_visual_policy | 0.3750 | 0.2734 | 0.2891 | 0.0859 | 12.0108 |
| world_model_policy | 0.2734 | 0.5859 | 0.0781 | 0.0859 | 8.4897 |
| random | 0.0000 | 0.7656 | 0.2188 | 0.0156 | -4.5721 |

Interpretation:

```text
The risk-aware policy made safety controllable: collision dropped relative to image BC.
It did not improve overall task success yet.
The learned risk head is still weak because collision labels are highly imbalanced.
The learned clearance head is useful enough to create a safety/progress tradeoff.
```

Conclusion:

```text
This makes the next path concrete: add better hard-negative collision/near-miss data,
train action-conditioned risk estimates, and tune the safety shield against both
success and collision rather than action imitation alone.
```

## Action-Conditioned Risk Scorer

We then added branch-label generation:

```text
same real game frame I_t
  -> evaluate hover/yaw/forward/strafe branches from the exact same simulator state
  -> train image + candidate_action -> outcome scorer
```

Dataset v2:

```text
data/gym_drone_game_action_risk_v2
episodes: 192
states: 3958
candidate-action examples: 23748
near-miss branches: 7544
collision branches: 756
success branches: 70
```

The target utility was changed to penalize near-miss clearance heavily, not just terminal collision:

```text
utility = reward + success/progress/clearance bonuses
          - collision/out-of-bounds penalties
          - near-miss penalty
          - small non-progress penalty
```

Training run:

```text
output/gym_drone_game_action_risk_scorer_v2
```

Best validation metrics:

```text
best val loss: 0.685777
collision accuracy: 0.8278
success accuracy: 0.9825
clearance MAE: 1.7188 m
```

Planner eval:

```text
output/gym_drone_game_action_risk_planner_eval_v2_128
success_rate: 0.0000
collision_rate: 0.0156
timeout_rate: 0.9141
mean_return: -17.4197
```

Matched benchmark with action-risk planner:

```text
output/gym_drone_game_model_benchmark_v3_action_risk
```

| Model | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| state_dqn | 0.8047 | 0.1406 | 0.0703 | 25.8457 |
| image_bc | 0.4297 | 0.3203 | 0.2578 | 16.8673 |
| heuristic | 0.3984 | 0.0078 | 0.5703 | 13.2396 |
| risk_visual_policy | 0.3750 | 0.2734 | 0.2891 | 12.0108 |
| action_risk_planner | 0.0000 | 0.0234 | 0.9141 | -18.1830 |
| random | 0.0000 | 0.7656 | 0.2188 | -4.5721 |

Interpretation:

```text
The action-risk scorer learned risk avoidance, but not useful closed-loop progress.
It chooses yaw/avoidance repeatedly and times out.
One-step action risk is insufficient as a controller by itself.
```

This result is useful because it separates two problems:

```text
Risk perception is becoming learnable from pixels.
Multi-step recovery/progress after avoidance is still missing.
```

Next fix:

```text
Generate multi-step branch labels: action sequence -> outcome.
Train a short-horizon scorer over [forward], [strafe_left, forward], [strafe_right, forward], [yaw_left, forward], [yaw_right, forward].
Penalize repeated yaw/hover loops explicitly.
```

## World-Model Decision Heads

We then combined the pretrained world model with decision heads:

```text
world-model encoder features
  -> policy head from DQN teacher actions
  -> value head from step reward
  -> action-conditioned risk/utility heads from branch labels
```

Unweighted run:

```text
output/gym_drone_game_world_model_decision_heads_v1
val policy accuracy: 0.7846
val collision accuracy: 0.9741
val clearance MAE: 1.3313 m
```

Closed-loop eval:

```text
output/gym_drone_game_world_model_decision_heads_eval_v1_128
success_rate: 0.1094
collision_rate: 0.4609
out_of_bounds_rate: 0.3047
mean_return: 3.1154
```

Weighted-risk run:

```text
output/gym_drone_game_world_model_decision_heads_v2_weighted
val policy accuracy: 0.7404
val collision accuracy: 0.8055
val clearance MAE: 1.4784 m
```

Closed-loop eval:

```text
output/gym_drone_game_world_model_decision_heads_eval_v2_weighted_128
success_rate: 0.0234
collision_rate: 0.0469
timeout_rate: 0.8672
mean_return: -5.9558
```

Matched benchmark:

```text
output/gym_drone_game_model_benchmark_v4_world_model_decision
```

| Model | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| state_dqn | 0.8047 | 0.1406 | 0.0703 | 25.8457 |
| image_bc | 0.4297 | 0.3203 | 0.2578 | 16.8673 |
| heuristic | 0.3984 | 0.0078 | 0.5703 | 13.2396 |
| risk_visual_policy | 0.3750 | 0.2734 | 0.2891 | 12.0108 |
| world_model_decision_heads | 0.0078 | 0.0391 | 0.8750 | -7.2262 |
| random | 0.0000 | 0.7656 | 0.2188 | -4.5721 |

Interpretation:

```text
The world-model encoder supports offline decision/risk heads.
But one-step risk/utility planning still fails closed-loop.
Unweighted heads are too aggressive and collide.
Weighted heads are safer but stall.
```

Conclusion:

```text
The architecture is now correct for world-model pretraining + mid-training heads.
The missing ingredient is not the wiring; it is temporal supervision.
The next dataset should label short action sequences, not single actions.
```

## World-Model Feature DQN

We then tested the more standard RL path:

```text
frozen world-model encoder features + goal features -> DQN -> action
```

This differs from the previous world-model decision heads:

```text
Decision heads were supervised from offline labels.
World-model DQN directly optimizes the environment reward through replay.
```

Training run:

```text
output/gym_drone_game_world_model_dqn_v1
steps: 12000
completed train episodes: 333
best training-eval success rate: 0.6875
```

Final 16-episode training eval:

```text
success_rate: 0.6875
collision_rate: 0.3750
timeout_rate: 0.0000
mean_return: 19.9337
```

Matched 64-seed benchmark:

```text
output/gym_drone_game_model_benchmark_v5_world_model_dqn
```

| Model | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| state_dqn | 0.8750 | 0.0938 | 0.0469 | 27.6508 |
| world_model_dqn | 0.5938 | 0.3125 | 0.0938 | 19.0019 |
| image_bc | 0.5781 | 0.2344 | 0.1875 | 20.4227 |
| heuristic | 0.5312 | 0.0156 | 0.4375 | 17.5211 |
| random | 0.0000 | 0.7656 | 0.2031 | -4.8450 |

Interpretation:

```text
This is the strongest evidence so far that the pretrained world-model representation
can support reward-driven policy learning from pixels.
It beats random and heuristic on success, and slightly beats image BC on success.
It still trails privileged-state DQN and collides more than image BC/heuristic.
```

Conclusion:

```text
World-model pretraining + RL is viable in this toy drone game.
The next issue is not whether reward learning works; it is safety.
The next version should add constrained RL, a learned action shield, or a collision-cost term that is tuned against success.
```

## Safety-Constrained Follow-Up

We tested two safety paths on top of world-model DQN:

```text
1. Train-time reward shaping.
2. Runtime clearance shield that blocks FORWARD when front clearance is too low.
```

### Train-Time Safety Shaping

Strong shaping run:

```text
output/gym_drone_game_world_model_dqn_v2_safety_shaped
```

Shaping terms:

```text
extra collision penalty: 8.0
extra out-of-bounds penalty: 5.0
near obstacle threshold: 1.4 m
near obstacle penalty: 1.5
forward low-clearance threshold: 2.2 m
forward low-clearance penalty: 1.0
clearance recovery bonus: 0.2
```

Matched benchmark:

```text
output/gym_drone_game_model_benchmark_v6_world_model_dqn_safety
world_model_dqn success_rate: 0.4375
world_model_dqn collision_rate: 0.1094
world_model_dqn timeout_rate: 0.4531
mean_return: 13.1409
```

Interpretation:

```text
Reward shaping reduced collisions substantially, but made the policy too conservative.
It often timed out instead of committing to progress.
```

Mild shaping run:

```text
output/gym_drone_game_world_model_dqn_v3_mild_safety
```

Matched benchmark:

```text
output/gym_drone_game_model_benchmark_v7_world_model_dqn_mild_safety
world_model_dqn success_rate: 0.4531
world_model_dqn collision_rate: 0.2969
mean_return: 12.4511
```

Interpretation:

```text
Mild shaping did not find a better tradeoff.
It retained too much collision while still losing success.
```

### Runtime Clearance Shield

We then tested a final safety layer on the stronger unshaped world-model DQN:

```text
if front_clearance_m < threshold:
    block FORWARD
    choose the next-best DQN action
```

Matched 64-seed results:

| Model / Variant | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| world_model_dqn, no shield | 0.5938 | 0.3125 | 0.0938 | 19.0019 |
| world_model_dqn, shield 1.0m | 0.6250 | 0.2188 | 0.1719 | 19.6487 |
| world_model_dqn, shield 1.2m | 0.6094 | 0.1875 | 0.2188 | 19.1207 |
| world_model_dqn, shield 1.5m | 0.3750 | 0.1406 | 0.5000 | 9.8417 |

Interpretation:

```text
Runtime shielding is currently better than reward shaping.
It preserves most of the learned goal-seeking behavior while reducing collisions.
The 1.0m and 1.2m thresholds are useful operating points.
The 1.5m threshold is too conservative.
```

Current best practical controller:

```text
world_model_dqn_v1 + runtime front-clearance shield
```

Current recommendation:

```text
Use 1.0m shield for best success/return.
Use 1.2m shield when prioritizing lower collision.
Next train a policy with the shield in the loop, so it learns recovery actions instead of relying on post-hoc blocking.
```

### Shield-In-Loop Training

We then made the shield part of training itself:

```text
current observation
  -> compute front clearance
  -> mask FORWARD if clearance < 1.0m
  -> choose exploration/expert/greedy action from remaining actions
  -> store transition
  -> DQN target also masks unsafe next-state FORWARD
```

Training run:

```text
output/gym_drone_game_world_model_dqn_v4_shield_in_loop_10
steps: 12000
train shielded steps: 2409
best small-eval success: 0.6250
```

Matched benchmark with the same 1.0m runtime shield:

```text
output/gym_drone_game_model_benchmark_v11_world_model_dqn_shield_in_loop_10
```

| Model | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| world_model_dqn shield-in-loop 1.0m | 0.6719 | 0.2344 | 0.1094 | 21.0176 |
| image_bc | 0.5781 | 0.2344 | 0.1875 | 20.4227 |
| heuristic | 0.5312 | 0.0156 | 0.4375 | 17.5211 |

Same checkpoint with stricter 1.2m runtime shield:

```text
output/gym_drone_game_model_benchmark_v12_world_model_dqn_shield_in_loop_runtime_12
success_rate: 0.6406
collision_rate: 0.2031
timeout_rate: 0.1719
mean_return: 20.1913
```

Interpretation:

```text
Training with the shield in the loop improves the practical controller.
It beats image BC on success and mean return while matching image BC collision at 1.0m.
The 1.2m runtime threshold reduces collisions further with a modest success cost.
This is now the best visual/world-model policy path in the toy game.
```

Current best practical controller:

```text
world_model_dqn_v4_shield_in_loop_10 + runtime shield 1.0m
```

### Random-Encoder Ablation

To isolate whether pretrained world-model features matter, we trained the same
DQN setup with the same shield-in-loop behavior but replaced the pretrained
world-model encoder with a frozen random encoder of the same architecture:

```text
output/gym_drone_game_random_encoder_dqn_v1_shield_in_loop_10
```

Matched benchmark:

```text
output/gym_drone_game_model_benchmark_v13_random_encoder_dqn_shield_in_loop_10
```

| Variant | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: |
| pretrained world-model encoder DQN | 0.6719 | 0.2344 | 0.1094 | 21.0176 |
| frozen random encoder DQN | 0.5156 | 0.3750 | 0.1094 | 15.7519 |
| image BC | 0.5781 | 0.2344 | 0.1875 | 20.4227 |
| heuristic | 0.5312 | 0.0156 | 0.4375 | 17.5211 |

Interpretation:

```text
The pretrained world-model encoder beats the random frozen encoder under the same RL/shield setup.
That supports the claim that world-model pretraining is providing useful visual state features.
The random encoder can still learn some behavior because the policy receives goal features,
but it is less successful and substantially less safe.
```

Conservative conclusion:

```text
In this Gym drone game, pretrained predictive visual features are better than random visual features
for reward-driven closed-loop navigation.
This is not yet proof of drone transfer, but it is direct evidence that the world-model representation matters.
```

### Generic CNN DQN Baseline

We then added a stronger generic visual RL baseline:

```text
RGB frame + goal features
  -> residual CNN encoder trained from scratch
  -> DQN
  -> same 1.0m shield-in-loop safety mask
```

The CNN uses random-shift augmentation during replay updates. It is not pretrained
and does not use the world-model objective.

Training run:

```text
output/gym_drone_game_cnn_dqn_v1_shield_in_loop_10
```

This run was stopped at 6000 steps because end-to-end CNN training is much slower
than the frozen world-model encoder DQN. The best checkpoint from that partial run
was benchmarked on the same 64 matched seeds:

```text
output/gym_drone_game_model_benchmark_v14_cnn_vs_world_model_dqn
```

| Variant | Training Steps | Success | Collision | Timeout | Mean Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| pretrained world-model encoder DQN | 12000 | 0.6719 | 0.2344 | 0.1094 | 21.0176 |
| generic CNN DQN | 6000 | 0.3906 | 0.3281 | 0.2812 | 15.9655 |
| image BC | offline BC | 0.5781 | 0.2344 | 0.1875 | 20.4227 |
| heuristic | n/a | 0.5312 | 0.0156 | 0.4375 | 17.5211 |

Interpretation:

```text
The generic CNN DQN can learn some closed-loop behavior from reward.
Under a small training budget, it is clearly less sample-efficient than the pretrained world-model encoder DQN.
This supports a fidelity/sample-efficiency claim, not a claim that CNN DQN cannot solve the task with more training.
```

Conservative conclusion:

```text
World-model pretraining gives a better early representation for control than training a generic CNN from scratch.
The next stronger baseline would be a longer CNN run or a pretrained ImageNet/ResNet visual encoder.
```

## Next Experiments

Recommended next steps:

```text
1. Generate multi-step hard-negative recovery labels.
2. Train short-horizon action-sequence scoring, not one-step action scoring.
3. Penalize repeated yaw/hover loops in planner utility.
4. Train image RL with a collision-constrained reward or learned action shield.
5. Evaluate the best visual policy in PX4/Gazebo shadow mode only after Gym success exceeds roughly 70%.
```
