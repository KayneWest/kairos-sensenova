# Latent Imagination Planning Algorithm

This note is the implementation reference for the adjusted Kairos/Sensenova plan. The goal is no longer to prove that a frozen scene-motion prior is a perfect action-conditioned simulator. The goal is to use action-labeled data as training-time causal scaffolding, then learn a latent imagination space where the model can search over possible futures before choosing an executable action sequence.

## Core Claim

Kairos/Sensenova supplies visual world knowledge through its tokenizer and video prior. We add a planning layer that learns:

1. A latent plan token that summarizes action-labeled future intent.
2. A future proposer that maps current visual context plus a plan token to imagined future latents.
3. A trajectory scorer that predicts reward or task value for imagined futures.
4. An inverse dynamics head that maps selected imagined futures back to executable actions.
5. Causal checks that wrong action scaffolds should produce worse future predictions than correct action scaffolds.

At inference time, the system can run without user-provided action labels:

```text
video context
  -> sample/search latent plans
  -> imagine candidate futures
  -> score candidate futures
  -> select a future
  -> inverse dynamics converts selected future to actions
```

So the model still predicts likely futures, but it can first branch through a decision tree of possible futures and optimize over that thinking state.

## Training Procedure

### Phase 0: Frozen Visual Backbone

Use the trained Kairos/Sensenova/Dreamer4 tokenizer as a frozen latent encoder:

```text
frames x_t -> packed latent tokens z_t
```

The tokenizer is frozen so all downstream losses operate in a stable visual latent space.

### Phase 1: Plan-Supervised Future Prediction

For each action-labeled trajectory window:

```text
context latents: z_0 ... z_c
future actions: a_c ... a_{c+h}
future latents: z_{c+1} ... z_{c+h}
```

Train a plan encoder:

```text
u = plan_encoder(z_context, a_future)
```

Train a future proposer:

```text
z_hat_future = future_proposer(z_context, u)
```

Primary loss:

```text
L_future = MSE(z_hat_future, z_future)
```

This makes action labels act as steering supervision during training, while the generated object at inference is the plan token `u`, not a required action sequence.

### Phase 2: Action Causality Pressure

The model must not treat action labels as irrelevant metadata. For each batch, compare:

```text
u_true    = plan_encoder(z_context, a_future_true)
u_wrong   = plan_encoder(z_context, a_future_wrong)
z_true    = future_proposer(z_context, u_true)
z_wrong   = future_proposer(z_context, u_wrong)
```

where `a_future_wrong` can be shuffled, zeroed, time-shifted, reversed, or randomly permuted.

Contrastive hinge:

```text
L_contrast = max(0, margin + MSE(z_true, z_future) - MSE(z_wrong, z_future))
```

This is the minimum gate for action-identifiability. If wrong action scaffolds are not worse, the planner is still mostly following scene motion.

### Phase 3: Reward and Value Scoring

Train a trajectory scorer:

```text
score = trajectory_scorer(z_context, z_hat_future, u)
```

Target:

```text
R = sum_t gamma^t reward_t
```

Loss:

```text
L_reward = MSE(score, R)
```

For sparse datasets, this head may be weak at first. It is still useful as the interface required for later imagination RL.

### Phase 4: Inverse Dynamics

Train an inverse dynamics head:

```text
a_hat_future = inverse_dynamics(z_context, z_future_or_selected_future, u)
```

Loss:

```text
L_inverse = MSE(a_hat_future, a_future)
```

This gives the system a way to turn imagined futures back into actions.

### Phase 5: Inference-Time Imagination

Given a new context:

```python
plans = sample_plan_tokens(num_candidates)
futures = [future_proposer(context, u) for u in plans]
scores = [trajectory_scorer(context, future, u) for future, u in zip(futures, plans)]
best = argmax(scores)
actions = inverse_dynamics(context, futures[best], plans[best])
```

The selected future is the observable thinking state. We should log candidate scores, selected candidate index, future latent distances, and decoded previews when possible.

## Current Implementation Target

The first trainer implements:

```text
frozen tokenizer encode
plan encoder
future proposer
trajectory scorer
inverse dynamics head
shuffle/zero/time contrast
candidate planning eval
JSONL metrics and checkpoints
```

It does not yet update the base world model. This keeps the experiment clean:

```text
Can we learn a useful latent imagination planner on top of the frozen Kairos visual space?
```

If this works, the next step is continued world-model training with action tokens. If it does not work, the failure is still publishable as evidence that action-conditioned imagination requires native action-token dynamics rather than post-hoc retrofitting.

